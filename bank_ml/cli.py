"""Command line interface orchestrating the ML workflow."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from . import data, preprocess, feature_select_ga, clustering, models, evaluate
from .config import Config, ensure_output_dirs, load_config


app = typer.Typer(help="Bank ML pipeline")


# ---------------------------------------------------------------------------
# Helper persistence utilities
# ---------------------------------------------------------------------------


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2)


def _load_json(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def fit(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Run the full training pipeline and persist all artefacts."""

    cfg = load_config(config)
    out_dir = ensure_output_dirs(cfg)

    # ------------------------------------------------------------------
    # Load data and split
    # ------------------------------------------------------------------
    X, y = data.load_dataset(cfg)
    X_train, X_test, y_train, y_test, cv = data.train_test_cv_split(X, y, cfg)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    preproc, feature_names = preprocess.build_preprocess(X_train)
    X_train_pre = preprocess.fit_transform_preprocess(preproc, X_train)
    preprocess.save_preprocess(preproc, out_dir / "preprocess.joblib")
    _save_json({"feature_names": feature_names}, out_dir / "feature_names.json")

    # ------------------------------------------------------------------
    # GA feature selection
    # ------------------------------------------------------------------
    ga_res = feature_select_ga.ga_select_features(
        X_train_pre, y_train.to_numpy(), feature_names, cv, cfg
    )
    mask = np.array(ga_res["mask"], dtype=bool)
    X_train_sel = X_train_pre[:, mask]
    _save_json(
        {"mask": mask.astype(int).tolist(), "selected_names": ga_res["selected_names"]},
        out_dir / "ga_selection.json",
    )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    cluster_res = clustering.select_k_and_cluster(X_train_sel, cfg.clustering.k_grid, cfg)
    labels = cluster_res["labels"]
    joblib.dump(cluster_res["model"], out_dir / "clustering.joblib")
    _save_json(
        {
            "k": int(cluster_res["k"]),
            "dbi_per_k": cluster_res["dbi_per_k"],
            "sil_per_k": cluster_res["sil_per_k"],
            "centers": cluster_res["centers"].tolist(),
        },
        out_dir / "clustering.json",
    )

    # ------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------
    model_res = models.train_models(X_train_sel, y_train.to_numpy(), labels, cv, cfg)
    # ``train_models`` already persists the individual model files and manifest

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    X_test_pre = preprocess.transform_preprocess(preproc, X_test)
    X_test_sel = X_test_pre[:, mask]
    cluster_model = cluster_res["model"]
    test_labels = cluster_model.predict(X_test_sel)
    X_test_aug = clustering.append_cluster_features(X_test_sel, test_labels)

    model_dict = {"mlp": model_res["mlp"]}
    model_dict.update(model_res["baselines"])
    evaluate.evaluate_models(model_dict, X_test_aug, y_test.to_numpy(), cfg)

    # Generate markdown report summarising the run
    _generate_report(cfg)

    logger.info("Training pipeline complete. Artefacts saved to %s", out_dir)


@app.command()
def predict(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    input: Path = typer.Option(..., "--input", "-i", exists=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Generate predictions for a CSV of new samples."""

    cfg = load_config(config)
    out_dir = Path(cfg.paths.output_dir)

    preproc = preprocess.load_preprocess(out_dir / "preprocess.joblib")
    ga_info = _load_json(out_dir / "ga_selection.json")
    mask = np.array(ga_info["mask"], dtype=bool)
    cluster_model = joblib.load(out_dir / "clustering.joblib")

    model = joblib.load(out_dir / "mlp.joblib")

    df = pd.read_csv(input)
    X = df.drop(columns=[cfg.label], errors="ignore")
    if cfg.id_column and cfg.id_column in X.columns:
        X = X.drop(columns=[cfg.id_column])

    X_pre = preprocess.transform_preprocess(preproc, X)
    X_sel = X_pre[:, mask]
    labels = cluster_model.predict(X_sel)
    X_aug = clustering.append_cluster_features(X_sel, labels)

    preds = model.predict(X_aug)
    pd.DataFrame({"prediction": preds}).to_csv(output, index=False)
    logger.info("Predictions written to %s", output)


@app.command()
def report(config: Path = typer.Option(..., "--config", "-c", exists=True)) -> None:
    """Generate a concise markdown report of the training run."""

    cfg = load_config(config)
    _generate_report(cfg)


def _generate_report(cfg: Config) -> None:
    """Internal helper to create a markdown report for a run."""

    out_dir = ensure_output_dirs(cfg)

    ga_info = (
        _load_json(out_dir / "ga_selection.json")
        if (out_dir / "ga_selection.json").exists()
        else {}
    )
    cluster_info = (
        _load_json(out_dir / "clustering.json")
        if (out_dir / "clustering.json").exists()
        else {}
    )
    manifest = (
        _load_json(out_dir / "manifest.json")
        if (out_dir / "manifest.json").exists()
        else {}
    )

    metrics_df = pd.DataFrame()
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, index_col=0)

    md_path = out_dir / "report.md"
    with md_path.open("w") as fh:
        fh.write("# Model Report\n\n")

        fh.write("## Selected Features\n")
        fh.write(", ".join(ga_info.get("selected_names", [])) + "\n\n")

        fh.write("## Clustering\n")
        if cluster_info:
            fh.write(f"k: {cluster_info.get('k')}\n\n")
        if "dbi_per_k" in cluster_info:
            fh.write(f"DBI per k: {cluster_info['dbi_per_k']}\n\n")
        if "sil_per_k" in cluster_info:
            fh.write(f"Silhouette per k: {cluster_info['sil_per_k']}\n\n")

        fh.write("## PSO Best Params\n")
        fh.write(json.dumps(manifest.get("mlp_params", {}), indent=2) + "\n\n")

        if not metrics_df.empty:
            fh.write("## Test Metrics\n")
            try:  # pandas.to_markdown requires optional tabulate dependency
                fh.write(metrics_df.to_markdown() + "\n")
            except Exception:  # pragma: no cover - fallback when tabulate missing
                fh.write(metrics_df.to_string() + "\n")

    logger.info("Report generated at %s", md_path)


if __name__ == "__main__":  # pragma: no cover
    app()

