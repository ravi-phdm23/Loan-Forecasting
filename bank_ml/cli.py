"""Command line interface for bank_ml."""
from __future__ import annotations
from pathlib import Path
import json

import joblib
import pandas as pd
import typer
from loguru import logger

from .config import load_config, ensure_output_dirs
from . import data, preprocess, models, evaluate

app = typer.Typer(help="Bank ML pipeline")


@app.command()
def fit(
    config: Path = typer.Option(..., "--config", "-c", exists=True)
) -> None:
    """Run the full pipeline and persist artifacts."""
    cfg = load_config(config)
    out_dir = ensure_output_dirs(cfg)
    X, y = data.load_csv(cfg.paths.input_csv, cfg.label, cfg.id_column)
    X_train, X_test, y_train, y_test = preprocess.train_test_split_data(
        X, y, cfg.cv.test_size, cfg.cv.random_state
    )
    X_train, y_train = preprocess.handle_imbalance(X_train, y_train, cfg.imbalance)
    model = models.build_model(cfg.imbalance)
    model.fit(X_train, y_train)
    joblib.dump(model, out_dir / "model.joblib")
    evaluate.evaluate_and_save(model, X_test, y_test, out_dir)
    logger.info("Training complete")


@app.command()
def predict(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    input: Path = typer.Option(..., "--input", "-i", exists=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Load persisted model and predict on new data."""
    cfg = load_config(config)
    model = joblib.load(Path(cfg.paths.output_dir) / "model.joblib")
    df = pd.read_csv(input)
    X = df.drop(columns=[cfg.label], errors="ignore")
    if cfg.id_column and cfg.id_column in X.columns:
        X = X.drop(columns=[cfg.id_column])
    preds = model.predict(X)
    pd.DataFrame({"prediction": preds}).to_csv(output, index=False)
    logger.info(f"Predictions saved to {output}")


@app.command()
def report(
    config: Path = typer.Option(..., "--config", "-c", exists=True)
) -> None:
    """Generate a simple markdown and figure report."""
    cfg = load_config(config)
    out_dir = ensure_output_dirs(cfg)
    metrics_path = out_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open() as fh:
            metrics = json.load(fh)
    md_path = out_dir / "report.md"
    with md_path.open("w") as fh:
        fh.write("# Model Report\n\n")
        if "accuracy" in metrics:
            fh.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
    import matplotlib.pyplot as plt

    fig_path = out_dir / "figure.png"
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("Placeholder Figure")
    plt.savefig(fig_path)
    logger.info("Report generated")


if __name__ == "__main__":  # pragma: no cover
    app()
