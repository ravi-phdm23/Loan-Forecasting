from pathlib import Path
import io
import json

import numpy as np
import pandas as pd
import joblib

import streamlit as st

from bank_ml import cli, preprocess, clustering
from bank_ml.config import load_config


@st.cache_resource
def load_artifacts(out_dir: str):
    out = Path(out_dir)
    preproc = preprocess.load_preprocess(out / "preprocess.joblib")
    ga_info = json.loads((out / "ga_selection.json").read_text())
    mask = np.array(ga_info.get("mask", []), dtype=bool)
    cluster_model = joblib.load(out / "clustering.joblib")
    model = joblib.load(out / "mlp.joblib")
    return {
        "preproc": preproc,
        "mask": mask,
        "cluster_model": cluster_model,
        "model": model,
        "ga_info": ga_info,
    }


@st.cache_data
def load_metrics(out_dir: str) -> pd.DataFrame:
    path = Path(out_dir) / "metrics.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


@st.cache_data
def load_report(out_dir: str) -> str:
    path = Path(out_dir) / "report.md"
    if path.exists():
        return path.read_text()
    return "Report not found."


st.title("Bank ML Streamlit UI")

# Sidebar widgets -----------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload CSV for prediction", type="csv")
path_input = st.sidebar.text_input("...or CSV path", "")
config_path = st.sidebar.text_input("Config path", "./config.yaml")
fast_mode = st.sidebar.checkbox("Fast mode")

train_btn = st.sidebar.button("Run Training")
predict_btn = st.sidebar.button("Predict on Uploaded CSV")
report_btn = st.sidebar.button("Show Report")


# Helper to resolve output directory for latest run -------------------------
def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        return root
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return root
    return max(dirs, key=lambda p: p.stat().st_mtime)


if train_btn:
    cfg = load_config(config_path)
    root_out = cfg.paths.output_dir
    before = set(p for p in root_out.glob("*/")) if root_out.exists() else set()
    log_stream = io.StringIO()
    logger_id = cli.logger.add(log_stream, level="INFO")
    with st.spinner("Running training..."):
        cli.fit(Path(config_path), fast=fast_mode)
    cli.logger.remove(logger_id)
    st.subheader("Training logs")
    st.text_area("Logs", log_stream.getvalue(), height=200)
    after = set(p for p in root_out.glob("*/")) if root_out.exists() else set()
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    run_dir = new_dirs[-1] if new_dirs else _latest_run_dir(root_out)
    st.session_state["run_dir"] = str(run_dir)
    st.success(f"Training complete. Artefacts in {run_dir}")

    # Display artefact summaries
    ga_path = run_dir / "ga_selection.json"
    if ga_path.exists():
        ga_info = json.loads(ga_path.read_text())
        st.write("### Selected Features")
        st.write(", ".join(ga_info.get("selected_names", [])))
    cluster_path = run_dir / "clustering.json"
    if cluster_path.exists():
        cluster_info = json.loads(cluster_path.read_text())
        st.write("### Clustering")
        st.write(f"k: {cluster_info.get('k')}")
        st.write("DBI per k:", cluster_info.get("dbi_per_k"))
        st.write("Silhouette per k:", cluster_info.get("sil_per_k"))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        st.write("### PSO Best Params")
        st.json(manifest.get("mlp_params", {}))

    metrics_df = load_metrics(str(run_dir))
    if not metrics_df.empty:
        st.write("### Test Metrics")
        st.dataframe(metrics_df)

    assets = Path("assets")
    for name in ["mlp_confusion_matrix.png", "mlp_roc_curve.png", "mlp_pr_curve.png"]:
        img = assets / name
        if img.exists():
            st.image(str(img), caption=name)


if predict_btn:
    run_dir = Path(st.session_state.get("run_dir", ""))
    if not run_dir:
        cfg = load_config(config_path)
        run_dir = _latest_run_dir(cfg.paths.output_dir)
    if not run_dir.exists():
        st.error("Run training first or provide a valid output directory in config.")
    else:
        if uploaded:
            df = pd.read_csv(uploaded)
        elif path_input:
            df = pd.read_csv(path_input)
        else:
            st.error("Please upload a CSV or provide a path.")
            df = None
        if df is not None:
            cfg = load_config(config_path)
            artefacts = load_artifacts(str(run_dir))
            X = df.drop(columns=[cfg.label], errors="ignore")
            if cfg.id_column and cfg.id_column in X.columns:
                X = X.drop(columns=[cfg.id_column])
            X_pre = preprocess.transform_preprocess(artefacts["preproc"], X)
            X_sel = X_pre[:, artefacts["mask"]]
            labels = artefacts["cluster_model"].predict(X_sel)
            X_aug = clustering.append_cluster_features(X_sel, labels)
            preds = artefacts["model"].predict(X_aug)
            out_df = df.copy()
            out_df["prediction"] = preds
            st.write("### Predictions head")
            st.dataframe(out_df.head())
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions", csv, file_name="predictions.csv", mime="text/csv"
            )


if report_btn:
    run_dir = Path(st.session_state.get("run_dir", ""))
    if not run_dir:
        cfg = load_config(config_path)
        run_dir = _latest_run_dir(cfg.paths.output_dir)
    report_md = load_report(str(run_dir))
    st.markdown(report_md)
