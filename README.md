# Bank ML Pipeline

This repository provides a backend Python package for a bank marketing machine-learning pipeline.

## Installation

Requires Python 3.10 or newer. Install using [Hatch](https://hatch.pypa.io):

```bash
pip install hatch
hatch build
pip install dist/*.whl
```

Or directly:

```bash
pip install .
```

## CLI Usage

After installation, the `bank-ml` command becomes available.

```bash
bank-ml fit --config config.yaml
bank-ml fit --config config.yaml --fast  # quick smoke test
bank-ml predict --config config.yaml --input new.csv --output preds.csv
bank-ml report --config config.yaml
```

Configuration files are YAML and follow the structure described in `bank_ml/config.py`.

### Minimal `config.yaml`

```yaml
paths:
  input_csv: PATH_TO_CSV
  output_dir: ./outputs/run1
label: Label_LoanDefault
id_column: CustomerID
cv:
  n_splits: 5
  test_size: 0.3
  random_state: 42
imbalance:
  method: class_weight
ga:
  pop: 20
  gens: 15
  cx_prob: 0.8
  mut_prob: 0.1
clustering:
  k_grid: [2,3,4,5,6]
  n_init: 15
  max_iter: 300
pso:
  particles: 16
  iters: 20
mlp_bounds:
  hidden1: [16, 64]
  hidden2: [16, 64]
  lr: [0.0001, 0.05]
  alpha: [1e-6, 1e-3]
  momentum: [0.5, 0.95]
  activation_choices: ["logistic","tanh","relu"]
```

### Demo

A small demo dataset (`banking_dataset.csv`) is bundled with the repository. Run
the full pipeline using the provided `demo-config.yaml` via:

```bash
make demo
```

### Smoke check

A lightweight verification script is provided to ensure the pipeline and its
dependencies work end‑to‑end:

```bash
python scripts/smoke_check.py
```

The script trains in fast mode on a small sample of the demo dataset, checks
that expected artefacts are produced and runs a short prediction.

Each training run stores artefacts under a timestamped subdirectory inside
`output_dir`, e.g. `outputs/run1/20240101-120000`. Plots are written to the
shared `assets` folder.

Expected files inside the run directory include:

- `preprocess.joblib` – fitted preprocessing pipeline
- `feature_names.json` – names of engineered features
- `ga_selection.json` – feature selection mask and chosen names
- `clustering.joblib` and `clustering.json` – clustering model and metrics
- `mlp.joblib`, `log_reg.joblib`, `decision_tree.joblib`, `random_forest.joblib`
  – trained models
- `manifest.json` – summary of training parameters and scores
- `metrics.csv` – evaluation metrics for the models
- `report.md` – Markdown report of the run

## Streamlit App

An optional Streamlit UI is provided for interacting with the pipeline. Launch
it with:

```bash
streamlit run app_streamlit.py
```

The sidebar lets you upload a CSV or specify a path, choose a config file
(`./config.yaml` by default) and toggle **Fast mode**. Buttons trigger training,
prediction on the uploaded data and rendering of the Markdown report. The main
area displays logs and artefacts from training, shows prediction results with a
download option and renders the generated report.

