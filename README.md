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

The outputs (models, metrics, report and plots) will be written to
`outputs/run1` and `assets`.
