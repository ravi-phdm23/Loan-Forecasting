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
