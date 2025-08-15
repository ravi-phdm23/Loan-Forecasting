from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from bank_ml.cli import app


def write_config(tmp_path: Path) -> Path:
    data = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0], "label": [0, 1, 0, 1]})
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    cfg = {
        "paths": {"input_csv": str(data_path), "output_dir": str(tmp_path / "out")},
        "label": "label",
    }
    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg_path


def test_cli_fit_predict_report(tmp_path: Path):
    cfg_path = write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["fit", "--config", str(cfg_path)])
    assert result.exit_code == 0
    model_path = tmp_path / "out" / "model.joblib"
    assert model_path.exists()

    # prediction
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"f1": [0], "f2": [1], "label": [0]}).to_csv(input_path, index=False)
    pred_path = tmp_path / "pred.csv"
    result = runner.invoke(app, [
        "predict",
        "--config",
        str(cfg_path),
        "--input",
        str(input_path),
        "--output",
        str(pred_path),
    ])
    assert result.exit_code == 0
    assert pred_path.exists()

    result = runner.invoke(app, ["report", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert (tmp_path / "out" / "report.md").exists()
