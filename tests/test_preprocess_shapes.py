from pathlib import Path

import numpy as np

from hd_tilag.cli.make_demo_data import main as make_demo_main
from hd_tilag.data.preprocess import preprocess_dataset
from hd_tilag.utils.config import load_config


def test_preprocess_demo_shapes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    source_config = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    default_config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    (tmp_path / "configs" / "demo.yaml").write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs" / "default.yaml").write_text(
        default_config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(
        "sys.argv",
        ["make_demo_data", "--output-dir", "data/raw/DEMO", "--stocks", "5", "--days", "40"],
    )
    make_demo_main()
    config = load_config(tmp_path / "configs" / "demo.yaml")
    config["features"]["selected_time_series_features"] = 8
    output = preprocess_dataset(config, project_root=tmp_path)

    data = np.load(output, allow_pickle=True)
    assert data["time_features"].shape == (40, 5, 8)
    assert data["sentiment_features"].shape == (40, 5, 3)
    assert data["industry_adj"].shape == (5, 5)
