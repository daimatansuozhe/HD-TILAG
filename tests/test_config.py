from hd_tilag.utils.config import load_config


def test_config_inheritance_loads_demo_values():
    config = load_config("configs/demo.yaml")

    assert config["data"]["raw_dir"] == "data/raw/DEMO"
    assert config["training"]["lookback_window"] == 8
    assert config["model"]["hidden_size"] == 32
    assert len(config["features"]["macro_columns"]) == 14
