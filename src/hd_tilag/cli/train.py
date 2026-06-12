from __future__ import annotations

import argparse
from pathlib import Path

from hd_tilag.training.trainer import train_model
from hd_tilag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HD-TILAG.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--processed", default=None, help="Path to processed .npz dataset.")
    args = parser.parse_args()
    config_path = Path(args.config)
    project_root = config_path.parent.parent
    config = load_config(config_path)
    processed = args.processed or (
        project_root / config["data"].get("processed_dir", "data/processed") / f"{config['data']['processed_name']}.npz"
    )
    checkpoint = train_model(config, processed, project_root=project_root)
    print(f"Best checkpoint written to {checkpoint}")


if __name__ == "__main__":
    main()
