from __future__ import annotations

import argparse
import json
from pathlib import Path

from hd_tilag.training.trainer import evaluate_checkpoint
from hd_tilag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HD-TILAG checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--processed", default=None, help="Path to processed .npz dataset.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "val", "test"])
    args = parser.parse_args()
    config_path = Path(args.config)
    project_root = config_path.parent.parent
    config = load_config(config_path)
    processed = args.processed or (
        project_root / config["data"].get("processed_dir", "data/processed") / f"{config['data']['processed_name']}.npz"
    )
    metrics = evaluate_checkpoint(config, processed, args.checkpoint, split=args.split)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
