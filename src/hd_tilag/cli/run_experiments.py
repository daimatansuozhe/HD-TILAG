from __future__ import annotations

import argparse
import json
from pathlib import Path

from hd_tilag.data.preprocess import preprocess_dataset
from hd_tilag.training.trainer import evaluate_checkpoint, train_model
from hd_tilag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HD-TILAG experiments.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/acl18.yaml", "configs/bigdata22.yaml", "configs/cmin.yaml"],
        help="Dataset config files to run.",
    )
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "validation", "val", "test"])
    args = parser.parse_args()

    results = {}
    for config_arg in args.configs:
        config_path = Path(config_arg)
        project_root = config_path.parent.parent
        config = load_config(config_path)
        name = config["data"]["processed_name"]
        processed = (
            project_root
            / config["data"].get("processed_dir", "data/processed")
            / f"{config['data']['processed_name']}.npz"
        )
        if not args.skip_preprocess:
            processed = preprocess_dataset(config, project_root=project_root)
        if args.preprocess_only:
            results[name] = {"processed": str(processed)}
            continue
        if args.skip_train:
            checkpoint = project_root / config["training"]["output_dir"] / "best_model.pt"
        else:
            checkpoint = train_model(config, processed, project_root=project_root)
        metrics = evaluate_checkpoint(config, processed, checkpoint, split=args.split)
        results[name] = {"processed": str(processed), "checkpoint": str(checkpoint), **metrics}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
