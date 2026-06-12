from __future__ import annotations

import argparse
from pathlib import Path

from hd_tilag.data.preprocess import preprocess_dataset
from hd_tilag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw stock datasets for HD-TILAG.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = config_path.parent.parent
    output = preprocess_dataset(config, project_root=project_root)
    print(f"Processed dataset written to {output}")


if __name__ == "__main__":
    main()
