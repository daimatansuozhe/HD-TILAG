$ErrorActionPreference = "Stop"
python -m hd_tilag.cli.preprocess --config configs/bigdata22.yaml
python -m hd_tilag.cli.train --config configs/bigdata22.yaml
python -m hd_tilag.cli.evaluate --config configs/bigdata22.yaml --checkpoint runs/bigdata22/best_model.pt
