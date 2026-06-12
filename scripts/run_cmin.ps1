$ErrorActionPreference = "Stop"
python -m hd_tilag.cli.preprocess --config configs/cmin.yaml
python -m hd_tilag.cli.train --config configs/cmin.yaml
python -m hd_tilag.cli.evaluate --config configs/cmin.yaml --checkpoint runs/cmin/best_model.pt
