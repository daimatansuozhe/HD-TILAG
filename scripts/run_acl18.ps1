$ErrorActionPreference = "Stop"
python -m hd_tilag.cli.preprocess --config configs/acl18.yaml
python -m hd_tilag.cli.train --config configs/acl18.yaml
python -m hd_tilag.cli.evaluate --config configs/acl18.yaml --checkpoint runs/acl18/best_model.pt
