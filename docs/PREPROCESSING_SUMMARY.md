# Preprocessing Summary

The current raw data in `data/raw/` was successfully converted with:

```powershell
python -m hd_tilag.cli.preprocess --config configs/acl18.yaml
python -m hd_tilag.cli.preprocess --config configs/bigdata22.yaml
python -m hd_tilag.cli.preprocess --config configs/cmin.yaml
```

Generated tensor shapes:

| Dataset | Time features | Sentiment features | Valid labels |
| --- | ---: | ---: | ---: |
| ACL18 | `(503, 81, 32)` | `(503, 81, 3)` | `24,814` |
| BIGDATA22 | `(250, 48, 32)` | `(250, 48, 3)` | `8,358` |
| CMIN | `(486, 113, 32)` | `(486, 113, 3)` | `43,847` |

The processed `.npz` files are intentionally ignored by Git because they are reproducible artifacts.
