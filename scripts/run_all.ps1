$ErrorActionPreference = "Stop"

python -m hd_tilag.cli.run_experiments --configs `
  configs/acl18.yaml `
  configs/bigdata22.yaml `
  configs/cmin.yaml
