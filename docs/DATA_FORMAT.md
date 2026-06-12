# Data Format

Place each public benchmark's raw files under:

- `data/raw/ACL18/`
- `data/raw/BIGDATA22/`
- `data/raw/CMIN/`

The preprocessing command supports two layouts. The preferred layout is the paper-style raw directory already used by this repository:

```text
data/raw/<DATASET>/
  price/*.csv
  tweet/<ticker>/<date>
  tweet/<ticker>.csv
  macro_eco/filled_macro_economy.csv
  style_industry/sector_style.csv
```

A flat fallback layout is also supported with `prices.csv`, `news.csv`, `metadata.csv`, and `macro.csv`. Column names are case-insensitive and common aliases such as `symbol` for `ticker` or `trade_date` for `date` are accepted.

## `prices.csv`

Required columns:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `volume`

Rows should be daily OHLCV observations. The label is generated from next-day close-to-close return:

- up if `return > 0.0055`
- down if `return < -0.005`
- sideways otherwise, excluded from loss and metrics

## `news.csv`

Required columns:

- `date`
- `ticker`

Use either:

- `sentiment`: one of `positive`, `neutral`, `negative`
- `text`: raw financial news/tweet text

If `sentiment` is missing, preprocessing uses the repository's lightweight lexicon fallback. For paper-faithful reproduction, annotate this file with an FDS-compatible model first and provide the `sentiment` column.

## `metadata.csv`

Required:

- `ticker`

Recommended:

- `industry`: primary industry or GICS sector
- `style`: one of the nine style classes

If `style` is missing, preprocessing infers a coarse style from `market_cap` and `book_to_market` when available.

## `macro.csv`

Required:

- `date`

Recommended macro columns, matching the paper:

- `GDP`, `CPI`, `UR`, `NFP`, `ICI`, `CCI`, `MPMI`
- `M1`, `M2`, `TY10`, `FSI`, `GPR`, `EPU`, `SPMI`

Macro observations are aligned by release date using backward as-of merge and forward fill, which avoids future leakage.
