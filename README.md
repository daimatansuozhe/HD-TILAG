# HD-TILAG: A Heterogeneous Data-fused Temporal Inter-Layer Attention-Gated Model for Stock Movement Prediction
This repository contains a reproducible implementation of the paper "Fusing Temporal, Sentiment, and Structural Data for Stock Movement Prediction: A HD-TILAG Model with Heterogeneous Feature Integration".

## Overview
HD-TILAG is a deep learning framework for stock price movement prediction by integrating structured trading data, financial news sentiment, and multi-factor inter-stock relationships. The model addresses key challenges in financial market prediction, including noisy high-dimensional indicators, heterogeneous data fusion, sentiment quantification, and temporal noise propagation in stacked recurrent architectures.

## Key Innovations
HD-TILAG addresses the challenges of stock movement prediction through three key modules:
1. **Heterogeneous Data Type-Specific Processing (HDTSP)**: Processes structured OHLCV, technical indicators, macroeconomic variables, news sentiment, and multi-factor classification data using modality-specific pipelines.
2. **Ensemble Feature Selection and Dual Graph Embedding**: Combines Pearson correlation, MIC-style normalized mutual information, Lasso, and RFE-style importance scores, then applies dual GCN branches to model industry and style relations.
3. **Temporal Inter-Layer Attention-Gated Network (TILAGNet)**: Uses an Attention-Gated Unit (AGU) and Inter-Layer Gating Unit (IGU) to capture temporal dependencies while suppressing noisy inter-layer propagation.

## Architecture
The HD-TILAG framework consists of three main components:
1. **Heterogeneous Data Type-Specific Processing Module** (feature selection + sentiment aggregation + industry/style graph embedding)
2. **Multi-Stock Temporal Feature Aggregation Module** (TILAGNet with AGU and IGU)
3. **Output Mapping Module** (binary classification for rise/fall movement prediction)

## Datasets
The model is evaluated on three real-world benchmark datasets:
- **ACL18**: U.S. stocks with OHLCV prices, tweets, macroeconomic variables, and sector/style metadata over the paper's 2014-01-02 to 2015-12-30 evaluation window.
- **BIGDATA22**: NASDAQ-style U.S. stock data with OHLCV prices, tweets, macroeconomic variables, and sector/style metadata over 2019-07-05 to 2020-06-30.
- **CMIN**: Chinese market stock data with OHLCV prices, Chinese financial news, macroeconomic variables, and sector/style metadata over 2020-01-01 to 2021-12-31.

The repository supports the raw directory layout currently used in `data/raw/<DATASET>/`:
```text
price/*.csv
tweet/<ticker>/<date>        # ACL18 and BIGDATA22
tweet/<ticker>.csv           # CMIN
macro_eco/filled_macro_economy.csv
style_industry/sector_style.csv
```

## Results
The code reports classification metrics (accuracy, precision, recall, F1-score) and trading metrics (cumulative return, annualized return, Sharpe ratio, maximum drawdown). Processed tensors are generated with an 8:1:1 chronological split, and sideways samples are excluded from the loss and metrics using the return thresholds described in the paper.

Due to the absence of released HD-TILAG source code and final FDS model weights in the provided manuscript, this repository implements the full HD-TILAG experiment pipeline from the paper equations and provides a replaceable sentiment adapter. If a `sentiment` column is unavailable, raw news and tweets are converted to daily positive/neutral/negative proportions with a lightweight fallback classifier.

We appreciate your understanding. If you have any specific questions or requests regarding the code, please feel free to contact us.

## Installation
```powershell
cd C:\Users\87471\Documents\app\hd-tilag-reproduction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

## Running Experiments
Preprocess all datasets:
```powershell
python -m hd_tilag.cli.run_experiments --preprocess-only
```

Run the full experiment pipeline:
```powershell
python -m hd_tilag.cli.run_experiments
```

Run one dataset:
```powershell
python -m hd_tilag.cli.preprocess --config configs/acl18.yaml
python -m hd_tilag.cli.train --config configs/acl18.yaml
python -m hd_tilag.cli.evaluate --config configs/acl18.yaml --checkpoint runs/acl18/best_model.pt
```

PowerShell helpers are also provided:
```powershell
.\scripts\run_acl18.ps1
.\scripts\run_bigdata22.ps1
.\scripts\run_cmin.ps1
.\scripts\run_all.ps1
```

