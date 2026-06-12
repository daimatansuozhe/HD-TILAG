from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMN_ALIASES = {
    "date": {"date", "datetime", "time", "timestamp", "trade_date", "trading_date"},
    "ticker": {"ticker", "symbol", "stock", "stock_id", "code", "asset", "permno"},
    "open": {"open", "open_price", "opening_price"},
    "high": {"high", "high_price", "highest", "highest_price"},
    "low": {"low", "low_price", "lowest", "lowest_price"},
    "close": {"close", "adj_close", "adjusted_close", "close_price", "closing_price"},
    "volume": {"volume", "vol", "turnover_volume"},
    "text": {"text", "content", "headline", "title", "tweet", "body", "news"},
    "sentiment": {"sentiment", "label", "polarity"},
    "industry": {"industry", "sector", "gics_sector", "industry_name"},
    "style": {"style", "style_type", "investment_style", "stock_style_box"},
    "market_cap": {"market_cap", "mktcap", "capitalization", "size"},
    "book_to_market": {"book_to_market", "btm", "bm", "value"},
    "momentum": {"momentum", "mom", "ret_12_1"},
    "SPMI": {"spmi", "serv_pmi", "services_pmi", "service_pmi"},
    "MPMI": {"mpmi", "manu_pmi", "manufacturing_pmi", "manufacturing_purchasing_managers_index"},
}


def read_csv_standard(path: str | Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return standardize_columns(df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    normalized = {_clean_col(c): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                rename[normalized[alias]] = canonical
                break
    out = df.rename(columns=rename).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str)
    return out


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _clean_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")
