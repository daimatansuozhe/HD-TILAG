from __future__ import annotations

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def add_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Add a practical subset of common technical indicators.

    The paper reports 44 technical indicators but only names a subset. This
    implementation computes a broad, deterministic indicator bank from OHLCV and
    lets the ensemble selector choose the final dimensions.
    """
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = group.copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float)
        open_ = g["open"].astype(float)

        for window in (3, 5, 10, 20, 30, 60):
            g[f"ma_{window}"] = close.rolling(window, min_periods=1).mean()
            g[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=1).mean()
            g[f"vol_ma_{window}"] = volume.rolling(window, min_periods=1).mean()
            g[f"ret_{window}"] = close.pct_change(window)
            g[f"std_{window}"] = close.pct_change().rolling(window, min_periods=2).std()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        typical = (high + low + close) / 3.0
        sma_typical = typical.rolling(20, min_periods=1).mean()
        mad = (typical - sma_typical).abs().rolling(20, min_periods=1).mean()
        g["cci_20"] = (typical - sma_typical) / (0.015 * mad.replace(0, np.nan))

        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["atr_14"] = true_range.rolling(14, min_periods=1).mean()

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / g["atr_14"].replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / g["atr_14"].replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        g["adx_14"] = dx.rolling(14, min_periods=1).mean()

        g["roc_10"] = close.pct_change(10)
        money_flow = typical * volume
        positive_flow = money_flow.where(typical.diff() > 0, 0.0).rolling(14, min_periods=1).sum()
        negative_flow = money_flow.where(typical.diff() < 0, 0.0).rolling(14, min_periods=1).sum()
        mfr = positive_flow / negative_flow.replace(0, np.nan)
        g["mfi_14"] = 100 - (100 / (1 + mfr))

        obv_direction = np.sign(close.diff()).fillna(0)
        g["obv"] = (obv_direction * volume).cumsum()

        low14 = low.rolling(14, min_periods=1).min()
        high14 = high.rolling(14, min_periods=1).max()
        g["stoch_k_14"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        g["willr_14"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

        ma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=2).std()
        g["bb_upper_20"] = ma20 + 2 * std20
        g["bb_lower_20"] = ma20 - 2 * std20
        g["bb_width_20"] = (g["bb_upper_20"] - g["bb_lower_20"]) / ma20.replace(0, np.nan)
        g["dpo_20"] = close.shift(11) - ma20
        g["hl_spread"] = (high - low) / close.replace(0, np.nan)
        g["oc_change"] = (close - open_) / open_.replace(0, np.nan)

        g["ticker"] = ticker
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out.groupby("ticker", sort=False)[numeric_cols].ffill().bfill()
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def feature_columns(frame: pd.DataFrame, macro_columns: list[str] | None = None) -> list[str]:
    macro_columns = macro_columns or []
    excluded = {"date", "ticker", "target", "return", "tradable"}
    columns = [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    for col in macro_columns:
        if col in frame.columns and col not in columns:
            columns.append(col)
    return columns
