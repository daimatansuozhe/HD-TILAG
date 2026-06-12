from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hd_tilag.data.paper_layout import is_paper_layout, load_paper_layout
from hd_tilag.data.schema import read_csv_standard, require_columns, standardize_columns
from hd_tilag.features.graphs import build_relation_matrices
from hd_tilag.features.selection import apply_selection, select_features
from hd_tilag.features.sentiment import aggregate_daily_sentiment
from hd_tilag.features.technical import OHLCV_COLUMNS, add_technical_indicators, feature_columns


def preprocess_dataset(config: dict[str, Any], project_root: Path | None = None) -> Path:
    project_root = project_root or Path.cwd()
    data_cfg = config["data"]
    feature_cfg = config["features"]
    raw_dir = _resolve(project_root, data_cfg["raw_dir"])
    processed_dir = _resolve(project_root, data_cfg.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_name = data_cfg["processed_name"]
    date_range = data_cfg.get("date_range", {})
    start_date = date_range.get("start")
    end_date = date_range.get("end")

    prices, news, metadata, macro = _load_raw_inputs(raw_dir, data_cfg, start_date, end_date)
    require_columns(prices, ["date", "ticker", *OHLCV_COLUMNS], "prices.csv")
    prices = prices.sort_values(["date", "ticker"]).copy()
    prices["date"] = prices["date"].dt.normalize()

    tickers = sorted(prices["ticker"].astype(str).unique().tolist())
    dates = pd.Series(sorted(prices["date"].unique()))

    frame = add_technical_indicators(prices)
    frame = _merge_macro(frame, macro, feature_cfg)
    frame = _add_targets(
        frame,
        up_threshold=float(data_cfg.get("up_threshold", 0.0055)),
        down_threshold=float(data_cfg.get("down_threshold", -0.005)),
        target_horizon=int(data_cfg.get("target_horizon", 1)),
    )

    feature_names = feature_columns(frame, feature_cfg.get("macro_columns", []))
    full_tensor = _to_tensor(frame, dates, tickers, feature_names)
    labels = _to_matrix(frame, dates, tickers, "target")
    returns = _to_matrix(frame, dates, tickers, "return")
    close = _to_matrix(frame, dates, tickers, "close")

    if news is not None:
        require_columns(news, ["date", "ticker"], "news.csv")
    sentiment_frame = aggregate_daily_sentiment(
        news,
        dates=dates,
        tickers=tickers,
        neutral_when_missing=bool(feature_cfg.get("sentiment", {}).get("neutral_when_missing", True)),
    )
    sentiment_tensor = _to_tensor(
        sentiment_frame,
        dates,
        tickers,
        ["sent_positive", "sent_neutral", "sent_negative"],
    )

    if metadata is None:
        metadata = pd.DataFrame({"ticker": tickers})
    metadata = standardize_columns(metadata)
    require_columns(metadata, ["ticker"], "metadata.csv")
    industry_adj, style_adj = build_relation_matrices(metadata, tickers)

    train_end, val_end = chronological_boundaries(len(dates), data_cfg.get("split", {}))
    selection = select_features(
        full_tensor,
        labels,
        feature_names=feature_names,
        k=int(feature_cfg.get("selected_time_series_features", 32)),
        train_end=train_end,
        lasso_alpha=float(feature_cfg.get("feature_selection", {}).get("lasso_alpha", 0.001)),
        random_state=int(feature_cfg.get("feature_selection", {}).get("random_state", 42)),
        max_samples=int(feature_cfg.get("feature_selection", {}).get("max_samples", 50000)),
    )
    selected_tensor = apply_selection(full_tensor, selection)

    output_path = processed_dir / f"{processed_name}.npz"
    np.savez_compressed(
        output_path,
        time_features=selected_tensor.astype(np.float32),
        sentiment_features=sentiment_tensor.astype(np.float32),
        labels=labels.astype(np.float32),
        returns=returns.astype(np.float32),
        close=close.astype(np.float32),
        industry_adj=industry_adj.astype(np.float32),
        style_adj=style_adj.astype(np.float32),
        dates=np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates], dtype=object),
        tickers=np.array(tickers, dtype=object),
        selected_feature_names=np.array(selection.selected_names, dtype=object),
        train_end=np.array(train_end),
        val_end=np.array(val_end),
    )

    metadata_out = {
        "processed_file": str(output_path),
        "num_dates": len(dates),
        "num_stocks": len(tickers),
        "num_time_features_before_selection": len(feature_names),
        "num_time_features_after_selection": len(selection.selected_names),
        "selected_feature_names": selection.selected_names,
        "train_end": train_end,
        "val_end": val_end,
        "date_range": [str(dates.iloc[0].date()), str(dates.iloc[-1].date())],
        "source_layout": "paper_directory" if is_paper_layout(raw_dir) else "flat_csv",
        "source_files": _source_files(raw_dir, data_cfg),
    }
    (processed_dir / f"{processed_name}_metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )
    return output_path


def _load_raw_inputs(
    raw_dir: Path,
    data_cfg: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    if is_paper_layout(raw_dir):
        return load_paper_layout(raw_dir, start_date=start_date, end_date=end_date)

    required = data_cfg.get("required_files", {})
    prices = read_csv_standard(raw_dir / required["prices"])
    news = _read_optional(raw_dir / required.get("news", "news.csv"))
    metadata = _read_optional(raw_dir / required.get("metadata", "metadata.csv"))
    macro = _read_optional(raw_dir / required.get("macro", "macro.csv"))
    if start_date or end_date:
        prices = _filter_frame_dates(prices, start_date, end_date)
        if news is not None and "date" in news.columns:
            news = _filter_frame_dates(news, start_date, end_date)
        if macro is not None and "date" in macro.columns:
            macro = _filter_frame_dates(macro, start_date, end_date)
    return prices, news, metadata, macro


def chronological_boundaries(num_dates: int, split: dict[str, float]) -> tuple[int, int]:
    train_ratio = float(split.get("train", 0.8))
    val_ratio = float(split.get("validation", 0.1))
    train_end = int(num_dates * train_ratio)
    val_end = train_end + int(num_dates * val_ratio)
    train_end = max(1, min(train_end, num_dates - 2))
    val_end = max(train_end + 1, min(val_end, num_dates - 1))
    return train_end, val_end


def _merge_macro(frame: pd.DataFrame, macro: pd.DataFrame | None, feature_cfg: dict[str, Any]) -> pd.DataFrame:
    if macro is None or macro.empty:
        return frame
    macro = standardize_columns(macro)
    require_columns(macro, ["date"], "macro.csv")
    macro = macro.sort_values("date").copy()
    macro["date"] = macro["date"].dt.normalize()
    macro_columns = [
        col for col in feature_cfg.get("macro_columns", []) if col in macro.columns
    ] or [col for col in macro.columns if col != "date" and pd.api.types.is_numeric_dtype(macro[col])]
    if not macro_columns:
        return frame

    dates = pd.DataFrame({"date": sorted(frame["date"].unique())})
    aligned = pd.merge_asof(dates, macro[["date", *macro_columns]], on="date", direction="backward")
    aligned[macro_columns] = aligned[macro_columns].ffill().bfill().fillna(0.0)
    return frame.merge(aligned, on="date", how="left")


def _add_targets(
    frame: pd.DataFrame,
    up_threshold: float,
    down_threshold: float,
    target_horizon: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = group.copy()
        future_close = g["close"].shift(-target_horizon)
        g["return"] = (future_close - g["close"]) / g["close"].replace(0, np.nan)
        g["target"] = np.nan
        g.loc[g["return"] > up_threshold, "target"] = 1.0
        g.loc[g["return"] < down_threshold, "target"] = 0.0
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _to_tensor(
    frame: pd.DataFrame,
    dates: pd.Series,
    tickers: list[str],
    columns: list[str],
) -> np.ndarray:
    tensors = []
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    keyed = frame.copy()
    keyed["date"] = pd.to_datetime(keyed["date"]).dt.normalize()
    keyed["ticker"] = keyed["ticker"].astype(str)
    keyed = keyed.drop_duplicates(["date", "ticker"]).set_index(["date", "ticker"])
    for col in columns:
        series = pd.to_numeric(keyed[col], errors="coerce").reindex(index).fillna(0.0)
        tensors.append(series.to_numpy().reshape(len(dates), len(tickers)))
    return np.stack(tensors, axis=-1).astype(np.float32)


def _to_matrix(frame: pd.DataFrame, dates: pd.Series, tickers: list[str], column: str) -> np.ndarray:
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    keyed = frame.copy()
    keyed["date"] = pd.to_datetime(keyed["date"]).dt.normalize()
    keyed["ticker"] = keyed["ticker"].astype(str)
    keyed = keyed.drop_duplicates(["date", "ticker"]).set_index(["date", "ticker"])
    return pd.to_numeric(keyed[column], errors="coerce").reindex(index).to_numpy().reshape(
        len(dates), len(tickers)
    )


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return read_csv_standard(path)


def _filter_frame_dates(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if start_date:
        frame = frame[frame["date"] >= pd.to_datetime(start_date).normalize()]
    if end_date:
        frame = frame[frame["date"] <= pd.to_datetime(end_date).normalize()]
    return frame


def _source_files(raw_dir: Path, data_cfg: dict[str, Any]) -> dict[str, str]:
    if is_paper_layout(raw_dir):
        return {
            "price": str(raw_dir / "price"),
            "tweet": str(raw_dir / "tweet"),
            "macro": str(raw_dir / "macro_eco"),
            "metadata": str(raw_dir / "style_industry"),
        }
    return {
        key: str(raw_dir / value) for key, value in data_cfg.get("required_files", {}).items()
    }


def _resolve(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path
