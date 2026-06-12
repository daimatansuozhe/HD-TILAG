from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from hd_tilag.data.schema import read_csv_standard, standardize_columns


def is_paper_layout(raw_dir: str | Path) -> bool:
    raw_dir = Path(raw_dir)
    return (raw_dir / "price").is_dir() and (
        (raw_dir / "tweet").is_dir()
        or (raw_dir / "macro_eco").is_dir()
        or (raw_dir / "style_industry").is_dir()
    )


def load_paper_layout(
    raw_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None]:
    raw_dir = Path(raw_dir)
    prices = load_price_directory(raw_dir / "price", start_date=start_date, end_date=end_date)
    tickers = sorted(prices["ticker"].unique().tolist())
    metadata = load_metadata(raw_dir / "style_industry" / "sector_style.csv", tickers)
    macro = load_macro(raw_dir / "macro_eco", start_date=start_date, end_date=end_date)
    news = load_tweet_directory(
        raw_dir / "tweet",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    return prices, news, metadata, macro


def load_price_directory(
    price_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    price_dir = Path(price_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(price_dir.glob("*.csv")):
        if path.name.startswith("."):
            continue
        frame = read_csv_standard(path)
        frame["ticker"] = path.stem
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No price CSV files found in {price_dir}")
    prices = pd.concat(frames, ignore_index=True)
    prices = standardize_columns(prices)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str)
    prices = _filter_date_range(prices, start_date, end_date)
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    prices = prices[keep].dropna(subset=keep)
    return prices.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_metadata(path: str | Path, tickers: Iterable[str]) -> pd.DataFrame:
    tickers = [str(t) for t in tickers]
    if not Path(path).exists():
        return pd.DataFrame({"ticker": tickers})
    metadata = read_csv_standard(path, dtype=str)
    metadata["ticker"] = metadata["ticker"].astype(str)
    metadata = _harmonize_ticker_width(metadata, tickers)
    if "industry" not in metadata.columns and "sector" in metadata.columns:
        metadata["industry"] = metadata["sector"]
    if "style" not in metadata.columns:
        metadata["style"] = "unknown"
    return metadata


def load_macro(
    macro_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    macro_dir = Path(macro_dir)
    candidates = [
        macro_dir / "filled_macro_economy.csv",
        macro_dir / "macro_economy.csv",
        macro_dir / "macro_economy_raw.csv",
        macro_dir / "macro.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        return None
    macro = read_csv_standard(path)
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
    macro = _filter_date_range(macro, start_date, end_date)
    return macro.sort_values("date").reset_index(drop=True)


def load_tweet_directory(
    tweet_dir: str | Path,
    tickers: Iterable[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    tweet_dir = Path(tweet_dir)
    if not tweet_dir.exists():
        return None
    rows: list[pd.DataFrame] = []
    ticker_set = set(str(t) for t in tickers)
    start = pd.to_datetime(start_date).normalize() if start_date else None
    end = pd.to_datetime(end_date).normalize() if end_date else None

    ticker_dirs = [p for p in tweet_dir.iterdir() if p.is_dir() and p.name in ticker_set]
    if ticker_dirs:
        for ticker_dir in sorted(ticker_dirs):
            ticker = ticker_dir.name
            files = sorted(p for p in ticker_dir.iterdir() if p.is_file() and not p.name.startswith("."))
            for path in files:
                date = _parse_date_from_name(path.name)
                if date is None or not _date_in_range(date, start, end):
                    continue
                frame = _read_tweet_file(path)
                if frame is None or frame.empty:
                    continue
                frame["date"] = date
                frame["ticker"] = ticker
                rows.append(frame[["date", "ticker", "text"]])
    else:
        for path in sorted(tweet_dir.glob("*.csv")):
            if path.name.startswith("."):
                continue
            ticker = path.stem
            if ticker not in ticker_set:
                ticker = _match_ticker_width(ticker, ticker_set)
            if ticker not in ticker_set:
                continue
            frame = _read_tweet_file(path)
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            frame["ticker"] = ticker
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame[frame["date"].map(lambda value: _date_in_range(value, start, end))]
            rows.append(frame[["date", "ticker", "text"]])

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _read_tweet_file(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return None
    except (UnicodeDecodeError, pd.errors.ParserError, OverflowError):
        try:
            frame = pd.read_csv(
                path,
                dtype=str,
                encoding="utf-8",
                encoding_errors="replace",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) <= 1:
                return None
            frame = pd.DataFrame({"text": [line for line in lines[1:] if line.strip()]})
    frame = standardize_columns(frame)
    if "text" not in frame.columns and "sentence" in frame.columns:
        frame["text"] = frame["sentence"]
    if "text" not in frame.columns:
        text_col = next((col for col in frame.columns if frame[col].dtype == object), None)
        if text_col is None:
            return None
        frame["text"] = frame[text_col]
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame


def _filter_date_range(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if start_date:
        frame = frame[frame["date"] >= pd.to_datetime(start_date).normalize()]
    if end_date:
        frame = frame[frame["date"] <= pd.to_datetime(end_date).normalize()]
    return frame


def _harmonize_ticker_width(metadata: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    width = max((len(t) for t in tickers if t.isdigit()), default=0)
    if width > 0:
        ticker_set = set(tickers)
        metadata["ticker"] = metadata["ticker"].map(
            lambda value: str(value).zfill(width)
            if str(value).isdigit() and str(value).zfill(width) in ticker_set
            else str(value)
        )
    return metadata


def _match_ticker_width(value: str, ticker_set: set[str]) -> str:
    if not value.isdigit():
        return value
    widths = sorted({len(t) for t in ticker_set if t.isdigit()}, reverse=True)
    for width in widths:
        candidate = value.zfill(width)
        if candidate in ticker_set:
            return candidate
    return value


def _parse_date_from_name(name: str) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(Path(name).stem).normalize()
    except ValueError:
        return None


def _date_in_range(value: pd.Timestamp, start: pd.Timestamp | None, end: pd.Timestamp | None) -> bool:
    value = pd.to_datetime(value).normalize()
    if start is not None and value < start:
        return False
    if end is not None and value > end:
        return False
    return True
