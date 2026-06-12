from __future__ import annotations

import re
from collections import Counter

import pandas as pd


POSITIVE_WORDS = {
    "beat",
    "beats",
    "benefit",
    "bullish",
    "buy",
    "gain",
    "gains",
    "growth",
    "improve",
    "improved",
    "positive",
    "profit",
    "profits",
    "raise",
    "rally",
    "strong",
    "surge",
    "up",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "decline",
    "declines",
    "downgrade",
    "drop",
    "fall",
    "falls",
    "loss",
    "losses",
    "miss",
    "negative",
    "risk",
    "sell",
    "slump",
    "weak",
    "warning",
}


def aggregate_daily_sentiment(
    news: pd.DataFrame | None,
    dates: pd.Series,
    tickers: list[str],
    neutral_when_missing: bool = True,
) -> pd.DataFrame:
    """Return daily positive/neutral/negative proportions for each stock."""
    grid = pd.MultiIndex.from_product(
        [pd.to_datetime(sorted(dates.unique())), tickers], names=["date", "ticker"]
    ).to_frame(index=False)
    if news is None or news.empty:
        return _neutral_frame(grid, neutral_when_missing)

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news["ticker"] = news["ticker"].astype(str)
    if "sentiment" not in news.columns:
        if "text" not in news.columns:
            return _neutral_frame(grid, neutral_when_missing)
        news["sentiment"] = news["text"].fillna("").map(lexicon_sentiment)
    news["sentiment"] = news["sentiment"].map(normalize_sentiment_label)

    counts = (
        news.groupby(["date", "ticker", "sentiment"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ("positive", "neutral", "negative"):
        if label not in counts.columns:
            counts[label] = 0
    counts["total"] = counts[["positive", "neutral", "negative"]].sum(axis=1).clip(lower=1)
    for label in ("positive", "neutral", "negative"):
        counts[f"sent_{label}"] = counts[label] / counts["total"]

    out = grid.merge(
        counts[["date", "ticker", "sent_positive", "sent_neutral", "sent_negative"]],
        on=["date", "ticker"],
        how="left",
    )
    if neutral_when_missing:
        out["sent_positive"] = out["sent_positive"].fillna(0.0)
        out["sent_neutral"] = out["sent_neutral"].fillna(1.0)
        out["sent_negative"] = out["sent_negative"].fillna(0.0)
    else:
        out[["sent_positive", "sent_neutral", "sent_negative"]] = out[
            ["sent_positive", "sent_neutral", "sent_negative"]
        ].fillna(0.0)
    return out


def lexicon_sentiment(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", str(text).lower())
    counts = Counter(tokens)
    pos = sum(counts[word] for word in POSITIVE_WORDS)
    neg = sum(counts[word] for word in NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def normalize_sentiment_label(label: object) -> str:
    text = str(label).strip().lower()
    if text in {"1", "+1", "pos", "positive", "bullish", "up"}:
        return "positive"
    if text in {"-1", "neg", "negative", "bearish", "down"}:
        return "negative"
    return "neutral"


def _neutral_frame(grid: pd.DataFrame, neutral_when_missing: bool) -> pd.DataFrame:
    out = grid.copy()
    out["sent_positive"] = 0.0
    out["sent_neutral"] = 1.0 if neutral_when_missing else 0.0
    out["sent_negative"] = 0.0
    return out
