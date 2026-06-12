from __future__ import annotations

import numpy as np
import pandas as pd


def build_relation_matrices(metadata: pd.DataFrame, tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized industry and style adjacency matrices."""
    meta = metadata.copy()
    meta["ticker"] = meta["ticker"].astype(str)
    meta = meta.drop_duplicates("ticker").set_index("ticker").reindex(tickers)
    if "industry" not in meta.columns:
        meta["industry"] = "unknown"
    if "style" not in meta.columns:
        meta["style"] = infer_style(meta)

    industry = _same_label_matrix(meta["industry"].fillna("unknown").astype(str).to_numpy())
    style = _same_label_matrix(meta["style"].fillna("unknown").astype(str).to_numpy())
    return normalize_adjacency(industry), normalize_adjacency(style)


def infer_style(meta: pd.DataFrame) -> pd.Series:
    size = _quantile_label(meta.get("market_cap"), ["small", "mid", "large"])
    value = _quantile_label(meta.get("book_to_market"), ["growth", "balanced", "value"])
    if len(size) == 0:
        size = pd.Series(["unknown"] * len(meta), index=meta.index)
    if len(value) == 0:
        value = pd.Series(["unknown"] * len(meta), index=meta.index)
    return size.astype(str) + "_" + value.astype(str)


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32)
    adj = np.maximum(adj, np.eye(adj.shape[0], dtype=np.float32))
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.power(np.clip(degree, 1e-12, None), -0.5)
    degree_matrix = np.diag(degree_inv_sqrt)
    return (degree_matrix @ adj @ degree_matrix).astype(np.float32)


def _same_label_matrix(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=str)
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def _quantile_label(values: pd.Series | None, labels: list[str]) -> pd.Series:
    if values is None:
        return pd.Series(["unknown"] * 0)
    series = pd.to_numeric(values, errors="coerce")
    if series.notna().sum() < len(labels):
        return pd.Series(["unknown"] * len(series), index=series.index)
    try:
        return pd.qcut(series.rank(method="first"), q=len(labels), labels=labels).astype(str)
    except ValueError:
        return pd.Series(["unknown"] * len(series), index=series.index)
