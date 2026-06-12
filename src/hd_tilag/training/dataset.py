from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ProcessedData:
    time_features: np.ndarray
    sentiment_features: np.ndarray
    labels: np.ndarray
    returns: np.ndarray
    close: np.ndarray
    industry_adj: np.ndarray
    style_adj: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray
    selected_feature_names: np.ndarray
    train_end: int
    val_end: int

    @property
    def num_dates(self) -> int:
        return int(self.time_features.shape[0])

    @property
    def num_stocks(self) -> int:
        return int(self.time_features.shape[1])

    @property
    def time_feature_dim(self) -> int:
        return int(self.time_features.shape[2])


def load_processed(path: str | Path) -> ProcessedData:
    loaded = np.load(path, allow_pickle=True)
    return ProcessedData(
        time_features=loaded["time_features"].astype(np.float32),
        sentiment_features=loaded["sentiment_features"].astype(np.float32),
        labels=loaded["labels"].astype(np.float32),
        returns=loaded["returns"].astype(np.float32),
        close=loaded["close"].astype(np.float32),
        industry_adj=loaded["industry_adj"].astype(np.float32),
        style_adj=loaded["style_adj"].astype(np.float32),
        dates=loaded["dates"],
        tickers=loaded["tickers"],
        selected_feature_names=loaded["selected_feature_names"],
        train_end=int(loaded["train_end"]),
        val_end=int(loaded["val_end"]),
    )


class WindowDataset:
    """Numpy-backed rolling-window dataset."""

    def __init__(self, data: ProcessedData, lookback: int, split: str) -> None:
        self.data = data
        self.lookback = lookback
        self.split = split
        if split == "train":
            start, end = lookback - 1, data.train_end - 1
        elif split in {"val", "validation"}:
            start, end = data.train_end, data.val_end - 1
        elif split == "test":
            start, end = data.val_end, data.num_dates - 1
        else:
            raise ValueError(f"Unknown split: {split}")
        self.target_indices = np.arange(max(start, lookback - 1), max(start, end) + 1)
        self.target_indices = self.target_indices[self.target_indices < data.num_dates]

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        target_idx = int(self.target_indices[idx])
        start = target_idx - self.lookback + 1
        labels = self.data.labels[target_idx]
        mask = np.isfinite(labels).astype(np.float32)
        return {
            "time_features": self.data.time_features[start : target_idx + 1],
            "sentiment_features": self.data.sentiment_features[start : target_idx + 1],
            "labels": np.nan_to_num(labels, nan=0.0).astype(np.float32),
            "mask": mask,
            "returns": np.nan_to_num(self.data.returns[target_idx], nan=0.0).astype(np.float32),
            "target_index": np.array(target_idx, dtype=np.int64),
        }
