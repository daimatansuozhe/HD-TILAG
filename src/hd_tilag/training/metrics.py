from __future__ import annotations

import numpy as np


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    valid = mask.astype(bool) & np.isfinite(labels)
    if valid.sum() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    preds = (probabilities[valid] >= threshold).astype(int)
    y = labels[valid].astype(int)
    tp = float(((preds == 1) & (y == 1)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def backtest_metrics(
    probabilities: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    top_fraction: float = 0.2,
    risk_free_daily: float = 0.0,
) -> dict[str, float]:
    """Long top-ranked predicted risers for each day."""
    daily_returns: list[float] = []
    for day_probs, day_returns, day_mask in zip(probabilities, returns, mask):
        valid = day_mask.astype(bool) & np.isfinite(day_returns)
        if valid.sum() == 0:
            daily_returns.append(0.0)
            continue
        valid_indices = np.where(valid)[0]
        k = max(1, int(np.ceil(len(valid_indices) * top_fraction)))
        selected = valid_indices[np.argsort(day_probs[valid_indices])[-k:]]
        daily_returns.append(float(np.nanmean(day_returns[selected])))

    daily = np.asarray(daily_returns, dtype=np.float64)
    wealth = np.cumprod(1.0 + daily)
    cumulative_return = float(wealth[-1] - 1.0) if wealth.size else 0.0
    annualized_return = float((1.0 + cumulative_return) ** (252.0 / max(len(daily), 1)) - 1.0)
    volatility = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float((daily.mean() - risk_free_daily) / volatility * np.sqrt(252.0)) if volatility > 0 else 0.0
    running_max = np.maximum.accumulate(wealth) if wealth.size else np.array([1.0])
    drawdown = (running_max - wealth) / np.clip(running_max, 1e-12, None)
    max_drawdown = float(drawdown.max()) if drawdown.size else 0.0
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
