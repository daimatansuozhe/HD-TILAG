from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureSelectionResult:
    selected_indices: np.ndarray
    selected_names: list[str]
    scores: dict[str, np.ndarray]
    composite_scores: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def select_features(
    tensor: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    k: int,
    train_end: int,
    lasso_alpha: float = 0.001,
    random_state: int = 42,
    max_samples: int = 50000,
) -> FeatureSelectionResult:
    """Run the paper's Pearson/MIC/Lasso/RFE-style ensemble feature selector."""
    train_x = tensor[:train_end].astype(np.float64, copy=True)
    train_y = labels[:train_end].astype(np.float64, copy=True)
    means = np.nanmean(train_x, axis=(0, 1))
    stds = np.nanstd(train_x, axis=(0, 1))
    stds = np.where(stds < 1e-8, 1.0, stds)
    train_x = (train_x - means) / stds

    x_flat = train_x.reshape(-1, train_x.shape[-1])
    y_flat = train_y.reshape(-1)
    mask = np.isfinite(y_flat)
    mask &= np.isfinite(x_flat).all(axis=1)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    if x_flat.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x_flat.shape[0], size=max_samples, replace=False)
        x_flat = x_flat[idx]
        y_flat = y_flat[idx]

    pearson = _pearson_scores(x_flat, y_flat)
    mic = _mic_scores(x_flat, y_flat)
    lasso = _lasso_scores(x_flat, y_flat, lasso_alpha, random_state)
    rfe = _rfe_scores(x_flat, y_flat, random_state)

    scores = {
        "pearson": pearson,
        "mic": mic,
        "lasso": lasso,
        "rfe": rfe,
    }
    normalized = [_minmax(value) for value in scores.values()]
    composite = np.sum(normalized, axis=0)
    k = min(k, len(feature_names), tensor.shape[-1])
    selected = np.argsort(composite)[::-1][:k]
    selected = np.sort(selected)
    return FeatureSelectionResult(
        selected_indices=selected.astype(np.int64),
        selected_names=[feature_names[i] for i in selected],
        scores=scores,
        composite_scores=composite,
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )


def apply_selection(tensor: np.ndarray, result: FeatureSelectionResult) -> np.ndarray:
    normalized = (tensor.astype(np.float32) - result.means) / result.stds
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized[..., result.selected_indices].astype(np.float32)


def _pearson_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_center = y - y.mean()
    x_center = x - x.mean(axis=0)
    numerator = np.abs((x_center * y_center[:, None]).sum(axis=0))
    denominator = np.sqrt((x_center**2).sum(axis=0)) * np.sqrt((y_center**2).sum())
    return numerator / np.clip(denominator, 1e-12, None)


def _mic_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = []
    y_int = y.astype(int)
    max_bins = int(max(2, min(10, np.floor((len(y_int) ** 0.6) / 2))))
    for col in range(x.shape[1]):
        best = 0.0
        for bins in range(2, max_bins + 1):
            mi = _discrete_mi(x[:, col], y_int, bins=bins)
            best = max(best, mi / np.log(2.0))
        scores.append(min(best, 1.0))
    return np.asarray(scores, dtype=np.float64)


def _lasso_scores(x: np.ndarray, y: np.ndarray, alpha: float, random_state: int) -> np.ndarray:
    try:
        from sklearn.linear_model import Lasso

        model = Lasso(alpha=alpha, random_state=random_state, max_iter=5000)
        model.fit(x, y)
        return np.abs(model.coef_)
    except ModuleNotFoundError:
        z = np.mean(x * y[:, None], axis=0)
        denom = np.mean(x**2, axis=0)
        beta = np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0) / np.clip(denom, 1e-12, None)
        return np.abs(beta)


def _rfe_scores(x: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE

        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        selector = RFE(model, n_features_to_select=1, step=1)
        selector.fit(x, y.astype(int))
        ranking = selector.ranking_.astype(np.float64)
        return 1.0 / ranking
    except ModuleNotFoundError:
        return _pearson_scores(x, y)


def _discrete_mi(feature: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(feature, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size <= 2:
        return 0.0
    x_bin = np.digitize(feature, edges[1:-1], right=False)
    mi = 0.0
    n = len(y)
    for xb in np.unique(x_bin):
        for yb in np.unique(y):
            joint = np.mean((x_bin == xb) & (y == yb))
            if joint <= 0:
                continue
            px = np.mean(x_bin == xb)
            py = np.mean(y == yb)
            mi += joint * np.log(joint / (px * py + 1e-12) + 1e-12)
    return float(max(mi, 0.0))


def _minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mn = scores.min()
    mx = scores.max()
    if mx - mn < 1e-12:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)
