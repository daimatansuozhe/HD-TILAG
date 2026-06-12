import numpy as np

from hd_tilag.training.metrics import backtest_metrics, classification_metrics


def test_classification_metrics_respect_mask():
    probs = np.array([[0.8, 0.2, 0.9]])
    labels = np.array([[1.0, 0.0, 0.0]])
    mask = np.array([[1.0, 1.0, 0.0]])

    metrics = classification_metrics(probs, labels, mask)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_backtest_metrics_returns_expected_keys():
    probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    returns = np.array([[0.01, -0.02], [-0.01, 0.03]])
    mask = np.ones_like(probs)

    metrics = backtest_metrics(probs, returns, mask, top_fraction=0.5)

    assert set(metrics) == {
        "cumulative_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
    }
    assert metrics["cumulative_return"] > 0
