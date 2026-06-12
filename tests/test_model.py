import pytest

torch = pytest.importorskip("torch")

from hd_tilag.models import HDTILAG


def test_model_forward_shape():
    model = HDTILAG(
        time_feature_dim=8,
        sentiment_dim=3,
        graph_dim=4,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        num_stocks=5,
    )
    time_features = torch.randn(2, 6, 5, 8)
    sentiment_features = torch.randn(2, 6, 5, 3)
    adjacency = torch.eye(5)

    logits = model(time_features, sentiment_features, adjacency, adjacency)

    assert logits.shape == (2, 5)
