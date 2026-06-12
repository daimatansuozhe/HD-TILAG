from __future__ import annotations

import torch
from torch import nn

from hd_tilag.models.components import DualGraphEmbedding, TILAGNet


class HDTILAG(nn.Module):
    """Heterogeneous Data-fused Temporal Inter-Layer Attention-Gated model."""

    def __init__(
        self,
        time_feature_dim: int,
        sentiment_dim: int = 3,
        graph_dim: int = 32,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        per_stock_output: bool = True,
        num_stocks: int | None = None,
    ) -> None:
        super().__init__()
        self.graph = DualGraphEmbedding(time_feature_dim, graph_dim, dropout=dropout)
        fused_dim = time_feature_dim + sentiment_dim + 2 * graph_dim
        self.temporal = TILAGNet(
            input_dim=fused_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.per_stock_output = per_stock_output
        if per_stock_output:
            if num_stocks is None:
                raise ValueError("num_stocks is required when per_stock_output=True")
            self.stock_weight = nn.Parameter(torch.empty(num_stocks, hidden_size))
            self.stock_bias = nn.Parameter(torch.zeros(num_stocks))
        else:
            self.output = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if hasattr(self, "stock_weight"):
            nn.init.xavier_uniform_(self.stock_weight)
            nn.init.zeros_(self.stock_bias)

    def forward(
        self,
        time_features: torch.Tensor,
        sentiment_features: torch.Tensor,
        industry_adj: torch.Tensor,
        style_adj: torch.Tensor,
    ) -> torch.Tensor:
        graph_features = self.graph(time_features, industry_adj, style_adj)
        fused = torch.cat([time_features, sentiment_features, graph_features], dim=-1)
        embeddings = self.temporal(fused)
        if self.per_stock_output:
            logits = (embeddings * self.stock_weight.unsqueeze(0)).sum(dim=-1) + self.stock_bias
        else:
            logits = self.output(embeddings).squeeze(-1)
        return logits
