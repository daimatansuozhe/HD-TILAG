from __future__ import annotations

import math

import torch
from torch import nn


class DualGraphEmbedding(nn.Module):
    """Dual-branch GCN over industry and style relation matrices."""

    def __init__(self, input_dim: int, graph_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.industry = nn.Linear(input_dim, graph_dim, bias=False)
        self.style = nn.Linear(input_dim, graph_dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        time_features: torch.Tensor,
        industry_adj: torch.Tensor,
        style_adj: torch.Tensor,
    ) -> torch.Tensor:
        # time_features: [B, W, N, dx], adjacency: [N, N]
        industry_msg = torch.einsum("ij,btjf->btif", industry_adj, time_features)
        style_msg = torch.einsum("ij,btjf->btif", style_adj, time_features)
        industry_out = self.activation(self.industry(industry_msg))
        style_out = self.activation(self.style(style_msg))
        return self.dropout(torch.cat([industry_out, style_out], dim=-1))


class AttentionGatedUnit(nn.Module):
    """Attention-Gated Unit from Eq. (9)-(14)."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha_proj = nn.Linear(hidden_size, hidden_size)

        self.reset_input = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        self.candidate_input = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_hidden = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, z_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        batch_shape = z_t.shape[:-1]
        q = self.q_proj(z_t).reshape(*batch_shape, self.num_heads, self.head_dim)
        k = self.k_proj(h_prev).reshape(*batch_shape, self.num_heads, self.head_dim)
        v = self.v_proj(h_prev).reshape(*batch_shape, self.num_heads, self.head_dim)

        attn_out = torch.sigmoid((q * k) / math.sqrt(self.head_dim)) * v
        attn_out = attn_out.reshape(*batch_shape, self.hidden_size)
        alpha = torch.sigmoid(self.alpha_proj(attn_out))

        reset = torch.sigmoid(self.reset_input(z_t) + self.reset_hidden(h_prev))
        candidate = torch.tanh(self.candidate_input(z_t) + self.candidate_hidden(h_prev * reset))
        return (1.0 - alpha) * candidate + alpha * h_prev


class InterLayerGatingUnit(nn.Module):
    """Inter-layer Gating Unit from Eq. (15)-(16)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.upper_context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lower_context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=True)
        self.activation = nn.ReLU()

    def forward(self, upper_prev: torch.Tensor, lower_current: torch.Tensor) -> torch.Tensor:
        evidence = self.activation(self.upper_context(upper_prev) + self.lower_context(lower_current))
        gate = torch.sigmoid(self.projection(evidence))
        return gate * lower_current


class TILAGNet(nn.Module):
    """Stacked AGU recurrent encoder with IGU inter-layer filtering."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.layers = nn.ModuleList(
            [AttentionGatedUnit(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.igus = nn.ModuleList(
            [InterLayerGatingUnit(hidden_size) for _ in range(max(0, num_layers - 1))]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B, W, N, F]. Stocks are encoded in parallel by flattening B*N.
        batch, window, stocks, _ = sequence.shape
        x = self.input_projection(sequence).permute(0, 2, 1, 3).reshape(
            batch * stocks, window, self.hidden_size
        )
        states = [
            x.new_zeros(batch * stocks, self.hidden_size)
            for _ in range(self.num_layers)
        ]
        for t in range(window):
            lower_output = self.layers[0](x[:, t, :], states[0])
            states[0] = lower_output
            for layer_idx in range(1, self.num_layers):
                gated_input = self.igus[layer_idx - 1](states[layer_idx], states[layer_idx - 1])
                upper_output = self.layers[layer_idx](gated_input, states[layer_idx])
                states[layer_idx] = upper_output
        final = self.dropout(states[-1]).reshape(batch, stocks, self.hidden_size)
        return final
