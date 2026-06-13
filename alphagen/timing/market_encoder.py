import torch
from torch import nn


class MarketEncoder(nn.Module):
    """Map 21 raw CSI300 market features into a 16-dimensional market embedding."""

    def __init__(self, d_input: int = 21, d_model: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.Tanh(),
        )

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        return self.net(market_features)


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output
