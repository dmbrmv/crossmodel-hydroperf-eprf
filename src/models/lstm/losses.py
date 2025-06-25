import torch
import torch.nn as nn


class NSELoss(nn.Module):
    """Negative Nash–Sutcliffe Efficiency (maximise NSE)."""

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        mse = torch.mean((y_hat - y) ** 2)
        var = torch.var(y, unbiased=False)
        nse = 1 - mse / var
        return -nse


class KGELoss(nn.Module):
    """Negative Kling–Gupta Efficiency (robust hydrology metric)."""

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        r = torch.corrcoef(torch.stack([y_hat, y]))[0, 1]
        alpha = torch.std(y_hat) / torch.std(y)
        beta = torch.mean(y_hat) / torch.mean(y)
        kge = 1 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
        return -kge


class WeightedMSELoss(nn.Module):
    """MSE weighted toward high-flow events.  α≈0.2-0.4 is typical."""

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        w = 1 + self.alpha * (y / y.mean())
        return torch.mean(w * (y_hat - y) ** 2)
