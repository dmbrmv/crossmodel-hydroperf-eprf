import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    """MLP that embeds raw static features."""

    def __init__(self, n_static: int, emb_dim: int = 32, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 2 * emb_dim
        self.net = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim))

    def forward(self, x: torch.Tensor):
        return self.net(x)  # [B, emb_dim]


class TemporalEncoder(nn.Module):
    """Backbone LSTM (swap later for GRU, Transformer, GNN, …)."""

    def __init__(self, n_dyn: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_dyn,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, h0: tuple[torch.Tensor, torch.Tensor] | None = None):
        _, (h_n, _) = self.lstm(x, h0)
        return h_n[-1]  # [B, hidden]


class HydroLSTM(nn.Module):
    """
    Complete network with three static-feature fusion modes:
    'repeat' | 'init' | 'late'   (default = repeat)
    """

    def __init__(
        self,
        n_dyn: int,
        n_static: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        static_emb: int = 32,
        static_mode: str = "repeat",
    ):
        super().__init__()
        assert static_mode in {"repeat", "init", "late"}
        self.static_mode = static_mode
        self.static_enc = StaticEncoder(n_static, static_emb)
        self.temp_enc = TemporalEncoder(
            n_dyn + (static_emb if static_mode == "repeat" else 0), hidden_size, num_layers
        )
        self.head = nn.Linear(hidden_size + (static_emb if static_mode == "late" else 0), 1)

    def forward(self, seq: torch.Tensor, static: torch.Tensor):
        """
        seq    – [B, T, n_dyn]
        static – [B, n_static]
        """
        s = self.static_enc(static)  # [B, E]

        if self.static_mode == "repeat":
            s_rep = s.unsqueeze(1).expand(-1, seq.size(1), -1)
            h = self.temp_enc(torch.cat([seq, s_rep], dim=-1))
            out = self.head(h)

        elif self.static_mode == "init":
            h0 = s.unsqueeze(0).repeat(self.temp_enc.lstm.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            h = self.temp_enc(seq, (h0, c0))
            out = self.head(h)

        else:  # late fusion
            h = self.temp_enc(seq)
            out = self.head(torch.cat([h, s], dim=-1))

        return out.squeeze(-1)  # [B]
