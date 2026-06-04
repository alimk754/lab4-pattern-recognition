import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """
    Bidirectional multi-layer GRU for sequence classification.

    Architecture
    ------------
    Input  (B, T, F)
        → LayerNorm
        → BiGRU  ×  num_layers
        → Attention-pooling over T
        → Dropout  →  Linear  →  logits (B, C)

    Parameters
    ----------
    input_size   : number of input features F
    hidden_size  : GRU hidden units per direction
    num_layers   : stacked GRU layers
    num_classes  : output classes (default = 10)
    dropout      : dropout between GRU layers and before FC
    bidirectional: use bidirectional GRU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int  = 2,
        num_classes: int = 10,
        dropout: float   = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size   = hidden_size

        self.norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_size * self.num_directions

        # Learnable temporal attention
        self.attn = nn.Linear(gru_out_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, T, F)
        lengths : (B,) original sequence lengths (optional, for masking)

        Returns
        -------
        logits  : (B, num_classes)
        """
        x = self.norm(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)         # (B, T, D)

        # Temporal attention pooling
        attn_w = self.attn(out)          # (B, T, 1)
        if lengths is not None:
            # mask padding positions
            mask = _length_mask(lengths, out.size(1), x.device)
            attn_w = attn_w.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn_w = torch.softmax(attn_w, dim=1)
        context = (attn_w * out).sum(dim=1)  # (B, D)

        return self.classifier(context)
