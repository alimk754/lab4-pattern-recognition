import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _length_mask(lengths: torch.Tensor, max_len: int,
                 device: torch.device) -> torch.Tensor:
    """Return bool mask  (B, T)  True for valid positions."""
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class GRUAttentionClassifier(nn.Module):
    """
    High-accuracy GRU classifier for audio sequences.

    Architecture
    ------------
    Input  (B, T, F)
        → LayerNorm
        → Projection  (F  →  proj_size)       # optional dim reduction
        → BiGRU  ×  num_layers  (with residual skip every 2 layers)
        → Multi-head attention pooling over T
        → Dropout  →  FC-256  →  ReLU  →  Dropout  →  FC-C
        → logits (B, C)

    Key improvements over the baseline
    -----------------------------------
    • Residual connections between GRU layer pairs
    • Multi-head attention pooling (richer temporal summary)
    • Input projection layer (decouples feature dim from GRU dim)
    • Larger hidden size with more dropout (regularisation)
    • Label-smoothing-aware design (works with nn.CrossEntropyLoss label_smoothing)
    """

    def __init__(
        self,
        input_size:   int,
        proj_size:    int   = 128,   # project input to this dim before GRU
        hidden_size:  int   = 256,   # GRU units per direction
        num_layers:   int   = 3,
        num_classes:  int   = 10,
        dropout:      float = 0.4,
        num_heads:    int   = 4,     # attention heads
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers

        # ── Input normalisation + projection ─────────────────────────────────
        self.input_norm = nn.LayerNorm(input_size)
        self.proj       = nn.Sequential(
            nn.Linear(input_size, proj_size),
            nn.ReLU(),
        ) if input_size != proj_size else nn.Identity()

        proj_dim = proj_size if input_size != proj_size else input_size
        gru_out  = hidden_size * self.num_directions

        # ── Stacked BiGRU layers (residual every 2 layers) ───────────────────
        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = proj_dim if i == 0 else gru_out
            self.gru_layers.append(nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            ))
            self.layer_norms.append(nn.LayerNorm(gru_out))

        # Residual projection when dimensions change (layer 0 → layer 2+)
        self.res_proj = nn.Linear(proj_dim, gru_out) if proj_dim != gru_out else nn.Identity()

        # ── Multi-head self-attention pooling ─────────────────────────────────
        self.attn_heads = num_heads
        self.attn_proj  = nn.Linear(gru_out, num_heads)   # (B, T, heads)
        self.out_proj   = nn.Linear(gru_out * num_heads, gru_out)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, T, F)
        lengths : (B,) original sequence lengths — optional

        Returns
        -------
        logits  : (B, num_classes)
        """
        # 1. Normalise + project
        out = self.input_norm(x)
        out = self.proj(out)          # (B, T, proj_dim)

        # 2. Stacked GRU with residuals
        prev = out
        for i, (gru, ln) in enumerate(zip(self.gru_layers, self.layer_norms)):
            gru_out, _ = gru(out)     # (B, T, gru_out)
            gru_out    = ln(gru_out)

            # Residual: add prev every 2 layers (skip connection)
            if i == 0:
                # First layer: project residual if needed, then add
                res  = self.res_proj(prev)
                out  = gru_out + res
            elif i % 2 == 0:
                out  = gru_out + out   # same dim residual
            else:
                out  = gru_out         # no residual on odd layers
            prev = out

        # 3. Multi-head attention pooling
        attn_w  = self.attn_proj(out)        # (B, T, H)
        if lengths is not None:
            mask   = _length_mask(lengths, out.size(1), x.device)   # (B, T)
            attn_w = attn_w.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        attn_w  = torch.softmax(attn_w, dim=1)     # (B, T, H)
        # Weighted sum for each head
        # out: (B, T, D)  attn_w: (B, T, H)
        heads   = []
        for h in range(self.attn_heads):
            w_h = attn_w[:, :, h].unsqueeze(-1)    # (B, T, 1)
            heads.append((w_h * out).sum(dim=1))    # (B, D)
        context = torch.cat(heads, dim=-1)          # (B, D*H)
        context = self.out_proj(context)            # (B, D)

        return self.classifier(context)


# ──────────────────────────────────────────────────────────────────────────────
# Bonus: CNN-GRU hybrid
# ──────────────────────────────────────────────────────────────────────────────

class CNNGRUClassifier(nn.Module):
    """
    CNN front-end  →  Bidirectional GRU  →  Attention pooling.

    The CNN stack extracts local temporal patterns (like n-grams) before
    passing the compressed sequence to the GRU — often +3–5 % over GRU alone.

    Architecture
    ------------
    Input  (B, T, F)
        → reshape to (B, 1, T, F)   [treat as single-channel 2-D image]
        → 3 × Conv2d blocks  (channel 1 → 32 → 64 → 128)
           each block: Conv → BN → GELU → MaxPool(freq only)
        → flatten freq+channel dim  →  (B, T', cnn_out_dim)
        → BiGRU (2 layers)
        → Attention pooling
        → FC head  →  logits
    """

    def __init__(
        self,
        input_size:  int,            # F — number of feature bins
        num_classes: int  = 10,
        hidden_size: int  = 256,
        num_layers:  int  = 2,
        dropout:     float = 0.4,
    ):
        super().__init__()

        # ── CNN front-end (operates on freq axis) ─────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1: (B, 1, T, F) → (B, 32, T, F//2)
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),   # halve freq dim
            nn.Dropout2d(0.1),

            # Block 2: (B, 32, T, F//2) → (B, 64, T, F//4)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(0.1),

            # Block 3: (B, 64, T, F//4) → (B, 128, T, F//8)
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(0.1),
        )

        # Compute flattened dim after CNN
        freq_after = input_size // 8
        cnn_out_dim = 128 * freq_after

        # ── GRU ───────────────────────────────────────────────────────────────
        self.gru_norm = nn.LayerNorm(cnn_out_dim)
        self.gru      = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        gru_out = hidden_size * 2
        self.attn = nn.Linear(gru_out, 1)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, F)"""
        B, T, F = x.shape

        # CNN  →  (B, C, T, F')
        x_img = x.unsqueeze(1)              # (B, 1, T, F)
        cnn_out = self.cnn(x_img)           # (B, 128, T, F//8)

        # Merge channel and freq dims  →  (B, T, 128*F//8)
        _, C, T2, F2 = cnn_out.shape
        seq = cnn_out.permute(0, 2, 1, 3).reshape(B, T2, C * F2)

        # GRU
        seq = self.gru_norm(seq)
        out, _ = self.gru(seq)              # (B, T2, gru_out)

        # Attention pooling
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T2, 1)
        context = (attn_w * out).sum(dim=1)             # (B, gru_out)

        return self.classifier(context)