import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

from configurations import DEVICE, LR, EPOCHS, SEED, AUDIO_DIR, SR, N_MELS, CLASS_NAMES
from gru import GRUAttentionClassifier, CNNGRUClassifier
from dataloader import load_audio


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_sum += loss.item() * len(yb)
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += len(yb)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []

    for Xb, yb in loader:
        Xb, yb  = Xb.to(DEVICE), yb.to(DEVICE)
        logits  = model(Xb)
        loss    = criterion(logits, yb)
        preds   = logits.argmax(1)

        loss_sum += loss.item() * len(yb)
        correct  += (preds == yb).sum().item()
        total    += len(yb)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_true, all_preds, average="weighted")
    return loss_sum / total, acc, f1, np.array(all_preds), np.array(all_true)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(name: str, input_size: int, loaders: tuple,
                model_type: str = "gru"):
    """
    Train one model variant.

    Parameters
    ----------
    name        : string label, e.g. "MFCC"
    input_size  : F dimension of the feature tensor
    loaders     : (train_loader, val_loader, test_loader)
    model_type  : "gru"  or  "cnn_gru"  (bonus)
    """
    tr_loader, vl_loader, te_loader = loaders

    # ── Build model ──────────────────────────────────────────────────────────
    if model_type == "cnn_gru":
        model = CNNGRUClassifier(input_size=input_size).to(DEVICE)
        tag   = "CNN-GRU"
    else:
        model = GRUAttentionClassifier(input_size=input_size).to(DEVICE)
        tag   = "GRU"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*65}")
    print(f"  Training {tag} — Feature: {name}   "
          f"input_size={input_size}   params={n_params:,}")
    print(f"{'='*65}")

    # ── Optimiser + scheduler + loss ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    # Cosine annealing: warmup 5 epochs, then cosine decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 5,
        epochs=EPOCHS,
        steps_per_epoch=len(tr_loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )

    history  = {k: [] for k in
                ["train_loss", "train_acc", "train_f1",
                 "val_loss",   "val_acc",   "val_f1"]}
    best_acc = 0.0
    patience = 20        # early stopping patience (epochs)
    no_improve = 0
    ckpt     = f"best_{tag.lower().replace('-','_')}_{name}.pt"

    for ep in range(1, EPOCHS + 1):
        tl, _             = train_epoch(model, tr_loader, criterion, optimizer)
        _,  ta, tf, _, _  = evaluate(model, tr_loader, criterion)
        vl, va, vf, _, _  = evaluate(model, vl_loader, criterion)

        for k, v in zip(history.keys(), [tl, ta, tf, vl, va, vf]):
            history[k].append(v)

        if va > best_acc:
            best_acc   = va
            no_improve = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{EPOCHS} | "
                  f"Train Acc={ta:.4f} F1={tf:.4f} | "
                  f"Val Acc={va:.4f} F1={vf:.4f}  "
                  f"[best={best_acc:.4f}]")

        if no_improve >= patience:
            print(f"  ✗ Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

        # OneCycleLR steps per batch (already done inside train_epoch)
        # — but we placed it outside; step it here per epoch instead:
        # (uncomment the line below if you prefer epoch-level stepping)
        # scheduler.step()

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    _, test_acc, test_f1, preds, true = evaluate(model, te_loader, criterion)

    print(f"\n  ► TEST  Accuracy={test_acc*100:.2f}%   F1={test_f1:.4f}")
    return history, test_acc, test_f1, preds, true


# NOTE: OneCycleLR must step every *batch*, not every epoch.
# Replace train_epoch to accept the scheduler:

def train_epoch_with_scheduler(model, loader, criterion, optimizer, scheduler):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()      # ← per-batch step for OneCycleLR

        loss_sum += loss.item() * len(yb)
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += len(yb)

    return loss_sum / total, correct / total


def train_model_v2(name: str, input_size: int, loaders: tuple,
                   model_type: str = "gru"):
    """
    Recommended entry point — uses per-batch OneCycleLR stepping.
    Drop-in replacement for train_model().
    """
    tr_loader, vl_loader, te_loader = loaders

    if model_type == "cnn_gru":
        model = CNNGRUClassifier(input_size=input_size).to(DEVICE)
        tag   = "CNN-GRU"
    else:
        model = GRUAttentionClassifier(input_size=input_size).to(DEVICE)
        tag   = "GRU"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*65}")
    print(f"  Training {tag} — Feature: {name}   "
          f"input_size={input_size}   params={n_params:,}")
    print(f"{'='*65}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 5,
        epochs=EPOCHS,
        steps_per_epoch=len(tr_loader),
        pct_start=0.05,
        anneal_strategy="cos",
    )

    history    = {k: [] for k in
                  ["train_loss", "train_acc", "train_f1",
                   "val_loss",   "val_acc",   "val_f1"]}
    best_acc   = 0.0
    patience   = 20
    no_improve = 0
    ckpt       = f"best_{tag.lower().replace('-','_')}_{name}.pt"

    for ep in range(1, EPOCHS + 1):
        tl, _             = train_epoch_with_scheduler(
                                model, tr_loader, criterion, optimizer, scheduler)
        _,  ta, tf, _, _  = evaluate(model, tr_loader, criterion)
        vl, va, vf, _, _  = evaluate(model, vl_loader, criterion)

        for k, v in zip(history.keys(), [tl, ta, tf, vl, va, vf]):
            history[k].append(v)

        if va > best_acc:
            best_acc   = va
            no_improve = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{EPOCHS} | "
                  f"Train Acc={ta:.4f} F1={tf:.4f} | "
                  f"Val Acc={va:.4f} F1={vf:.4f}  "
                  f"[best={best_acc:.4f}]")

        if no_improve >= patience:
            print(f"  ✗ Early stopping at epoch {ep}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    _, test_acc, test_f1, preds, true = evaluate(model, te_loader, criterion)

    print(f"\n  ► TEST  Accuracy={test_acc*100:.2f}%   F1={test_f1:.4f}")
    return history, test_acc, test_f1, preds, true


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers  (unchanged API, same as original)
# ──────────────────────────────────────────────────────────────────────────────

def plot_waveform_and_spectrogram(meta_df: pd.DataFrame, n_classes: int = 3):
    print("\n--- Waveform & Spectrogram Examples ---")
    sampled = (meta_df.groupby("classID")
               .apply(lambda g: g.sample(1, random_state=SEED))
               .reset_index(drop=True)
               .head(n_classes))

    for _, row in sampled.iterrows():
        path  = AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
        y     = load_audio(str(path))
        label = row["class"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 3))
        librosa.display.waveshow(y, sr=SR, ax=axes[0], alpha=0.8)
        axes[0].set(title=f"Waveform  —  {label}",
                    xlabel="Time (s)", ylabel="Amplitude")

        S_dB = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS), ref=np.max)
        img = librosa.display.specshow(S_dB, sr=SR,
                                       x_axis="time", y_axis="mel", ax=axes[1])
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
        axes[1].set_title(f"Mel-Spectrogram  —  {label}")

        plt.suptitle(
            f"Duration: {len(y)/SR:.2f}s  |  File: {row['slice_file_name']}",
            fontsize=10)
        plt.tight_layout()
        plt.savefig(f"example_{label}.png", dpi=120)
        plt.show()


def plot_class_distribution(meta_df):
    fig, ax = plt.subplots(figsize=(12, 4))
    meta_df["class"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set(title="Class Distribution — UrbanSound8K",
           xlabel="Class", ylabel="Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=120)
    plt.show()


def plot_training_curves(histories, names, colors):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(
        axes,
        ["val_loss", "val_acc", "val_f1"],
        ["Validation Loss", "Validation Accuracy", "Validation F1-Score"],
    ):
        for hist, name, color in zip(histories, names, colors):
            ax.plot(hist[key], label=name, color=color, linewidth=2)
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("GRU Training Curves — Feature Comparison", fontsize=15)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrices(all_true, all_preds, names):
    fig, axes = plt.subplots(1, len(names), figsize=(9 * len(names), 8))
    if len(names) == 1:
        axes = [axes]

    for ax, true, preds, name in zip(axes, all_true, all_preds, names):
        cm = confusion_matrix(true, preds, normalize="true") * 100
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.4, ax=ax, cbar=False)
        ax.set(title=name, xlabel="Predicted", ylabel="True")
        ax.tick_params(axis="x", rotation=40)

        cm_counts = confusion_matrix(true, preds)
        np.fill_diagonal(cm_counts, 0)
        i, j = np.unravel_index(cm_counts.argmax(), cm_counts.shape)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                    edgecolor="red", lw=2.5))

    plt.suptitle(
        "Confusion Matrices — Normalised (%)  |  red = most confused pair",
        fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()


def plot_final_comparison(results_df):
    n   = len(results_df)
    colors = ["royalblue", "seagreen", "darkorange", "crimson"][:n]
    labels = results_df["Feature"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, ylabel in zip(
        axes,
        ["Accuracy (%)", "F1-Score"],
        ["Test Accuracy (%)", "Test Weighted F1-Score"],
    ):
        bars = ax.bar(labels, results_df[col],
                      color=colors, edgecolor="black", width=0.45)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(results_df[col]) * 1.18)
        ax.set(title=ylabel, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("GRU Feature Comparison — Test Set Results", fontsize=15)
    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=150)
    plt.show()