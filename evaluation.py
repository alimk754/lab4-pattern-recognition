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
from gru import GRUAttentionClassifier
from dataloader import load_audio

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
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

def train_model(name: str, input_size: int, loaders: tuple):
    tr_loader, vl_loader, te_loader = loaders

    model     = GRUAttentionClassifier(input_size=input_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history  = {k: [] for k in ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]}
    best_acc = 0.0
    ckpt     = f"best_gru_{name}.pt"

    print(f"\n{'='*60}\n  Training GRU — Feature: {name}   input_size={input_size}\n{'='*60}")

    for ep in range(1, EPOCHS + 1):
        tl, _            = train_epoch(model, tr_loader, criterion, optimizer)
        _,  ta, tf, _, _ = evaluate(model, tr_loader, criterion)
        vl, va, vf, _, _ = evaluate(model, vl_loader, criterion)
        scheduler.step(vl)

        for k, v in zip(history.keys(), [tl, ta, tf, vl, va, vf]):
            history[k].append(v)

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), ckpt)

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{EPOCHS} | Train Acc={ta:.4f} F1={tf:.4f} | Val Acc={va:.4f} F1={vf:.4f}")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    _, test_acc, test_f1, preds, true = evaluate(model, te_loader, criterion)

    print(f"\n  ► TEST  Accuracy={test_acc*100:.2f}%   F1={test_f1:.4f}")
    return history, test_acc, test_f1, preds, true

def plot_waveform_and_spectrogram(meta_df: pd.DataFrame, n_classes: int = 3):
    print("\n--- Waveform & Spectrogram Examples ---")
    sampled = meta_df.groupby("classID").apply(lambda g: g.sample(1, random_state=SEED)).reset_index(drop=True).head(n_classes)
    for _, row in sampled.iterrows():
        path = AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
        y    = load_audio(str(path))
        label = row["class"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 3))
        librosa.display.waveshow(y, sr=SR, ax=axes[0], alpha=0.8)
        axes[0].set(title=f"Waveform  —  {label}", xlabel="Time (s)", ylabel="Amplitude")

        S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS), ref=np.max)
        img = librosa.display.specshow(S_dB, sr=SR, x_axis="time", y_axis="mel", ax=axes[1])
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
        axes[1].set_title(f"Mel-Spectrogram  —  {label}")

        plt.suptitle(f"Duration: {len(y)/SR:.2f}s  |  File: {row['slice_file_name']}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"example_{label}.png", dpi=120)
        plt.show()

def plot_class_distribution(meta_df):
    fig, ax = plt.subplots(figsize=(12, 4))
    meta_df["class"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set(title="Class Distribution — UrbanSound8K", xlabel="Class", ylabel="Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=120)
    plt.show()

def plot_training_curves(histories, names, colors):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(axes, ["val_loss", "val_acc", "val_f1"], ["Validation Loss", "Validation Accuracy", "Validation F1-Score"]):
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
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, true, preds, name in zip(axes, all_true, all_preds, names):
        cm = confusion_matrix(true, preds, normalize="true") * 100
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.4, ax=ax, cbar=False)
        ax.set(title=name, xlabel="Predicted", ylabel="True")
        ax.tick_params(axis="x", rotation=40)

        cm_counts = confusion_matrix(true, preds)
        np.fill_diagonal(cm_counts, 0)
        i, j = np.unravel_index(cm_counts.argmax(), cm_counts.shape)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2.5))
    plt.suptitle("Confusion Matrices — Normalised (%)  |  red = most confused pair", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()

def plot_final_comparison(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors, labels = ["royalblue", "seagreen", "darkorange"], ["MFCCs\n(40-dim)", "Mel-Spec\n(128-dim)", "Energy\n(1-dim)"]
    for ax, col, ylabel in zip(axes, ["Accuracy (%)", "F1-Score"], ["Test Accuracy (%)", "Test Weighted F1-Score"]):
        bars = ax.bar(labels, results_df[col], color=colors, edgecolor="black", width=0.45)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(results_df[col]) * 1.18)
        ax.set(title=ylabel, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("GRU Feature Comparison — Test Set Results", fontsize=15)
    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=150)
    plt.show()