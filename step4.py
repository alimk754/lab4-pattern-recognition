"""
Step 4: Evaluation — Accuracy, F1-Score & Confusion Matrices
=============================================================
Load saved model checkpoints, run on the test folds (9 & 10),
and produce:
  • per-class accuracy & macro F1
  • confusion matrix heatmap
  • side-by-side comparison: MFCC-only vs Mel Spectrogram

Requirements:
    pip install torch scikit-learn matplotlib seaborn numpy
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
from torch.utils.data import DataLoader, TensorDataset

# ── Reuse definitions from step3 ─────────────────────────────────────────────
import sys
sys.path.insert(0, ".")
from step3_model import GRUClassifier, load_splits, normalise, make_loader

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURE_DIR = "./features"
MODEL_DIR   = "./models"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_CLASSES = 10
BATCH_SIZE  = 64

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing",
    "dog_bark",        "drilling", "engine_idling",
    "gun_shot",        "jackhammer", "siren", "street_music",
]
# ──────────────────────────────────────────────────────────────────────────────


# ── 1.  Inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        preds   = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


def load_model(mode: str, input_dim: int):
    path  = os.path.join(MODEL_DIR, f"best_gru_{mode}.pth")
    model = GRUClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                          num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                          dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


# ── 2.  Confusion matrix plot ─────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, mode: str,
                           save_fig: bool = True, ax=None):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_title(f"Confusion Matrix — {mode.upper()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=9)
    if save_fig and ax.get_figure():
        fname = f"step4_confusion_{mode}.png"
        ax.get_figure().savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Saved → {fname}")


# ── 3.  Per-class bar chart ───────────────────────────────────────────────────
def plot_per_class_f1(y_true, y_pred, mode: str,
                       save_fig: bool = True, ax=None):
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    colors = ["#2196F3" if v >= 0.7 else "#FF9800" if v >= 0.5 else "#F44336"
              for v in per_class_f1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASS_NAMES, per_class_f1, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Per-Class F1 Score — {mode.upper()}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.7, color="green", linestyle="--", linewidth=1, label="0.7 threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    if save_fig and ax.get_figure():
        fname = f"step4_f1_{mode}.png"
        ax.get_figure().savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Saved → {fname}")


# ── 4.  Comparison table ──────────────────────────────────────────────────────
def comparison_table(results: dict, save_fig: bool = True):
    """Bar chart comparing Accuracy and F1 across feature modes."""
    modes  = list(results.keys())
    accs   = [results[m]["accuracy"] for m in modes]
    f1s    = [results[m]["f1"]       for m in modes]

    x     = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy", color="#2196F3", edgecolor="black")
    ax.bar(x + width/2, f1s,  width, label="F1 (macro)", color="#4CAF50", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Feature Sets", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (acc, f1) in enumerate(zip(accs, f1s)):
        ax.text(i - width/2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=10)
        ax.text(i + width/2, f1  + 0.01, f"{f1:.3f}",  ha="center", fontsize=10)

    plt.tight_layout()
    if save_fig:
        plt.savefig("step4_comparison.png", dpi=120, bbox_inches="tight")
        print("Saved → step4_comparison.png")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    for mode in ["mfcc", "mel"]:
        print(f"\n{'='*55}")
        print(f"  Evaluating feature mode: {mode.upper()}")
        print(f"{'='*55}")

        # Load test split
        _, _, _, _, X_te, y_te = load_splits(FEATURE_DIR, mode)
        X_tr, _, X_vl, _, _, _ = load_splits(FEATURE_DIR, mode)
        X_tr, X_vl, X_te = normalise(X_tr, X_vl, X_te)
        test_loader = make_loader(X_te, y_te, BATCH_SIZE, shuffle=False)

        # Load model
        input_dim = X_te.shape[2]
        model     = load_model(mode, input_dim)

        # Predict
        y_pred, y_true = predict(model, test_loader, DEVICE)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f} %)")
        print(f"  F1 (macro): {f1:.4f}")
        print("\n  Per-class report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

        results[mode] = {"accuracy": acc, "f1": f1,
                         "y_true": y_true, "y_pred": y_pred}

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_confusion_matrix(y_true, y_pred, mode, save_fig=True, ax=ax)
        plt.tight_layout()
        plt.savefig(f"step4_confusion_{mode}.png", dpi=120, bbox_inches="tight")
        plt.close()

        # Per-class F1
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_per_class_f1(y_true, y_pred, mode, save_fig=True, ax=ax)
        plt.tight_layout()
        plt.savefig(f"step4_f1_{mode}.png", dpi=120, bbox_inches="tight")
        plt.close()

    # Cross-feature comparison
    comparison_table(results)

    print("\n✓ All evaluation plots saved.")