import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import from custom modules
from configurations import DATASET_ROOT, META_CSV, CLASS_NAMES, DEVICE
from dataloader import (
    build_dataset, extract_mfcc, extract_melspectrogram, extract_energy,
    split_and_normalise, make_loaders
)
from evaluation import (
    train_model, plot_class_distribution, plot_waveform_and_spectrogram,
    plot_training_curves, plot_confusion_matrices, plot_final_comparison
)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print(f"Device : {DEVICE}")

    # ── Validate dataset path ─────────────────────────────────────────────────
    assert DATASET_ROOT.exists(), (
        f"\n[ERROR] Dataset not found at '{DATASET_ROOT}'.\n"
        "  Edit DATASET_ROOT in configurations.py.\n"
    )

    meta = pd.read_csv(META_CSV)
    print(f"Total clips : {len(meta)}")
    print(f"Classes     : {sorted(meta['class'].unique())}\n")

    # ── Explore dataset ───────────────────────────────────────────────────────
    plot_class_distribution(meta)
    plot_waveform_and_spectrogram(meta, n_classes=3)

    # ── FEATURE EXTRACTION ────────────────────────────────────────────────────
    print("\n[1/3] MFCC features")
    X_mfcc, y_all, folds_all = build_dataset(meta, extract_mfcc, "cache_mfcc.npz")

    print("\n[2/3] Mel-Spectrogram features")
    X_mel, _, _ = build_dataset(meta, extract_melspectrogram, "cache_mel.npz")

    print("\n[3/3] RMS Energy features")
    X_energy, _, _ = build_dataset(meta, extract_energy, "cache_energy.npz")

    # ── SPLIT & NORMALISE ─────────────────────────────────────────────────────
    print("\n--- Splitting folds ---")
    print("MFCC:")
    mfcc_train, mfcc_val, mfcc_test = split_and_normalise(X_mfcc, y_all, folds_all)
    print("Mel-Spectrogram:")
    mel_train, mel_val, mel_test     = split_and_normalise(X_mel,  y_all, folds_all)
    print("Energy:")
    en_train, en_val, en_test        = split_and_normalise(X_energy, y_all, folds_all)

    # ── DATALOADERS ───────────────────────────────────────────────────────────
    mfcc_loaders   = make_loaders(mfcc_train, mfcc_val, mfcc_test)
    mel_loaders    = make_loaders(mel_train,  mel_val,  mel_test)
    energy_loaders = make_loaders(en_train,   en_val,   en_test)

    # ── TRAIN 3 GRU MODELS ────────────────────────────────────────────────────
    hist_mfcc,   acc_mfcc,   f1_mfcc,   pred_mfcc,   true_mfcc   = train_model("MFCC",   40,  mfcc_loaders)
    hist_mel,    acc_mel,    f1_mel,    pred_mel,    true_mel    = train_model("Mel",    128, mel_loaders)
    hist_energy, acc_energy, f1_energy, pred_energy, true_energy = train_model("Energy", 1,   energy_loaders)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    plot_training_curves(
        [hist_mfcc, hist_mel, hist_energy],
        ["GRU-MFCC", "GRU-Mel", "GRU-Energy"],
        ["royalblue", "seagreen", "darkorange"],
    )

    plot_confusion_matrices(
        [true_mfcc,  true_mel,  true_energy],
        [pred_mfcc,  pred_mel,  pred_energy],
        ["GRU-MFCC", "GRU-Mel", "GRU-Energy"],
    )

    # ── FINAL COMPARISON TABLE ────────────────────────────────────────────────
    results = pd.DataFrame([
        {"Feature": "MFCCs (40-dim)",           "Accuracy (%)": round(acc_mfcc   * 100, 2), "F1-Score": round(f1_mfcc,   4)},
        {"Feature": "Mel-Spectrogram (128-dim)", "Accuracy (%)": round(acc_mel    * 100, 2), "F1-Score": round(f1_mel,    4)},
        {"Feature": "RMS Energy (1-dim)",        "Accuracy (%)": round(acc_energy * 100, 2), "F1-Score": round(f1_energy, 4)},
    ])

    plot_final_comparison(results)

    print("\n" + "=" * 60)
    print("          FINAL COMPARISON — TEST SET")
    print("=" * 60)
    print(results.to_string(index=False))
    print("=" * 60)

    best_row = results.loc[results["Accuracy (%)"].idxmax()]
    print(f"\n  Best model : {best_row['Feature']}  →  Acc={best_row['Accuracy (%)']}%   F1={best_row['F1-Score']}")

    best_idx   = results["Accuracy (%)"].idxmax()
    best_name  = ["MFCC", "Mel-Spectrogram", "Energy"][best_idx]
    best_preds = [pred_mfcc, pred_mel, pred_energy][best_idx]
    best_true  = [true_mfcc, true_mel, true_energy][best_idx]

    print(f"\n--- Per-class report: GRU-{best_name} (best model) ---\n")
    print(classification_report(best_true, best_preds, target_names=CLASS_NAMES))

    print("\n--- Most Confused Class Pairs ---")
    for name, true, preds in [("MFCC", true_mfcc, pred_mfcc), ("Mel", true_mel, pred_mel), ("Energy", true_energy, pred_energy)]:
        cm = confusion_matrix(true, preds)
        np.fill_diagonal(cm, 0)
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        print(f"  GRU-{name:6s}: {CLASS_NAMES[i]:20s} → {CLASS_NAMES[j]:20s}  ({cm[i,j]} errors)")