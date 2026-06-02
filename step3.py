"""
Step 2: Feature Extraction
==========================
Convert raw 1-D audio waveforms into 2-D feature matrices suitable for GRU input.

Features extracted per clip
---------------------------
  • MFCCs          — 40 coefficients  (captures timbral texture)
  • Mel Spectrogram — 128 bands       (full spectral shape)
  • Energy         — 1 value/frame    (loudness over time)

Final shape per clip: (Time_Steps, n_features)
  → fed to GRU as a sequence

Requirements:
    pip install librosa numpy pandas tqdm joblib
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "./UrbanSound8K"
META_CSV     = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR    = os.path.join(DATASET_ROOT, "audio")
SR           = 22050          # Hz — consistent for all clips
N_MFCC       = 40             # number of MFCC coefficients
N_MELS       = 128            # Mel filter-bank bands
HOP_LENGTH   = 512            # samples between frames
N_FFT        = 2048           # FFT window size
MAX_PAD_LEN  = 174            # ~4 s at SR=22050, hop=512  →  ⌈22050*4/512⌉ = 173
SAVE_DIR     = "./features"   # where .npy files are saved
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pad_or_truncate(arr: np.ndarray, max_len: int) -> np.ndarray:
    """
    Pad (with zeros) or truncate a feature matrix along the time axis
    so every clip becomes shape (max_len, n_features).

    arr shape: (n_features, time_steps)  → output: (max_len, n_features)
    """
    n_features, time_steps = arr.shape
    if time_steps < max_len:
        pad_width = max_len - time_steps
        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
    else:
        arr = arr[:, :max_len]
    return arr.T   # → (max_len, n_features)


def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Mel-Frequency Cepstral Coefficients.
    Returns shape (MAX_PAD_LEN, N_MFCC).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    return pad_or_truncate(mfcc, MAX_PAD_LEN)


def extract_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Log-Mel Spectrogram.
    Returns shape (MAX_PAD_LEN, N_MELS).
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                        n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    return pad_or_truncate(log_S, MAX_PAD_LEN)


def extract_energy(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Short-Time Energy per frame (RMS).
    Returns shape (MAX_PAD_LEN, 1).
    """
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)   # (1, frames)
    return pad_or_truncate(rms, MAX_PAD_LEN)   # (MAX_PAD_LEN, 1)


def extract_features(y: np.ndarray, sr: int, mode: str = "mfcc") -> np.ndarray:
    """
    Dispatch to the requested feature extractor.

    mode options
    ------------
    'mfcc'     → (MAX_PAD_LEN, 40)
    'mel'      → (MAX_PAD_LEN, 128)
    'combined' → (MAX_PAD_LEN, 40+1)   MFCC + Energy concatenated
    """
    if mode == "mfcc":
        return extract_mfcc(y, sr)
    elif mode == "mel":
        return extract_mel(y, sr)
    elif mode == "combined":
        mfcc  = extract_mfcc(y, sr)    # (T, 40)
        energy = extract_energy(y, sr) # (T, 1)
        return np.concatenate([mfcc, energy], axis=1)  # (T, 41)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


# ── Per-file worker ───────────────────────────────────────────────────────────
def process_row(row: pd.Series, audio_dir: str, mode: str):
    """Load one clip, extract features, return (feature_array, label)."""
    path = os.path.join(audio_dir, f"fold{row['fold']}", row["slice_file_name"])
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        feat  = extract_features(y, sr, mode=mode)
        return feat, int(row["classID"])
    except Exception as e:
        print(f"  [WARN] Skipped {path}: {e}")
        return None, None


# ── Build full dataset ────────────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame, audio_dir: str, mode: str = "mfcc",
                  n_jobs: int = 4):
    """
    Process all clips in parallel and save features + labels to disk.

    Saved files
    -----------
    features/{mode}_features.npy  — shape (N, MAX_PAD_LEN, n_features)
    features/{mode}_labels.npy    — shape (N,)
    features/{mode}_folds.npy     — shape (N,)  fold number for each clip
    """
    print(f"\n=== Extracting [{mode}] features from {len(df)} clips ===")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row, audio_dir, mode)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    features, labels, folds = [], [], []
    for (feat, label), (_, row) in zip(results, df.iterrows()):
        if feat is not None:
            features.append(feat)
            labels.append(label)
            folds.append(int(row["fold"]))

    X = np.stack(features)         # (N, T, F)
    y = np.array(labels)           # (N,)
    f = np.array(folds)            # (N,)

    np.save(os.path.join(SAVE_DIR, f"{mode}_features.npy"), X)
    np.save(os.path.join(SAVE_DIR, f"{mode}_labels.npy"),   y)
    np.save(os.path.join(SAVE_DIR, f"{mode}_folds.npy"),    f)

    print(f"  Features shape : {X.shape}")
    print(f"  Labels shape   : {y.shape}")
    print(f"  Saved to       : {SAVE_DIR}/")
    return X, y, f


# ── Train / Val / Test split ──────────────────────────────────────────────────
def get_splits(X: np.ndarray, y: np.ndarray, folds: np.ndarray):
    """
    Split by fold number:
      Train : folds 1–6
      Val   : folds 7–8
      Test  : folds 9–10
    """
    train_mask = folds <= 6
    val_mask   = (folds == 7) | (folds == 8)
    test_mask  = (folds == 9) | (folds == 10)

    splits = {
        "train": (X[train_mask], y[train_mask]),
        "val"  : (X[val_mask],   y[val_mask]),
        "test" : (X[test_mask],  y[test_mask]),
    }
    for name, (Xs, ys) in splits.items():
        print(f"  {name:>5}: {Xs.shape}  labels={ys.shape}")
    return splits


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(META_CSV)

    # Extract two feature sets: MFCC-only and Mel Spectrogram
    for mode in ["mfcc", "mel", "combined"]:
        X, y, folds = build_dataset(df, AUDIO_DIR, mode=mode, n_jobs=4)
        splits = get_splits(X, y, folds)

    print("\nFeature extraction complete. Files in ./features/")