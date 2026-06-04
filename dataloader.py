import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from configurations import (
    SR, N_MFCC, N_MELS, HOP_LENGTH, N_FFT, AUDIO_DIR, BATCH_SIZE, MAX_FRAMES, SEED
)

# ──────────────────────────────────────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = SR) -> np.ndarray:
    """Load audio, resample to target_sr, return mono float32 array."""
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y


def pad_or_truncate(y: np.ndarray, sr: int = SR, duration: float = 4.0) -> np.ndarray:
    """Pad with zeros or truncate so every clip is exactly `duration` seconds."""
    target_len = int(sr * duration)
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


# ──────────────────────────────────────────────────────────────────────────────
# Data augmentation  (applied in raw-audio space, training only)
# ──────────────────────────────────────────────────────────────────────────────

def augment_audio(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Random chain of: additive noise → time-shift → pitch-shift."""
    rng = np.random.default_rng()

    # 1. Additive white noise  (always, low level)
    noise_amp = rng.uniform(0.001, 0.008)
    y = y + noise_amp * rng.standard_normal(len(y)).astype(np.float32)

    # 2. Random time shift  ±0.2 s
    shift = int(rng.uniform(-0.2, 0.2) * sr)
    y = np.roll(y, shift)

    # 3. Pitch shift  ±2 semitones  (50 % chance)
    if rng.random() < 0.5:
        n_steps = rng.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # 4. Time-stretch  ×[0.9, 1.1]  (50 % chance)
    if rng.random() < 0.5:
        rate = rng.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)

    return y.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extractors  —  all return  (T, F) arrays
# ──────────────────────────────────────────────────────────────────────────────

def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """40 MFCCs + delta + delta-delta  →  (T, 120)."""
    mfcc  = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1    = librosa.feature.delta(mfcc)
    d2    = librosa.feature.delta(mfcc, order=2)
    feat  = np.concatenate([mfcc, d1, d2], axis=0)   # (120, T)
    return feat.T                                      # (T, 120)


def extract_melspectrogram(y: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram  →  (T, 128)."""
    mel   = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_dB = librosa.power_to_db(mel, ref=np.max)
    return mel_dB.T                                    # (T, 128)


def extract_energy(y: np.ndarray) -> np.ndarray:
    """RMS energy  →  (T, 1)."""
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    return rms.T                                       # (T, 1)


def extract_combined(y: np.ndarray) -> np.ndarray:
    """
    MFCC+delta+delta2  ||  log-mel  ||  RMS  →  (T, 249).
    Best single feature set for the bonus / high-accuracy run.
    """
    mfcc_feat = extract_mfcc(y)          # (T, 120)
    mel_feat  = extract_melspectrogram(y) # (T, 128)
    rms_feat  = extract_energy(y)         # (T, 1)
    # All extractors use the same hop_length so T is identical
    T = min(mfcc_feat.shape[0], mel_feat.shape[0], rms_feat.shape[0])
    return np.concatenate([mfcc_feat[:T], mel_feat[:T], rms_feat[:T]], axis=1)  # (T,249)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder  (with optional augmentation)
# ──────────────────────────────────────────────────────────────────────────────

def _pad_frames(feat: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad or truncate time axis to max_frames."""
    T, F = feat.shape
    if T >= max_frames:
        return feat[:max_frames]
    pad = np.zeros((max_frames - T, F), dtype=np.float32)
    return np.concatenate([feat, pad], axis=0)


def build_dataset(meta_df: pd.DataFrame, extractor_fn, cache_path: str,
                  augment: bool = False) -> tuple:
    """
    Extract features for every clip in meta_df.
    Results are cached as .npz after the first run.

    augment=True  adds one extra augmented copy of every training clip.
    NOTE: augmentation is done in audio space before feature extraction,
          so cached files never contain augmented data — augmentation is
          handled at Dataset level via SoundDataset(augment=True).
    """
    cache = Path(cache_path)
    if cache.exists():
        print(f"  [cache] Loading {cache_path}")
        d = np.load(cache)
        return d["X"], d["y"], d["folds"]

    print(f"  Extracting features → {cache_path}  (runs once, then cached)")
    all_feats, all_labels, all_folds = [], [], []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), ncols=80):
        path = AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
        try:
            y    = load_audio(str(path))
            y    = pad_or_truncate(y)                 # fixed 4-second clip
            feat = extractor_fn(y)                    # (T_raw, F)
            feat = _pad_frames(feat, MAX_FRAMES)      # (MAX_FRAMES, F)
            all_feats.append(feat)
            all_labels.append(int(row["classID"]))
            all_folds.append(int(row["fold"]))
        except Exception as e:
            print(f"\n  [skip] {path.name}: {e}")

    X     = np.stack(all_feats).astype(np.float32)   # (N, MAX_FRAMES, F)
    y_arr = np.array(all_labels, dtype=np.int64)
    folds = np.array(all_folds,  dtype=np.int64)

    np.savez_compressed(cache, X=X, y=y_arr, folds=folds)
    print(f"  Saved  X={X.shape}")
    return X, y_arr, folds


# ──────────────────────────────────────────────────────────────────────────────
# Split & normalise
# ──────────────────────────────────────────────────────────────────────────────

def split_and_normalise(X, y, folds):
    tr = (folds >= 1) & (folds <= 6)
    vl = (folds == 7) | (folds == 8)
    te = (folds == 9) | (folds == 10)

    X_tr, y_tr = X[tr], y[tr]
    X_vl, y_vl = X[vl], y[vl]
    X_te, y_te = X[te], y[te]

    # Normalise per feature dimension using training statistics
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    sig = X_tr.std( axis=(0, 1), keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sig
    X_vl = (X_vl - mu) / sig
    X_te = (X_te - mu) / sig

    print(f"  Train {X_tr.shape} | Val {X_vl.shape} | Test {X_te.shape}")
    return (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class SoundDataset(Dataset):
    """
    Optionally applies on-the-fly SpecAugment-style masking
    (frequency + time masking) during training.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].clone()         # (T, F)

        if self.augment:
            T, F = x.shape
            # ── Frequency masking  (mask up to 15 % of bins) ──
            f_mask = max(1, int(F * 0.15))
            f0     = torch.randint(0, F - f_mask + 1, (1,)).item()
            x[:, f0: f0 + f_mask] = 0.0

            # ── Time masking  (mask up to 10 % of frames) ──
            t_mask = max(1, int(T * 0.10))
            t0     = torch.randint(0, T - t_mask + 1, (1,)).item()
            x[t0: t0 + t_mask, :] = 0.0

        return x, self.y[i]


def make_loaders(train, val, test, augment_train: bool = True):
    tr_ds = SoundDataset(*train, augment=augment_train)
    vl_ds = SoundDataset(*val,   augment=False)
    te_ds = SoundDataset(*test,  augment=False)
    return (
        DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True),
        DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True),
    )