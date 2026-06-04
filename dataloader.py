import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from configurations import (
    SR, N_MFCC, N_MELS, HOP_LENGTH, N_FFT, AUDIO_DIR, BATCH_SIZE
)

def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y

def extract_mfcc(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc.T

def extract_melspectrogram(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_dB = librosa.power_to_db(mel, ref=np.max)
    return mel_dB.T

def extract_energy(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    return rms.T

def build_dataset(meta_df: pd.DataFrame, extractor_fn, cache_path: str) -> tuple:
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
            feat = extractor_fn(y)
            all_feats.append(feat)
            all_labels.append(int(row["classID"]))
            all_folds.append(int(row["fold"]))
        except Exception as e:
            print(f"\n  [skip] {path.name}: {e}")

    T_min = min(f.shape[0] for f in all_feats)
    print(f"  Shortest clip = {T_min} frames  →  all clips truncated to {T_min}")

    X     = np.stack([f[:T_min] for f in all_feats]).astype(np.float32)
    y_arr = np.array(all_labels, dtype=np.int64)
    folds = np.array(all_folds,  dtype=np.int64)

    np.savez_compressed(cache, X=X, y=y_arr, folds=folds)
    print(f"  Saved  X={X.shape}")
    return X, y_arr, folds

def split_and_normalise(X, y, folds):
    tr = (folds >= 1) & (folds <= 6)
    vl = (folds == 7) | (folds == 8)
    te = (folds == 9) | (folds == 10)

    X_tr, y_tr = X[tr], y[tr]
    X_vl, y_vl = X[vl], y[vl]
    X_te, y_te = X[te], y[te]

    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    sig = X_tr.std( axis=(0, 1), keepdims=True) + 1e-8

    X_tr = (X_tr - mu) / sig
    X_vl = (X_vl - mu) / sig
    X_te = (X_te - mu) / sig

    print(f"  Train {X_tr.shape} | Val {X_vl.shape} | Test {X_te.shape}")
    return (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te)

class SoundDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def make_loaders(train, val, test):
    return (
        DataLoader(SoundDataset(*train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        DataLoader(SoundDataset(*val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(SoundDataset(*test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )