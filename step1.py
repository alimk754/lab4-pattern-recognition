"""
Step 1: Dataset Preparation & Exploration
==========================================
UrbanSound8K — Load audio, plot waveforms/spectrograms, listen to samples.

Requirements:
    pip install librosa soundfile pandas matplotlib numpy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from IPython.display import Audio, display   # works in Jupyter; prints path in script mode

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "./UrbanSound8K"          # <-- change to your local path
META_CSV     = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR    = os.path.join(DATASET_ROOT, "audio")
SR           = 22050                     # consistent sampling rate (Hz)
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}


# ── 1.  Load metadata ─────────────────────────────────────────────────────────
def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load the UrbanSound8K metadata CSV and return a DataFrame."""
    df = pd.read_csv(csv_path)
    df["class_name"] = df["classID"].map(CLASS_NAMES)
    return df


# ── 2.  Load a single audio file ─────────────────────────────────────────────
def load_audio(row: pd.Series, audio_dir: str, sr: int = SR):
    """
    Load one audio clip at a fixed sample-rate.

    Returns
    -------
    y  : np.ndarray  — raw waveform (1-D)
    sr : int         — sample rate used
    """
    path = os.path.join(audio_dir, f"fold{row['fold']}", row["slice_file_name"])
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


# ── 3.  Plot waveform ─────────────────────────────────────────────────────────
def plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform", ax=None):
    """Plot the time-domain waveform of an audio clip."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#2196F3")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return ax


# ── 4.  Plot Mel-spectrogram ──────────────────────────────────────────────────
def plot_mel_spectrogram(y: np.ndarray, sr: int, title: str = "Mel Spectrogram", ax=None):
    """Convert waveform to Mel spectrogram and visualise it."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time",
                                   y_axis="mel", fmax=8000, ax=ax, cmap="magma")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return ax


# ── 5.  Dataset exploration: one clip per class ───────────────────────────────
def explore_dataset(df: pd.DataFrame, audio_dir: str, save_fig: bool = True):
    """
    For each of the 10 classes pick one example and plot:
      - waveform
      - Mel spectrogram
    Saves a combined figure to 'step1_exploration.png'.
    """
    fig, axes = plt.subplots(10, 2, figsize=(16, 40))
    fig.suptitle("UrbanSound8K — One Sample per Class\n(waveform  |  Mel spectrogram)",
                 fontsize=16, fontweight="bold", y=1.002)

    for class_id in range(10):
        sample = df[df["classID"] == class_id].iloc[0]
        y, sr = load_audio(sample, audio_dir)

        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[class_id, 0], color="#2196F3")
        axes[class_id, 0].set_title(f"[{class_id}] {CLASS_NAMES[class_id]}  — Waveform",
                                    fontsize=11, fontweight="bold")
        axes[class_id, 0].set_xlabel("Time (s)")
        axes[class_id, 0].set_ylabel("Amplitude")

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time",
                                       y_axis="mel", fmax=8000,
                                       ax=axes[class_id, 1], cmap="magma")
        axes[class_id, 1].set_title(f"[{class_id}] {CLASS_NAMES[class_id]}  — Mel Spectrogram",
                                    fontsize=11, fontweight="bold")
        plt.colorbar(img, ax=axes[class_id, 1], format="%+2.0f dB")

    plt.tight_layout()
    if save_fig:
        plt.savefig("step1_exploration.png", dpi=120, bbox_inches="tight")
        print("Saved → step1_exploration.png")
    plt.show()


# ── 6.  Audio playback helper ─────────────────────────────────────────────────
def listen_to_sample(df: pd.DataFrame, audio_dir: str, class_id: int):
    """Display an audio widget for one clip (Jupyter) or print the file path."""
    sample = df[df["classID"] == class_id].iloc[0]
    path = os.path.join(audio_dir, f"fold{sample['fold']}", sample["slice_file_name"])
    print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {path}")
    try:
        display(Audio(path))
    except Exception:
        print("  ↳ Run in Jupyter to hear the audio.")


# ── 7.  Dataset statistics ────────────────────────────────────────────────────
def print_dataset_stats(df: pd.DataFrame):
    print("=" * 50)
    print(f"Total clips   : {len(df)}")
    print(f"Classes       : {df['classID'].nunique()}")
    print(f"Folds         : {sorted(df['fold'].unique())}")
    print("\nSamples per class:")
    print(df["class_name"].value_counts().to_string())
    print("=" * 50)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_metadata(META_CSV)
    print_dataset_stats(df)

    # Plot one sample per class
    explore_dataset(df, AUDIO_DIR)

    # Audio playback — all classes
    for cid in range(10):
        listen_to_sample(df, AUDIO_DIR, cid)