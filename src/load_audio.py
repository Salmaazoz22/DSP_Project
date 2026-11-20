import os
import io
import json
import numpy as np
import pandas as pd
import soundfile as sf
from collections import Counter
import zipfile
import random
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# import sounddevice as sd


ROOT = Path(__file__).resolve().parent.parent  # Speech-Hum-Removal/
DATA_DIR = ROOT / "data" / "wav"
SAVE_DIR = ROOT / "outputs"
SAVE_DIR.mkdir(exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
TARGET_SR = 16000


def analyze_dataset(data_dir, save_dir=None, recursive=False):

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {data_dir}")

    pattern = data_dir.rglob("*") if recursive else data_dir.iterdir()
    files = sorted(p for p in pattern if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise FileNotFoundError(f"No audio files {sorted(AUDIO_EXTS)} in: {data_dir}")

    file_info = []
    for p in files:
        audio, sr = sf.read(p)  # reads the header only, much faster than sf.read
        file_info.append(
            {
                "filename": p.name,
                "relative_path": p.relative_to(data_dir).as_posix(),  # portable
                "sampling_rate": int(sr),
                "num_samples": int(audio.shape[0]),
                "channels": int(audio.shape[1]) if audio.ndim > 1 else 1,
                "duration_seconds": audio.shape[0] / sr,
            }
        )

    df = pd.DataFrame(file_info)

    sr_counts = Counter(df["sampling_rate"])
    len_counts = Counter(df["num_samples"])

    summary = {
        "total_files": len(df),
        "mean_duration_seconds": float(df["duration_seconds"].mean()),
        "std_duration_seconds": float(df["duration_seconds"].std()),
        "min_duration_seconds": float(df["duration_seconds"].min()),
        "max_duration_seconds": float(df["duration_seconds"].max()),
        "mean_num_samples": float(df["num_samples"].mean()),
        "min_num_samples": int(df["num_samples"].min()),
        "max_num_samples": int(df["num_samples"].max()),
        "channels_counts": {str(k): int(v) for k, v in Counter(df["channels"]).items()},
        "sampling_rate_counts": {str(k): int(v) for k, v in sr_counts.items()},
        "num_samples_counts": {str(k): int(v) for k, v in len_counts.items()},
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(save_dir / "DataFrame.csv", index=False)
        with open(save_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "files": file_info},
                f,
                indent=4,
                ensure_ascii=False,
            )

        summary_df = pd.DataFrame(
            [
                (k, json.dumps(v) if isinstance(v, dict) else v)
                for k, v in summary.items()
            ],
            columns=["Metric", "Value"],
        )
        with pd.ExcelWriter(save_dir / "dataset_statistics.xlsx") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            df.to_excel(writer, sheet_name="Files", index=False)

    return df, summary


def load_audio(path, target_sr=TARGET_SR):
    x, sr = sf.read(path, dtype="float32")
    # >>>jt for checking , all data is mono <<<
    if x.ndim > 1:
        x = x.mean(axis=1)

    if sr != target_sr:  # resample
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype("float32")

    x = x - x.mean()  # remove DC offset
    return x


def peak_normalize(x, peak=0.95):
    return x * (peak / (np.max(np.abs(x)) + 1e-9))


# def fix_length(x, length=TARGET_LEN):
#     if len(x) >= length:
#         return x[:length]
#     return np.pad(x, (0, length - len(x)))   # zero-pad


def protect_clipping(x, limit=0.99):
    peak = np.max(np.abs(x))
    if peak > limit:
        x = x * (limit / peak)
    return x


def save_wav(path, audio, sr):
    path = Path(path)
    if path.suffix == "":
        raise ValueError(f"Give a file name with extension, not a folder: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def wav_bytes(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# data = wav_bytes(noisy, sr)
