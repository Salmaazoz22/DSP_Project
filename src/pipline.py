import numpy as np
import matplotlib.pyplot as plt
from openai import audio
import random
import pandas as pd
from pathlib import Path
import os
import sounddevice as sd

from analysis import (
    make_hum,
    add_hum,
    fft_magnitude_db,
    save_spectrum_plot,
    detect_mains,
    plot_detection,
)
from load_audio import load_audio, peak_normalize, protect_clipping, save_wav

ROOT = Path(__file__).resolve().parent.parent  # Speech-Hum-Removal/

SAVE_DIR = ROOT / "outputs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = SAVE_DIR / "plots"
AUDIO_DIR = SAVE_DIR / "audio"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
sr = 16000

df = pd.read_excel(SAVE_DIR / "DataFrame.xlsx")
files = list(df["filePath"])
random.seed(20000)
chosen = random.sample(files, k=min(2, len(files)))

for path in chosen:
    stem = Path(path).stem  # >>>"arctic_a0229" <<<

    x_normalized = peak_normalize(load_audio(path))
    hum_noise = make_hum(len(x_normalized), sr)
    noisy = protect_clipping(add_hum(x_normalized, sr, f0=50, snr_db=10), limit=0.99)

    f_a, db_a, mag_a = fft_magnitude_db(x_normalized, sr, n_fft=512, window=True)
    f_h, db_h, mag_h = fft_magnitude_db(hum_noise, sr, n_fft=512, window=True)
    f_n, db_n, mag_n = fft_magnitude_db(noisy, sr, n_fft=512, window=True)
    res = detect_mains(noisy, sr)
    print(res["message"])
    print(
        f"measured {res['measured_freq']:.2f} Hz | peak {res['peak_db']:.1f} dB | "
        f"prominence {res['prominence_db']:.1f} dB | resolution {res['resolution_hz']:.2f} Hz"
    )
    plot_detection(
        noisy, sr, res, ROOT / "outputs" / "plots" / f"detection_example {stem}.png"
    )

    save_spectrum_plot(f_a, db_a, PLOT_DIR / f"{stem}_audio_spectrum.png")
    save_spectrum_plot(f_h, db_h, PLOT_DIR / f"{stem}_hum_spectrum.png")
    save_spectrum_plot(f_n, db_n, PLOT_DIR / f"{stem}_noisy_spectrum.png")

    save_wav(AUDIO_DIR / f"{stem}_clean.wav", x_normalized, sr)
    save_wav(AUDIO_DIR / f"{stem}_hum.wav", hum_noise, sr)
    save_wav(AUDIO_DIR / f"{stem}_noisy.wav", noisy, sr)

    print("Done:", stem)
