import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks


def make_hum(
    n_samples, sr, f0=50, harmonics=(1,), amps=(1.0,)
):
    t = np.arange(n_samples) / sr
    hum = np.zeros(n_samples, dtype="float32")
    for h, a in zip(harmonics, amps):
        hum += a * np.sin(2 * np.pi * f0 * h * t)
    return hum / np.max(np.abs(hum))


def add_hum(clean, sr, f0=50, snr_db=10):
    hum = make_hum(len(clean), sr, f0)
    # scale hum so the mix has the requested SNR
    p_clean = np.mean(clean**2)
    p_hum = np.mean(hum**2)
    hum *= np.sqrt(p_clean / (p_hum * 10 ** (snr_db / 10)))
    return clean + hum


def fft_magnitude_db(x, sr, n_fft=None, window=True):
    n = len(x) if n_fft is None else max(n_fft, len(x))  # >> to be smoother <<
    w = np.hanning(len(x)) if window else np.ones(len(x))
    X = np.fft.rfft(x * w, n=n)

    freqs = np.fft.rfftfreq(n, 1 / sr)
    mag = np.abs(X) / (np.sum(w) / 2)
    return freqs, 20 * np.log10(mag + 1e-12), mag


def play(audio, sr, volume=0.5, fade_ms=20):

    x = np.asarray(audio, dtype="float32").copy()
    x = x / (np.max(np.abs(x)) + 1e-9) * volume  # safe loudness

    n = int(sr * fade_ms / 1000)  # fade in/out, avoids clicks
    if len(x) > 2 * n:
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import Audio, display

            display(Audio(x, rate=sr))
            return
    except ImportError:
        pass


def save_spectrum_plot(freq, db, out_path, xmax=500):

    plt.figure(figsize=(9, 4))
    plt.plot(freq, db)
    plt.xlim(0, xmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)  # save BEFORE show
    plt.close()


def detect_mains(x, sr, search=(40, 70), tol=2.0, min_prom_db=10.0):
    freqs, db, mag = fft_magnitude_db(x, sr)
    df = sr / len(x)  # frequency resolution
    band = (freqs >= search[0]) & (freqs <= search[1])
    band_median = float(np.median(db[band]))

    def strongest(f_nom):
        idx = np.flatnonzero((freqs >= f_nom - tol) & (freqs <= f_nom + tol))
        k = idx[np.argmax(db[idx])]
        a, b, c = db[k - 1], db[k], db[k + 1]  # parabolic interpolation
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return {
            "freq": float((k + delta) * df),
            "db": float(b),
            "prominence": float(b - band_median),
            "mag": float(mag[k]),
        }

    c50, c60 = strongest(50), strongest(60)
    best_f0, best = (50, c50) if c50["db"] >= c60["db"] else (60, c60)
    detected = best["prominence"] >= min_prom_db

    return {
        "detected": detected,
        "selected_f0": best_f0 if detected else None,  # None -> GUI keeps sidebar value
        "measured_freq": best["freq"],
        "peak_db": best["db"],
        "peak_mag": best["mag"],
        "prominence_db": best["prominence"],
        "band_median_db": band_median,
        "c50": c50,
        "c60": c60,
        "resolution_hz": df,
        "message": (
            (
                f"Hum at {best['freq']:.2f} Hz ({best['prominence']:.1f} dB above band median) "
                f"-> mains {best_f0} Hz"
            )
            if detected
            else "no clear hum"
        ),
    }


def plot_detection(x, sr, res, out_path=None):
    freqs, db, mag = fft_magnitude_db(x, sr)
    m = (freqs >= 40) & (freqs <= 70)
    plt.figure(figsize=(9, 4))
    plt.plot(freqs[m], db[m], label="spectrum")
    plt.axvspan(48, 52, alpha=0.15, label="50 ± 2 Hz")
    plt.axvspan(58, 62, alpha=0.15, label="60 ± 2 Hz")
    plt.axhline(res["band_median_db"], linestyle="--", label="band median")
    for c in (res["c50"], res["c60"]):
        plt.plot(c["freq"], c["db"], "o")
        plt.annotate(
            f"freq: {c['freq']:.2f} Hz\n Peak:{c['db']:.1f} dB\n prominence:({c['prominence']:.1f} dB",
            (c["freq"], c["db"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    plt.title(res["message"])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBFS)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()
