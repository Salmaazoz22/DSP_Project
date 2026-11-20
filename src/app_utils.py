import io

import matplotlib

matplotlib.use("Agg")  # before src.analysis imports pyplot: never open a GUI window

import numpy as np
import soundfile as sf
from matplotlib.figure import Figure

from src.analysis import detect_mains, fft_magnitude_db, plot_detection
from src.load_audio import load_audio, protect_clipping, wav_bytes

_SF_ERRORS = (RuntimeError, sf.SoundFileError)


def read_upload(data):
    """Decode WAV bytes at their native rate. Returns (x float64 1-D, fs int, info dict)."""
    if not data:
        raise ValueError("Uploaded file is empty")
    try:
        meta = sf.info(io.BytesIO(data))  # fresh buffer for each read
        fs = int(meta.samplerate)
        x = load_audio(io.BytesIO(data), target_sr=fs)  # native fs -> no resampling
    except _SF_ERRORS as e:
        raise ValueError(f"Could not read audio file: {e}") from e

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Audio file contains no samples")

    info = {
        "fs": fs,
        "channels": int(meta.channels),
        "frames": int(meta.frames),
        "duration_s": float(meta.frames / fs),
        "subtype": str(meta.subtype),
        "peak": float(np.max(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x**2))),
    }
    return x, fs, info


def to_wav(x, fs):
    """16-bit WAV bytes for playback/download (scaled down only if it would clip)."""
    return wav_bytes(protect_clipping(np.asarray(x, dtype=np.float64)), int(fs))


def safe_detect(x, fs):
    """detect_mains result dict, or None if the signal is too short or analysis fails."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < fs // 4:  # need <= 4 Hz resolution for the ±2 Hz windows
        return None
    try:
        return detect_mains(x, fs)
    except ValueError:
        return None


def detection_png(x, fs, res):
    """Render Student 1's plot_detection into PNG bytes (no plt.show)."""
    buf = io.BytesIO()
    plot_detection(x, fs, res, out_path=buf)  # BytesIO is truthy -> save + close branch
    return buf.getvalue()


def waveform_figure(signals, fs):
    """Overlay time-domain traces. signals: {label: 1-D array}."""
    fig = Figure(figsize=(9, 3))
    ax = fig.subplots()
    for i, (label, x) in enumerate(signals.items()):
        x = np.asarray(x, dtype=np.float64)
        # first trace wide + translucent underneath, later traces thin on top -> both visible where equal
        style = dict(linewidth=1.2, alpha=0.45, zorder=1) if i == 0 else dict(linewidth=0.6, alpha=1.0, zorder=2 + i)
        ax.plot(np.arange(len(x)) / fs, x, label=label, **style)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def spectrum_figure(signals, fs, fmax=200):
    """Overlay magnitude spectra (dB) from fft_magnitude_db, 0..fmax Hz."""
    fmax = min(fmax, fs / 2)
    fig = Figure(figsize=(9, 3.5))
    ax = fig.subplots()
    top, bottom = -np.inf, np.inf
    for i, (label, x) in enumerate(signals.items()):
        freqs, db, _ = fft_magnitude_db(np.asarray(x, dtype=np.float64), fs)
        band = freqs <= fmax
        # first trace wide + translucent underneath, later traces thin dashed on top
        if i == 0:
            style = dict(linewidth=2.2, alpha=0.45, linestyle="-", zorder=1)
        else:
            style = dict(linewidth=0.9, alpha=1.0, linestyle="--", zorder=2 + i)
        ax.plot(freqs[band], db[band], label=label, **style)
        top, bottom = max(top, db[band].max()), min(bottom, db[band].min())
    ax.set_xlim(0, fmax)
    ax.set_ylim(max(bottom, top - 120) - 5, top + 5)  # keep -240 dB floor of silence off-scale
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", title="overlapping lines = unchanged", title_fontsize="small")
    fig.tight_layout()
    return fig
