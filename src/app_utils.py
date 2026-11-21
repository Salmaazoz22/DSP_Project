import io

import matplotlib

matplotlib.use("Agg")  # before src.analysis imports pyplot: never open a GUI window

import numpy as np
import soundfile as sf
from matplotlib.figure import Figure
from scipy.signal import spectrogram

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


def _quietest_window(clean, fs, duration=0.1, step_s=0.01):
    """(start, length) in samples of the `duration` window where `clean` has the lowest RMS.

    Starts are searched every `step_s` seconds; a signal shorter than `duration` uses all of it.
    """
    clean = np.asarray(clean, dtype=np.float64)
    n = len(clean)
    win = max(1, int(round(duration * fs)))
    if n <= win:
        return 0, n
    step = max(1, int(round(step_s * fs)))
    starts = np.arange(0, n - win + 1, step)
    csum = np.concatenate([[0.0], np.cumsum(clean**2)])
    energy = csum[starts + win] - csum[starts]  # same length for every window -> lowest energy = lowest RMS
    return int(starts[np.argmin(energy)]), win


def hum_zoom_figure(clean, noisy, fs, f0, duration=0.1):
    """Zoom on the quietest `duration` s of the clean signal: (a) clean vs noisy, (b) hum only (noisy - clean)."""
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    start, length = _quietest_window(clean, fs, duration)
    seg = slice(start, start + length)
    t_ms = np.arange(length) / fs * 1000
    win_ms = length / fs * 1000
    cycles = round(f0 * length / fs, 2)  # = f0 * duration when the signal is long enough

    fig = Figure(figsize=(9, 5))
    ax_a, ax_b = fig.subplots(2, 1, sharex=True)
    # same styling idea as waveform_figure: first trace wide + translucent underneath
    ax_a.plot(t_ms, clean[seg], label="clean", linewidth=1.2, alpha=0.45, zorder=1)
    ax_a.plot(t_ms, noisy[seg], label="noisy", linewidth=0.6, alpha=1.0, zorder=2)
    ax_a.set_title(f"(a) Quietest {win_ms:g} ms of the clean signal (window starts at {start / fs:.2f} s)")
    ax_a.set_ylabel("Amplitude")
    ax_a.grid(True, alpha=0.4)
    ax_a.legend(loc="upper right")

    ax_b.plot(t_ms, noisy[seg] - clean[seg], color="tab:red", linewidth=1.0, label="noisy − clean")
    ax_b.set_title(f"(b) Hum only: {f0:g} Hz -> {cycles:g} cycles in {win_ms:g} ms")
    ax_b.set_xlabel("Time from window start (ms)")
    ax_b.set_ylabel("Amplitude")
    ax_b.grid(True, alpha=0.4)
    ax_b.legend(loc="upper right")
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


def _spectrogram_nperseg(fs, n):
    nperseg = 1 << int(np.ceil(np.log2(fs / 4)))  # <= ~4 Hz bins: 50/100/150 Hz stay separate
    if n < nperseg:  # short signal: largest power of two that fits, at least 256
        nperseg = max(256, 1 << int(np.floor(np.log2(max(n, 1)))))
    return nperseg


def spectrogram_figure(signals, fs, fmax=300):
    """Side-by-side spectrograms (dB) with one shared colour scale. signals: {label: 1-D array}."""
    fmax = min(fmax, fs / 2)
    panels = []
    for label, x in signals.items():
        x = np.asarray(x, dtype=np.float64)
        nperseg = _spectrogram_nperseg(fs, len(x))
        noverlap = (3 * nperseg) // 4
        f, t, sxx = spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        band = f <= fmax
        f, db = f[band], 10 * np.log10(sxx[band] + 1e-12)
        # explicit cell edges: works even with a single time frame (short signals)
        hop, df = (nperseg - noverlap) / fs, f[1] - f[0]
        t_edges = np.concatenate([t - hop / 2, [t[-1] + hop / 2]])
        f_edges = np.concatenate([f - df / 2, [f[-1] + df / 2]])
        panels.append((label, t_edges, f_edges, db))

    vmax = max(db.max() for *_, db in panels)
    vmin = vmax - 80  # same colour scale on every panel

    fig = Figure(figsize=(4.5 * len(panels) + 1, 4), layout="constrained")
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for ax, (label, t_edges, f_edges, db) in zip(axes, panels):
        mesh = ax.pcolormesh(t_edges, f_edges, db, shading="flat", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, fmax)
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=list(axes), label="Power (dB)")
    return fig
