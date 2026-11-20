import numbers

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import freqz, lfilter


def _check_positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _check_coefficients(b, a):
    coeffs = []
    for name, c in (("b", b), ("a", a)):
        c = np.asarray(c)
        if np.iscomplexobj(c) or not np.issubdtype(c.dtype, np.number):
            raise ValueError(f"{name} must contain real numbers")
        c = c.astype(np.float64)
        if c.shape != (3,):
            raise ValueError(f"{name} must have exactly 3 coefficients, got shape {c.shape}")
        if not np.all(np.isfinite(c)):
            raise ValueError(f"{name} contains NaN or infinite values")
        coeffs.append(c)
    b, a = coeffs
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    if np.max(np.abs(np.roots(a))) >= 1:  # IIR filter must be stable
        raise ValueError("Filter is unstable: poles lie on or outside the unit circle")
    return b, a


def design_notch(f0, fs, Q=30):
    """Second-order IIR notch at f0 Hz. Returns float64 arrays (b, a) of length 3."""
    fs = _check_positive_number(fs, "fs")
    f0 = _check_positive_number(f0, "f0")
    Q = _check_positive_number(Q, "Q")
    if f0 >= fs / 2:
        raise ValueError(f"f0 must be below Nyquist (fs/2 = {fs / 2} Hz), got {f0} Hz")

    w0 = 2 * np.pi * f0 / fs  # notch frequency (rad/sample)
    bw = w0 / Q  # -3 dB bandwidth (rad/sample)
    beta = np.tan(bw / 2)
    g = 1 / (1 + beta)

    b = g * np.array([1.0, -2 * np.cos(w0), 1.0], dtype=np.float64)
    a = np.array([1.0, -2 * g * np.cos(w0), 2 * g - 1], dtype=np.float64)
    return b, a


def apply_notch(x, b, a):
    """Filter 1-D signal x. Returns a float64 array with the same length as x."""
    b, a = _check_coefficients(b, a)

    x = np.asarray(x)
    if np.iscomplexobj(x) or not np.issubdtype(x.dtype, np.number):
        raise ValueError("x must contain real numbers")
    if x.ndim != 1:
        raise ValueError(f"x must be a 1-D signal, got shape {x.shape}")
    x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")

    return lfilter(b, a, x).astype(np.float64)


def plot_response(b, a, fs, n_points=4096):
    """Magnitude (dB) and phase response on a log-frequency axis. Returns a Figure."""
    b, a = _check_coefficients(b, a)
    fs = _check_positive_number(fs, "fs")

    # log-spaced grid: dense enough at low frequency to resolve a narrow 50/60 Hz notch
    freqs = np.geomspace(1.0, fs / 2, n_points)
    # add the exact zero (notch) frequencies so the plotted dip shows its true depth
    zero_freqs = np.abs(np.angle(np.roots(b))) * fs / (2 * np.pi)
    zero_freqs = zero_freqs[(zero_freqs >= 1.0) & (zero_freqs <= fs / 2)]
    freqs = np.unique(np.concatenate([freqs, zero_freqs]))
    _, h = freqz(b, a, worN=freqs, fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)

    fig = Figure(figsize=(9, 6))  # not registered with pyplot -> no plt.show / leaks
    ax_mag, ax_phase = fig.subplots(2, 1, sharex=True)

    ax_mag.semilogx(freqs, mag_db)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_ylim(max(mag_db.min() - 5, -100), 5)
    ax_mag.set_title("Notch filter frequency response")
    ax_mag.grid(True, which="both", alpha=0.4)

    ax_phase.semilogx(freqs, np.degrees(np.angle(h)))
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    return fig
