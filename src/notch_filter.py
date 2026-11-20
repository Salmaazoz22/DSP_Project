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


def _plot_roots(ax, zeros, poles, size):
    ax.plot(zeros.real, zeros.imag, "o", markerfacecolor="none", markersize=size, label="zeros")
    ax.plot(poles.real, poles.imag, "x", markersize=size, markeredgewidth=1.8, label="poles")


def plot_pole_zero(b, a):
    """Pole-zero plot: full z-plane view + zoom on the upper-half-plane zero/pole pair. Returns a Figure."""
    b, a = _check_coefficients(b, a)
    zeros, poles = np.roots(b), np.roots(a)

    fig = Figure(figsize=(11, 5))  # not registered with pyplot -> no plt.show / leaks
    ax_full, ax_zoom = fig.subplots(1, 2)
    fig.suptitle("Pole-zero plot (zeros on the unit circle, poles just inside)")

    # full view
    theta = np.linspace(0, 2 * np.pi, 721)
    ax_full.plot(np.cos(theta), np.sin(theta), "--", linewidth=0.8, color="gray", label="unit circle")
    _plot_roots(ax_full, zeros, poles, size=9)
    ax_full.axhline(0, color="black", linewidth=0.6)
    ax_full.axvline(0, color="black", linewidth=0.6)
    ax_full.set_xlim(-1.2, 1.2)
    ax_full.set_ylim(-1.2, 1.2)
    ax_full.set_aspect("equal")
    ax_full.set_xlabel("Real")
    ax_full.set_ylabel("Imaginary")
    ax_full.grid(True, alpha=0.4)
    ax_full.legend(loc="lower left", fontsize="small")

    # zoom on the upper-half-plane pair: at 50 Hz / 16 kHz they sit ~3e-4 apart near (1, 0)
    z0 = zeros[np.argmax(zeros.imag)]
    p0 = poles[np.argmax(poles.imag)]
    center = (z0 + p0) / 2
    half = max(3 * abs(z0 - p0), 1e-4)
    ang = np.angle(center)
    arc = np.linspace(ang - 2 * half, ang + 2 * half, 401)  # radius ~1 -> arc length ~ angle span
    ax_zoom.plot(np.cos(arc), np.sin(arc), "--", linewidth=0.8, color="gray", label="unit circle")
    _plot_roots(ax_zoom, zeros, poles, size=12)
    ax_zoom.set_xlim(center.real - half, center.real + half)
    ax_zoom.set_ylim(center.imag - half, center.imag + half)
    ax_zoom.set_aspect("equal")
    ax_zoom.ticklabel_format(useOffset=False)
    ax_zoom.tick_params(axis="x", labelrotation=30)
    ax_zoom.set_xlabel("Real")
    ax_zoom.set_ylabel("Imaginary")
    ax_zoom.set_title("Zoom: upper-half-plane zero and pole", fontsize="medium")
    ax_zoom.grid(True, alpha=0.4)
    ax_zoom.legend(loc="lower left", fontsize="small")
    ax_zoom.text(
        0.03,
        0.97,
        f"|zero| = {abs(z0):.4f}, angle = {np.angle(z0):.5f} rad\n"
        f"|pole| = {abs(p0):.4f}, angle = {np.angle(p0):.5f} rad\n"
        f"zero-pole distance = {abs(z0 - p0):.2e}",
        transform=ax_zoom.transAxes,
        va="top",
        fontsize="small",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    return fig
