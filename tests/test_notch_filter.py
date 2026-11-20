import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.signal import freqz, iirnotch

from src.notch_filter import apply_notch, design_notch, plot_pole_zero, plot_response

FS_VALUES = [16000, 8000]
F0_VALUES = [50, 60]
CASES = [(fs, f0) for fs in FS_VALUES for f0 in F0_VALUES]


def _response_db(b, a, fs, freqs):
    _, h = freqz(b, a, worN=np.asarray(freqs, dtype=np.float64), fs=fs)
    return 20 * np.log10(np.abs(h) + 1e-300)


def _amplitude(x, f, fs):
    # amplitude of the f Hz component; exact when the window holds whole cycles
    t = np.arange(len(x)) / fs
    return 2 * np.abs(np.sum(x * np.exp(-2j * np.pi * f * t))) / len(x)


# ---------- design_notch ----------


@pytest.mark.parametrize("fs, f0", CASES)
def test_coefficients_are_float64_shape_3(fs, f0):
    b, a = design_notch(f0, fs)
    for c in (b, a):
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.float64
        assert c.shape == (3,)


@pytest.mark.parametrize("fs, f0", CASES)
def test_a0_is_one(fs, f0):
    _, a = design_notch(f0, fs)
    assert a[0] == 1


@pytest.mark.parametrize("fs, f0", CASES)
def test_matches_scipy_iirnotch(fs, f0):
    b, a = design_notch(f0, fs, Q=30)
    b_ref, a_ref = iirnotch(f0, 30, fs)  # reference only
    np.testing.assert_allclose(b, b_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(a, a_ref, rtol=0, atol=1e-10)


@pytest.mark.parametrize("fs, f0", CASES)
def test_poles_inside_unit_circle(fs, f0):
    _, a = design_notch(f0, fs)
    assert np.all(np.abs(np.roots(a)) < 1)


@pytest.mark.parametrize("fs, f0", CASES)
def test_attenuation_at_notch(fs, f0):
    b, a = design_notch(f0, fs)
    assert _response_db(b, a, fs, [f0])[0] < -40


@pytest.mark.parametrize("fs, f0", CASES)
def test_passband_near_0db(fs, f0):
    b, a = design_notch(f0, fs)
    db_100, db_1k = _response_db(b, a, fs, [100, 1000])
    assert abs(db_100) < 0.1
    assert abs(db_1k) < 0.1


# ---------- apply_notch ----------


@pytest.mark.parametrize("fs, f0", CASES)
def test_apply_keeps_length_and_dtype(fs, f0):
    b, a = design_notch(f0, fs)
    x = np.random.default_rng(0).standard_normal(1234).astype(np.float32)
    y = apply_notch(x, b, a)
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.dtype == np.float64
    assert len(y) == len(x)


def test_apply_empty_input_returns_empty_float64():
    y = apply_notch(np.array([]), *design_notch(50, 16000))
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float64
    assert y.shape == (0,)


@pytest.mark.parametrize("fs, f0", CASES)
def test_end_to_end_removes_hum_keeps_tone(fs, f0):
    t = np.arange(2 * fs) / fs
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    hum = 0.2 * np.sin(2 * np.pi * f0 * t)
    x = tone + hum
    y = apply_notch(x, *design_notch(f0, fs))

    last = slice(fs, 2 * fs)  # last 1 s, past the start-up transient
    hum_drop_db = 20 * np.log10(_amplitude(x[last], f0, fs) / _amplitude(y[last], f0, fs))
    tone_change_db = 20 * np.log10(_amplitude(y[last], 440, fs) / _amplitude(x[last], 440, fs))

    assert hum_drop_db > 30
    assert abs(tone_change_db) < 0.1


# ---------- plot_response ----------


@pytest.mark.parametrize("fs, f0", CASES)
def test_plot_response_returns_figure_without_pyplot(fs, f0):
    before = plt.get_fignums()
    fig = plot_response(*design_notch(f0, fs), fs)
    assert isinstance(fig, Figure)
    assert plt.get_fignums() == before


# ---------- plot_pole_zero ----------


@pytest.mark.parametrize("fs, f0", CASES)
def test_plot_pole_zero_returns_figure_without_pyplot(fs, f0):
    before = plt.get_fignums()
    fig = plot_pole_zero(*design_notch(f0, fs))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # full view + zoom
    assert plt.get_fignums() == before


@pytest.mark.parametrize("fs, f0", CASES)
def test_plot_pole_zero_plots_zeros_on_circle_poles_inside(fs, f0):
    b, a = design_notch(f0, fs)
    ax_full = plot_pole_zero(b, a).axes[0]
    lines = {ln.get_label(): ln for ln in ax_full.lines}
    assert {"zeros", "poles"} <= set(lines)
    z = lines["zeros"].get_xdata() + 1j * lines["zeros"].get_ydata()
    p = lines["poles"].get_xdata() + 1j * lines["poles"].get_ydata()
    np.testing.assert_allclose(np.abs(z), 1.0, atol=1e-9)
    assert np.all(np.abs(p) < 1)


# ---------- invalid inputs ----------

NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "f0, fs, Q",
    [
        (50, 0, 30),  # fs <= 0
        (50, -16000, 30),
        (0, 16000, 30),  # f0 <= 0
        (-50, 16000, 30),
        (8000, 16000, 30),  # f0 == fs/2
        (9000, 16000, 30),  # f0 > fs/2
        (50, 16000, 0),  # Q <= 0
        (50, 16000, -1),
        (True, 16000, 30),  # bool params
        (50, True, 30),
        (50, 16000, True),
        (NAN, 16000, 30),  # NaN params
        (50, NAN, 30),
        (50, 16000, NAN),
        (INF, 16000, 30),  # inf params
        (50, INF, 30),
        (50, 16000, INF),
    ],
)
def test_design_notch_invalid_params(f0, fs, Q):
    with pytest.raises(ValueError):
        design_notch(f0, fs, Q)


_B, _A = design_notch(50, 16000)

INVALID_COEFFS = [
    (_B[:2], _A),  # wrong coefficient shape
    (_B, np.append(_A, 0.0)),
    (_B.reshape(1, 3), _A),
    (_B, _A.reshape(3, 1)),
    (_B, [0.0, 1.0, 0.5]),  # a[0] == 0
    (_B, [1.0, -2.5, 1.5]),  # unstable a (poles at 1 and 1.5)
    ([np.nan, 0.0, 1.0], _A),  # NaN coefficient
]


@pytest.mark.parametrize("b, a", INVALID_COEFFS)
def test_apply_notch_invalid_coefficients(b, a):
    with pytest.raises(ValueError):
        apply_notch(np.zeros(16), b, a)


@pytest.mark.parametrize("b, a", INVALID_COEFFS)
def test_plot_pole_zero_invalid_coefficients(b, a):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_pole_zero(b, a)
    assert plt.get_fignums() == before


@pytest.mark.parametrize(
    "x",
    [
        np.zeros((16, 2)),  # 2-D
        np.zeros(16, dtype=complex),  # complex
        np.array([0.0, np.nan, 0.0]),  # NaN
    ],
)
def test_apply_notch_invalid_signal(x):
    with pytest.raises(ValueError):
        apply_notch(x, _B, _A)
