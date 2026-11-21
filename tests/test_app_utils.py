import io

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")  # app_utils needs soundfile (via Student 1's loader)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.app_utils import (
    _quietest_window,
    _spectrogram_nperseg,
    detection_png,
    hum_zoom_figure,
    read_upload,
    safe_detect,
    spectrogram_figure,
    spectrum_figure,
    to_wav,
    waveform_figure,
)


def _wav(x, fs, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, x, fs, format="WAV", subtype=subtype)
    return buf.getvalue()


def _tone(f, fs, seconds, amp=0.5):
    t = np.arange(int(fs * seconds)) / fs
    return amp * np.sin(2 * np.pi * f * t)


def _speech_like_with_hum(fs, hum_f0, seconds=2.0):
    rng = np.random.default_rng(0)
    clean = _tone(440, fs, seconds, 0.3) + 0.02 * rng.standard_normal(int(fs * seconds))
    t = np.arange(len(clean)) / fs
    return clean + 0.2 * np.sin(2 * np.pi * hum_f0 * t)  # hum built here, not with make_hum


# ---------- read_upload ----------


@pytest.mark.parametrize("fs", [16000, 8000])
def test_read_upload_keeps_native_fs_mono(fs):
    x_in = _tone(440, fs, 1.0)
    x, fs_out, info = read_upload(_wav(x_in, fs))
    assert fs_out == fs and type(fs_out) is int
    assert isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 1
    assert len(x) == len(x_in)  # no resampling
    assert info["fs"] == fs
    assert info["channels"] == 1
    assert info["frames"] == len(x_in)
    assert info["duration_s"] == pytest.approx(1.0)
    assert info["subtype"] == "PCM_16"
    assert info["peak"] == pytest.approx(np.max(np.abs(x)))
    assert info["rms"] == pytest.approx(np.sqrt(np.mean(x**2)))


def test_read_upload_stereo_becomes_mono():
    left = _tone(440, 8000, 0.5)
    stereo = np.stack([left, 0.5 * left], axis=1)
    x, fs, info = read_upload(_wav(stereo, 8000))
    assert x.ndim == 1 and len(x) == len(left)
    assert info["channels"] == 2
    assert fs == 8000


@pytest.mark.parametrize("data", [b"", b"definitely not a wav file"])
def test_read_upload_rejects_bad_bytes(data):
    with pytest.raises(ValueError):
        read_upload(data)


# ---------- to_wav ----------


@pytest.mark.parametrize("fs", [16000, 8000])
def test_to_wav_roundtrip(fs):
    x = _tone(440, fs, 0.5)
    y, fs_back = sf.read(io.BytesIO(to_wav(x, fs)))
    assert fs_back == fs
    assert len(y) == len(x)
    np.testing.assert_allclose(y, x, atol=1 / 32768 + 1e-9)


def test_to_wav_scales_instead_of_clipping():
    x = np.array([0.0, 2.0, -1.0, 0.5])
    y, _ = sf.read(io.BytesIO(to_wav(x, 16000)))
    assert np.max(np.abs(y)) <= 1.0
    np.testing.assert_allclose(y / y[1], x / x[1], atol=1e-3)  # shape kept, not hard-clipped


# ---------- safe_detect / detection_png ----------


def test_safe_detect_too_short_returns_none():
    fs = 16000
    assert safe_detect(np.zeros(fs // 4 - 1), fs) is None
    assert safe_detect(np.zeros(100), fs) is None


@pytest.mark.parametrize("fs, hum_f0", [(16000, 50), (16000, 60), (8000, 50), (8000, 60)])
def test_safe_detect_finds_hum(fs, hum_f0):
    res = safe_detect(_speech_like_with_hum(fs, hum_f0), fs)
    assert res is not None
    assert res["detected"] is True
    assert res["selected_f0"] == hum_f0
    assert res["measured_freq"] == pytest.approx(hum_f0, abs=0.5)


def test_detection_png_returns_png_without_open_figures():
    fs = 16000
    x = _speech_like_with_hum(fs, 50)
    before = plt.get_fignums()
    png = detection_png(x, fs, safe_detect(x, fs))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert plt.get_fignums() == before


# ---------- figures ----------


def test_waveform_figure_returns_figure():
    fs = 8000
    before = plt.get_fignums()
    fig = waveform_figure({"a": _tone(440, fs, 0.5), "b": _tone(220, fs, 0.5)}, fs)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2
    assert plt.get_fignums() == before


def test_spectrum_figure_returns_figure():
    fs = 16000
    before = plt.get_fignums()
    fig = spectrum_figure({"noisy": _speech_like_with_hum(fs, 50), "tone": _tone(440, fs, 2.0)}, fs, fmax=200)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].get_xlim() == (0, 200)
    assert plt.get_fignums() == before


# ---------- spectrogram_figure ----------


def _panels(fig, labels):
    return [ax for ax in fig.axes if ax.get_title() in labels]


@pytest.mark.parametrize("fs, seconds", [(16000, 2.0), (8000, 2.0), (16000, 0.3), (8000, 0.3)])
def test_spectrogram_figure_shared_scale(fs, seconds):
    noisy = _speech_like_with_hum(fs, 50, seconds=seconds)
    filtered = _tone(440, fs, seconds, 0.3)
    before = plt.get_fignums()
    fig = spectrogram_figure({"noisy": noisy, "filtered": filtered}, fs)
    assert isinstance(fig, Figure)
    assert plt.get_fignums() == before

    panels = _panels(fig, {"noisy", "filtered"})
    assert [ax.get_title() for ax in panels] == ["noisy", "filtered"]
    clims = [ax.collections[0].get_clim() for ax in panels]
    assert clims[0] == clims[1]
    assert clims[0][1] - clims[0][0] == pytest.approx(80)
    for ax in panels:
        assert ax.get_ylim() == (0, 300)
    assert any(ax.get_ylabel() == "Power (dB)" for ax in fig.axes)  # shared colorbar


def test_spectrogram_figure_very_short_signal():
    fs = 16000
    fig = spectrogram_figure({"x": _tone(440, fs, 0.05)}, fs)  # 800 samples -> nperseg 512
    assert isinstance(fig, Figure)
    assert len(_panels(fig, {"x"})) == 1


@pytest.mark.parametrize(
    "fs, n, expected, full_resolution",
    [
        (16000, 32000, 4096, True),  # >= fs/4 -> ~3.9 Hz bins
        (8000, 16000, 2048, True),
        (16000, 4800, 4096, True),  # 0.3 s still fits one full segment
        (16000, 3200, 2048, False),  # shorter than nperseg -> power of two that fits
        (16000, 300, 256, False),  # minimum 256
    ],
)
def test_spectrogram_nperseg(fs, n, expected, full_resolution):
    assert _spectrogram_nperseg(fs, n) == expected
    if full_resolution:
        assert fs / expected <= 4  # frequency resolution 4 Hz or finer


# ---------- hum_zoom_figure ----------


def _loud_with_quiet_gap(fs, seconds=2.0, quiet=(1.2, 1.5)):
    """Loud noise everywhere except a near-silent segment [quiet[0], quiet[1]) s."""
    rng = np.random.default_rng(1)
    clean = 0.3 * rng.standard_normal(int(fs * seconds))
    a, b = int(quiet[0] * fs), int(quiet[1] * fs)
    clean[a:b] *= 0.001
    return clean


@pytest.mark.parametrize("fs", [16000, 8000])
def test_hum_zoom_figure_two_axes_no_pyplot(fs):
    clean = _loud_with_quiet_gap(fs)
    noisy = clean + 0.1 * np.sin(2 * np.pi * 50 * np.arange(len(clean)) / fs)
    before = plt.get_fignums()
    fig = hum_zoom_figure(clean, noisy, fs, 50)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert plt.get_fignums() == before
    ax_a, ax_b = fig.axes
    assert len(ax_a.lines) == 2 and len(ax_b.lines) == 1
    assert "5 cycles in 100 ms" in ax_b.get_title()
    t, hum = ax_b.lines[0].get_data()
    assert len(t) == round(0.1 * fs)
    start, _ = _quietest_window(clean, fs)
    np.testing.assert_allclose(hum, (noisy - clean)[start : start + len(t)])


@pytest.mark.parametrize("fs", [16000, 8000])
def test_hum_zoom_figure_short_signal_uses_all(fs):
    clean = _tone(440, fs, 0.05, 0.3)  # 50 ms < 100 ms window
    noisy = clean + 0.1 * np.sin(2 * np.pi * 60 * np.arange(len(clean)) / fs)
    before = plt.get_fignums()
    fig = hum_zoom_figure(clean, noisy, fs, 60)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert plt.get_fignums() == before
    assert len(fig.axes[1].lines[0].get_xdata()) == len(clean)
    assert _quietest_window(clean, fs) == (0, len(clean))


@pytest.mark.parametrize("fs", [16000, 8000])
def test_quietest_window_inside_known_quiet_segment(fs):
    quiet = (1.2, 1.5)
    clean = _loud_with_quiet_gap(fs, quiet=quiet)
    start, length = _quietest_window(clean, fs, duration=0.1)
    assert length == round(0.1 * fs)
    assert quiet[0] * fs <= start and start + length <= quiet[1] * fs  # whole window inside the gap
    assert start % round(0.01 * fs) == 0  # 10 ms search grid
    fig = hum_zoom_figure(clean, clean + 0.01, fs, 50)
    assert f"{start / fs:.2f} s" in fig.axes[0].get_title()  # absolute start shown in the title
