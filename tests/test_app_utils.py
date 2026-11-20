import io

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")  # app_utils needs soundfile (via Student 1's loader)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.app_utils import (
    detection_png,
    read_upload,
    safe_detect,
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
