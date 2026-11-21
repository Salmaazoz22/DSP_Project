from pathlib import Path

import numpy as np
import pytest

from src.metrics import snr_db
from src.notch_filter import apply_notch, design_notch

WAV_PATH = Path(__file__).resolve().parent.parent / "data" / "wav" / "arctic_a0001.wav"


def test_identical_signals_return_inf():
    x = np.random.default_rng(0).standard_normal(1000)
    assert snr_db(x, x) == float("inf")


def test_hand_calculated_case():
    assert snr_db([1, 1, 1, 1], [1, 1, 1, 0]) == pytest.approx(10 * np.log10(4))
    assert snr_db([1, 1, 1, 1], [1, 1, 1, 0]) == pytest.approx(6.0206, abs=1e-4)


def test_known_20db_noise():
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(10000)
    noise = rng.standard_normal(10000)
    noise *= np.sqrt(np.sum(clean**2) / (np.sum(noise**2) * 10 ** (20 / 10)))
    assert snr_db(clean, clean + noise) == pytest.approx(20.0, abs=1e-9)


def test_float32_input_returns_python_float():
    rng = np.random.default_rng(2)
    clean = rng.standard_normal(1000).astype(np.float32)
    test = (clean + 0.1 * rng.standard_normal(1000)).astype(np.float32)
    result = snr_db(clean, test)
    assert type(result) is float
    assert np.isfinite(result)


def test_list_input_returns_python_float():
    result = snr_db([1.0, 2.0, 3.0], [1.0, 2.0, 3.5])
    assert type(result) is float


@pytest.mark.parametrize(
    "clean, test",
    [
        (np.ones(4), np.ones(5)),  # length mismatch
        ([], []),  # empty
        (np.ones(4), []),
        (np.ones((4, 1)), np.ones((4, 1))),  # 2-D
        (np.ones(4), np.ones((2, 2))),
        (np.ones(4, dtype=complex), np.ones(4)),  # complex
        (np.ones(4), np.ones(4, dtype=complex)),
        ([True, False, True, True], np.ones(4)),  # bool
        (np.ones(4), np.ones(4, dtype=bool)),
        ([1.0, np.nan, 1.0, 1.0], np.ones(4)),  # NaN / inf
        (np.ones(4), [1.0, np.inf, 1.0, 1.0]),
        (np.ones(4), [1.0, -np.inf, 1.0, 1.0]),
        (np.zeros(4), np.ones(4)),  # all-zero clean
        (np.zeros(4), np.zeros(4)),
    ],
)
def test_invalid_inputs_raise(clean, test):
    with pytest.raises(ValueError):
        snr_db(clean, test)


def test_real_speech_snr_improves_after_notch():
    sf = pytest.importorskip("soundfile")
    if not WAV_PATH.is_file():
        pytest.skip(f"dataset file not found: {WAV_PATH}")

    clean, fs = sf.read(WAV_PATH, dtype="float64")
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    noisy = clean + 0.2 * np.sin(2 * np.pi * 50 * np.arange(len(clean)) / fs)
    filtered = apply_notch(noisy, *design_notch(50, fs))

    assert snr_db(clean, filtered) > snr_db(clean, noisy)
