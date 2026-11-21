import csv

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")  # q_experiment -> app_utils -> Student 1's loader needs soundfile

import q_experiment

Q_LIST = [5, 10, 30, 60]


def _write_tones(path, fs, seconds=1.5):
    t = np.arange(int(fs * seconds)) / fs
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    sf.write(path, x, fs, subtype="PCM_16")


@pytest.fixture
def wav_dir(tmp_path):
    d = tmp_path / "wav"
    d.mkdir()
    _write_tones(d / "tones_16k.wav", 16000)
    _write_tones(d / "tones_8k.wav", 8000)
    return d


@pytest.fixture
def run(wav_dir, tmp_path):
    out_dir = tmp_path / "out"  # does not exist yet
    rows = q_experiment.main(
        ["--data-dir", str(wav_dir), "--n-files", "2", "--q", "5,10,30,60", "--f0", "50,60", "--out-dir", str(out_dir)]
    )
    return rows, out_dir


def test_bandwidth_shrinks_and_settling_grows_with_q(run):
    rows, _ = run
    for f0 in (50, 60):
        sub = sorted((r for r in rows if r["f0"] == f0), key=lambda r: r["q"])
        assert [r["q"] for r in sub] == Q_LIST
        bw = [r["bw_measured_hz"] for r in sub]
        settle = [r["settling_s"] for r in sub]
        assert None not in bw and None not in settle
        assert all(x > y for x, y in zip(bw, bw[1:]))
        assert all(x < y for x, y in zip(settle, settle[1:]))
        for r in sub:  # measured bandwidth matches the f0/Q design
            assert r["bw_measured_hz"] == pytest.approx(r["bw_theory_hz"], rel=0.02)


def test_csv_has_one_row_per_q_f0_and_png_exists(run):
    _, out_dir = run
    with open(out_dir / "q_experiment.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(Q_LIST) * 2
    assert {(float(r["q"]), int(r["f0"])) for r in rows} == {(float(q), f) for q in Q_LIST for f in (50, 60)}
    assert list(rows[0].keys()) == q_experiment.COLUMNS
    assert all(r["n_files"] == "2" for r in rows)

    png = out_dir / "q_experiment.png"
    assert png.is_file() and png.stat().st_size > 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
