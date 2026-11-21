import csv
import math

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")  # evaluate -> app_utils -> Student 1's loader needs soundfile

import evaluate


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


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_evaluate_writes_one_row_per_experiment(wav_dir, tmp_path):
    out = tmp_path / "results" / "eval.csv"  # folder does not exist yet
    evaluate.main(["--data-dir", str(wav_dir), "--n-files", "2", "--snr", "0,10,20", "--f0", "50,60", "--out", str(out)])

    assert out.is_file()
    rows = _read_csv(out)
    assert len(rows) == 2 * 2 * 3  # files x f0 x snr
    assert set(evaluate.COLUMNS) <= set(rows[0].keys())
    cols = list(rows[0].keys())
    assert cols.index("filter_floor_db") == cols.index("improvement_db") + 1

    harmonics, amps = evaluate.hum_version()
    for r in rows:
        assert r["error"] == ""
        assert math.isfinite(float(r["improvement_db"]))
        floor = float(r["filter_floor_db"])  # finite, or inf if the notch leaves the clean signal unchanged
        assert math.isfinite(floor) or floor == math.inf
        assert int(r["f0_used"]) in (50, 60)
        assert r["hum_harmonics"] == str(harmonics) and r["hum_amps"] == str(amps)
        assert r["fs"] in ("16000", "8000")
        assert r["improvement_skip_db"] != ""  # 1.5 s > 1 s -> skip measurement filled
        assert math.isfinite(float(r["improvement_skip_db"]))
    assert {r["file"] for r in rows} == {"tones_16k.wav", "tones_8k.wav"}


def test_file_selection_is_reproducible(tmp_path):
    for i in range(6):
        _write_tones(tmp_path / f"f{i}.wav", 8000, seconds=0.3)
    a = evaluate.select_files(tmp_path, 3, seed=7)
    b = evaluate.select_files(tmp_path, 3, seed=7)
    assert a == b and len(a) == 3
    assert a == sorted(a)


def test_unreadable_file_is_recorded_not_dropped(wav_dir, tmp_path):
    (wav_dir / "broken.wav").write_bytes(b"not a wav")
    out = tmp_path / "eval.csv"
    evaluate.main(["--data-dir", str(wav_dir), "--n-files", "3", "--snr", "10", "--f0", "50", "--out", str(out)])

    rows = _read_csv(out)
    assert len(rows) == 3  # 3 files x 1 f0 x 1 snr, nothing dropped
    broken = [r for r in rows if r["file"] == "broken.wav"]
    assert len(broken) == 1 and broken[0]["error"].startswith("load:")
    assert all(r["error"] == "" for r in rows if r["file"] != "broken.wav")
