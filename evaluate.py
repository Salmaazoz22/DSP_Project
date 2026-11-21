"""Batch evaluation of hum removal: add hum -> detect -> notch -> SNR, one CSV row per experiment.

Reuses the project code exactly as app.py does. Usage (from the repo root):
    python evaluate.py --n-files 20 --snr 0,10,20 --f0 50,60 --out results/evaluation.csv
"""

import argparse
import csv
import inspect
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before any project module imports pyplot

import numpy as np

from src.analysis import add_hum, make_hum
from src.app_utils import read_upload, safe_detect
from src.metrics import snr_db
from src.notch_filter import apply_notch, design_notch

SKIP_S = 0.5  # extra measurement that ignores the notch start-up transient

COLUMNS = [
    "file",
    "fs",
    "duration_s",
    "f0_true",
    "target_snr_db",
    "q",
    "hum_harmonics",
    "hum_amps",
    "peak_hz",
    "prominence_db",
    "selected_f0",
    "detection_correct",
    "f0_used",
    "snr_before_db",
    "snr_after_db",
    "improvement_db",
    "filter_floor_db",
    "snr_before_skip_db",
    "snr_after_skip_db",
    "improvement_skip_db",
    "error",
]


def _parse_list(text, cast):
    try:
        values = [cast(v) for v in text.split(",") if v.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid list {text!r}: {e}") from e
    if not values:
        raise argparse.ArgumentTypeError(f"empty list {text!r}")
    return values


def hum_version():
    """Current defaults of make_hum (harmonics, amps), so each result records which hum produced it."""
    params = inspect.signature(make_hum).parameters
    return params["harmonics"].default, params["amps"].default


def select_files(data_dir, n_files, seed):
    """Sorted .wav files in data_dir, n_files of them picked reproducibly with default_rng(seed)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise FileNotFoundError(f"No .wav files in {data_dir}")
    n = min(n_files, len(files))
    idx = np.random.default_rng(seed).choice(len(files), size=n, replace=False)
    return [files[i] for i in sorted(idx)]


def _num(v, digits=4):
    return round(float(v), digits) if v is not None else None


def _gain(before, after):
    if before is None or after is None or not (math.isfinite(before) or math.isfinite(after)):
        return None
    return after - before


def _snr_pair(clean, noisy, filtered):
    return snr_db(clean, noisy), snr_db(clean, filtered)


def evaluate_file(path, f0_list, snr_list, q, harmonics, amps):
    """All experiments for one file. Never raises: failures are recorded in the 'error' column."""
    base = {"file": Path(path).name, "q": q, "hum_harmonics": str(harmonics), "hum_amps": str(amps)}
    try:
        clean, fs, info = read_upload(Path(path).read_bytes())
        base.update(fs=fs, duration_s=_num(info["duration_s"]))
        load_error = ""
    except Exception as e:  # noqa: BLE001 - record, don't crash
        clean, fs, load_error = None, None, f"load: {type(e).__name__}: {e}"

    rows = []
    floors = {}  # (f0_used, q) -> snr_db(clean, notch(clean)); computed once per file
    for f0 in f0_list:
        for snr in snr_list:
            row = {**base, "f0_true": f0, "target_snr_db": snr, "error": load_error}
            rows.append(row)
            if load_error:
                continue
            try:
                noisy = add_hum(clean, fs, f0=f0, snr_db=snr)
                res = safe_detect(noisy, fs)
                selected = res["selected_f0"] if res is not None else None
                if res is not None:
                    row.update(peak_hz=_num(res["measured_freq"]), prominence_db=_num(res["prominence_db"], 2))
                row.update(selected_f0=selected, detection_correct=selected == f0)
                f0_used = selected if selected is not None else f0
                row["f0_used"] = f0_used

                b, a = design_notch(f0_used, fs, q)
                filtered = apply_notch(noisy, b, a)
                before, after = _snr_pair(clean, noisy, filtered)
                row.update(snr_before_db=_num(before), snr_after_db=_num(after), improvement_db=_num(_gain(before, after)))
                if (f0_used, q) not in floors:  # best achievable SNR: notch on the clean speech alone
                    floors[(f0_used, q)] = _num(snr_db(clean, apply_notch(clean, b, a)))
                row["filter_floor_db"] = floors[(f0_used, q)]
                if len(clean) > fs:  # extra measurement only for signals longer than 1 s
                    k = int(SKIP_S * fs)
                    b_s, a_s = _snr_pair(clean[k:], noisy[k:], filtered[k:])
                    row.update(snr_before_skip_db=_num(b_s), snr_after_skip_db=_num(a_s), improvement_skip_db=_num(_gain(b_s, a_s)))
            except Exception as e:  # noqa: BLE001 - record, don't crash
                row["error"] = f"{type(e).__name__}: {e}"
    return rows


def write_csv(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: "" if row.get(c) is None else row.get(c) for c in COLUMNS})
    return out


def summarize(rows, harmonics, amps):
    lines = [f"\nExperiments: {len(rows)} | rows with error: {sum(bool(r['error']) for r in rows)}"]
    lines.append(f"Hum version (make_hum defaults): harmonics={harmonics}, amps={amps}")

    lines.append("\nDetection accuracy (selected_f0 == f0_true):")
    for f0 in sorted({r["f0_true"] for r in rows}):
        sub = [r for r in rows if r["f0_true"] == f0]
        ok = sum(r.get("detection_correct") is True for r in sub)
        lines.append(f"  {f0} Hz: {ok}/{len(sub)} ({100 * ok / len(sub):.1f}%)")

    lines.append("\nSNR improvement (dB)       full signal: mean / min / max     after skipping 0.5 s: mean / min / max")
    for f0 in sorted({r["f0_true"] for r in rows}):
        for snr in sorted({r["target_snr_db"] for r in rows}):
            sub = [r for r in rows if r["f0_true"] == f0 and r["target_snr_db"] == snr]
            cells = []
            for key in ("improvement_db", "improvement_skip_db"):
                vals = [r[key] for r in sub if r.get(key) is not None and math.isfinite(r[key])]
                cells.append(f"{np.mean(vals):7.2f} / {min(vals):7.2f} / {max(vals):7.2f}  (n={len(vals)})" if vals else "    n/a")
            lines.append(f"  f0={f0} Hz, SNR={snr:>3} dB:   {cells[0]}    {cells[1]}")

    lines.append("\nLowest filter_floor_db (SNR of clean vs notch(clean)), 3 files per f0:")
    for f0 in sorted({r["f0_true"] for r in rows}):
        per_file = {}
        for r in rows:
            v = r.get("filter_floor_db")
            if r["f0_true"] == f0 and v is not None:
                per_file[r["file"]] = min(v, per_file.get(r["file"], v))
        worst = sorted(per_file.items(), key=lambda kv: kv[1])[:3]
        lines.append(f"  {f0} Hz: " + (", ".join(f"{name} {v:.2f} dB" for name, v in worst) if worst else "n/a"))
    lines.append("  Low improvement at high input SNR is expected when snr_before approaches filter_floor_db.")
    return "\n".join(lines)


def run_evaluation(data_dir, n_files, seed, q, snr_list, f0_list, out):
    harmonics, amps = hum_version()
    files = select_files(data_dir, n_files, seed)
    print(f"Selected {len(files)} file(s) from {data_dir} (seed {seed}):")
    for p in files:
        print(f"  {p.name}")

    rows = []
    for p in files:
        rows.extend(evaluate_file(p, f0_list, snr_list, q, harmonics, amps))

    out = write_csv(rows, out)
    print(summarize(rows, harmonics, amps))
    print(f"\nWrote {len(rows)} rows to {out}")
    return rows


def build_parser():
    p = argparse.ArgumentParser(description="Batch evaluation of 50/60 Hz hum removal with the IIR notch filter.")
    p.add_argument("--data-dir", default="data/wav", help="folder with .wav files (default: data/wav)")
    p.add_argument("--n-files", type=int, default=20, help="number of files to evaluate (default: 20)")
    p.add_argument("--seed", type=int, default=0, help="seed for reproducible file selection (default: 0)")
    p.add_argument("--q", type=float, default=30.0, help="notch quality factor (default: 30)")
    p.add_argument("--snr", type=lambda s: _parse_list(s, float), default="0,10,20", help='target SNRs in dB (default: "0,10,20")')
    p.add_argument("--f0", type=lambda s: _parse_list(s, int), default="50,60", help='hum frequencies in Hz (default: "50,60")')
    p.add_argument("--out", default="results/evaluation.csv", help="output CSV (default: results/evaluation.csv)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_files < 1:
        raise SystemExit("--n-files must be at least 1")
    return run_evaluation(args.data_dir, args.n_files, args.seed, args.q, args.snr, args.f0, args.out)


if __name__ == "__main__":
    main()
