"""Effect of the notch quality factor Q: bandwidth, settling time and SNR improvement on real speech.

For each (Q, f0): measured -3 dB bandwidth (freqz) vs theory f0/Q, settling time on a pure sine vs
4.6*tau with tau = 1/(pi*BW), and mean/min/max improvement over the same files evaluate.py selects.
The notch is placed at the true f0 so only Q varies. Usage (from the repo root):
    python q_experiment.py --q 5,10,30,60 --f0 50,60 --snr 10 --n-files 20 --out-dir results
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before any project module imports pyplot

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import freqz

from evaluate import _parse_list, select_files
from src.analysis import add_hum
from src.app_utils import read_upload
from src.metrics import snr_db
from src.notch_filter import apply_notch, design_notch

SKIP_S = 0.5  # same start-up skip as evaluate.py
RESP_FS = 16000  # sample rate for the bandwidth / settling measurements
SINE_S = 2.0  # length of the pure sine used for the settling time
SETTLE_LEVEL = 0.01  # -40 dB of the unit input amplitude

COLUMNS = [
    "q",
    "f0",
    "bw_measured_hz",
    "bw_theory_hz",
    "settling_s",
    "tau_s",
    "t40_theory_s",
    "n_files",
    "improvement_mean_db",
    "improvement_min_db",
    "improvement_max_db",
    "improvement_skip_mean_db",
    "improvement_skip_min_db",
    "improvement_skip_max_db",
    "filter_floor_mean_db",
    "filter_floor_min_db",
    "filter_floor_max_db",
]


def _crossing(f, mag2, i, j):
    """Linear interpolation of the frequency where |H|^2 crosses 0.5 between grid points i and j."""
    return f[i] + (0.5 - mag2[i]) * (f[j] - f[i]) / (mag2[j] - mag2[i])


def measure_bandwidth(b, a, f0, fs, n_points=200001):
    """-3 dB width (Hz) of the notch, from freqz on a fine grid around f0."""
    half = 0.98 * min(f0, fs / 2 - f0)  # wide span (e.g. 1-99 Hz at 50 Hz), ~0.5 mHz resolution
    f = np.linspace(f0 - half, f0 + half, n_points)
    _, h = freqz(b, a, worN=f, fs=fs)
    mag2 = np.abs(h) ** 2
    below = np.flatnonzero(mag2 < 0.5)
    if below.size == 0 or below[0] == 0 or below[-1] == len(f) - 1:
        return None  # band edges not inside the grid
    lo, hi = below[0], below[-1]
    return float(_crossing(f, mag2, hi, hi + 1) - _crossing(f, mag2, lo - 1, lo))


def settling_time(b, a, f0, fs=RESP_FS, seconds=SINE_S, level=SETTLE_LEVEL):
    """First time after which |notch(sin)| stays below level for the rest of the signal (None if never)."""
    t = np.arange(int(seconds * fs)) / fs
    y = np.abs(apply_notch(np.sin(2 * np.pi * f0 * t), b, a))
    above = np.flatnonzero(y >= level)
    if above.size == 0:
        return 0.0
    if above[-1] == len(y) - 1:
        return None  # still above the level at the end
    return float((above[-1] + 1) / fs)


def _stats(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(min(vals)), float(max(vals))


def load_files(paths):
    signals = []
    for p in paths:
        try:
            clean, fs, _ = read_upload(Path(p).read_bytes())
            signals.append((Path(p).name, clean, fs))
        except Exception as e:  # noqa: BLE001 - report and skip unreadable files
            print(f"  skipped {Path(p).name}: {type(e).__name__}: {e}")
    return signals


def speech_metrics(signals, q, f0, snr):
    full, skip, floor = [], [], []
    for _, clean, fs in signals:
        b, a = design_notch(f0, fs, q)
        noisy = add_hum(clean, fs, f0=f0, snr_db=snr)
        filtered = apply_notch(noisy, b, a)
        full.append(snr_db(clean, filtered) - snr_db(clean, noisy))
        floor.append(snr_db(clean, apply_notch(clean, b, a)))
        if len(clean) > fs:
            k = int(SKIP_S * fs)
            skip.append(snr_db(clean[k:], filtered[k:]) - snr_db(clean[k:], noisy[k:]))
    return _stats(full), _stats(skip), _stats(floor)


def run_q_experiment(q_list, f0_list, snr, signals):
    rows = []
    for f0 in f0_list:
        for q in q_list:
            b, a = design_notch(f0, RESP_FS, q)
            bw_theory = f0 / q
            tau = 1 / (np.pi * bw_theory)
            (im, imn, imx), (sm, smn, smx), (fm, fmn, fmx) = speech_metrics(signals, q, f0, snr)
            rows.append(
                dict(
                    q=q, f0=f0,
                    bw_measured_hz=measure_bandwidth(b, a, f0, RESP_FS), bw_theory_hz=bw_theory,
                    settling_s=settling_time(b, a, f0), tau_s=tau, t40_theory_s=4.6 * tau,
                    n_files=len(signals),
                    improvement_mean_db=im, improvement_min_db=imn, improvement_max_db=imx,
                    improvement_skip_mean_db=sm, improvement_skip_min_db=smn, improvement_skip_max_db=smx,
                    filter_floor_mean_db=fm, filter_floor_min_db=fmn, filter_floor_max_db=fmx,
                )
            )
    return rows


def write_csv(rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: "" if r[c] is None else (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in COLUMNS})
    return out


def make_figure(rows, q_list, f0_list, snr):
    fig = Figure(figsize=(12, 9))  # not registered with pyplot -> no state leaks
    (ax_a, ax_b), (ax_c, ax_d) = fig.subplots(2, 2)

    f = np.linspace(30, 70, 4001)
    for q in q_list:
        _, h = freqz(*design_notch(50, RESP_FS, q), worN=f, fs=RESP_FS)
        ax_a.plot(f, 20 * np.log10(np.abs(h) + 1e-12), label=f"Q={q:g}")
    ax_a.axhline(-3, color="gray", linestyle=":", linewidth=1, label="-3 dB")
    ax_a.set(xlim=(30, 70), ylim=(-40, 2), xlabel="Frequency (Hz)", ylabel="Magnitude (dB)",
             title=f"(a) Notch response at f0 = 50 Hz (fs = {RESP_FS} Hz)")

    for f0 in f0_list:
        sub = [r for r in rows if r["f0"] == f0]
        qs = [r["q"] for r in sub]
        line, = ax_b.plot(qs, [np.nan if r["settling_s"] is None else r["settling_s"] for r in sub], "o-",
                          label=f"{f0} Hz measured")
        ax_b.plot(qs, [r["t40_theory_s"] for r in sub], "--", color=line.get_color(), label=f"{f0} Hz 4.6·tau")
        line, = ax_c.plot(qs, [r["improvement_mean_db"] for r in sub], "o-", label=f"{f0} Hz full")
        ax_c.plot(qs, [r["improvement_skip_mean_db"] for r in sub], "s--", color=line.get_color(),
                  label=f"{f0} Hz skip {SKIP_S} s")
        ax_d.plot(qs, [r["filter_floor_mean_db"] for r in sub], "o-", label=f"{f0} Hz")
    ax_b.set(xlabel="Q", ylabel="Time (s)", title="(b) Settling time to -40 dB on a pure sine")
    ax_c.set(xlabel="Q", ylabel="Mean improvement (dB)", title=f"(c) SNR improvement at input SNR {snr:g} dB")
    ax_d.set(xlabel="Q", ylabel="Mean filter_floor_db (dB)", title="(d) Filter floor: SNR of clean vs notch(clean)")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize="small")
    for ax in (ax_b, ax_c, ax_d):
        ax.set_xticks(q_list)
    fig.tight_layout()
    return fig


def format_table(rows):
    def fmt(v, spec):
        return f"{v:{spec}}" if v is not None else "n/a".rjust(len(f"{0:{spec}}"))

    head = ("   Q  f0 | BW meas  BW f0/Q |  settle  4.6tau |  impr full  impr skip  floor  (means, dB)")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['q']:4g} {r['f0']:3d} | {fmt(r['bw_measured_hz'], '7.3f')}  {r['bw_theory_hz']:7.3f} | "
            f"{fmt(r['settling_s'], '7.3f')} {r['t40_theory_s']:7.3f} | "
            f"{fmt(r['improvement_mean_db'], '9.2f')}  {fmt(r['improvement_skip_mean_db'], '9.2f')}  "
            f"{fmt(r['filter_floor_mean_db'], '5.2f')}"
        )
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description="Effect of the notch quality factor Q on bandwidth, settling and SNR.")
    p.add_argument("--q", type=lambda s: _parse_list(s, float), default="5,10,30,60", help='Q values (default: "5,10,30,60")')
    p.add_argument("--f0", type=lambda s: _parse_list(s, int), default="50,60", help='hum frequencies in Hz (default: "50,60")')
    p.add_argument("--snr", type=float, default=10.0, help="input SNR in dB (default: 10)")
    p.add_argument("--n-files", type=int, default=20, help="number of files (default: 20)")
    p.add_argument("--seed", type=int, default=0, help="seed, same selection as evaluate.py (default: 0)")
    p.add_argument("--data-dir", default="data/wav", help="folder with .wav files (default: data/wav)")
    p.add_argument("--out-dir", default="results", help="output folder (default: results)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_files < 1:
        raise SystemExit("--n-files must be at least 1")
    q_list = sorted(args.q)
    files = select_files(args.data_dir, args.n_files, args.seed)
    print(f"Selected {len(files)} file(s) from {args.data_dir} (seed {args.seed}), input SNR {args.snr:g} dB")
    signals = load_files(files)

    rows = run_q_experiment(q_list, args.f0, args.snr, signals)
    out_dir = Path(args.out_dir)
    csv_path = write_csv(rows, out_dir / "q_experiment.csv")
    png_path = out_dir / "q_experiment.png"
    make_figure(rows, q_list, args.f0, args.snr).savefig(png_path, dpi=120)

    print(format_table(rows))
    print(f"\nWrote {len(rows)} rows to {csv_path} and figure to {png_path}")
    return rows


if __name__ == "__main__":
    main()
