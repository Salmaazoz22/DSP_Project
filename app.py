import math
from pathlib import Path

import streamlit as st

# app_utils must be imported before anything that imports pyplot: it selects the Agg backend
from src.app_utils import (
    detection_png,
    read_upload,
    safe_detect,
    spectrogram_figure,
    spectrum_figure,
    to_wav,
    waveform_figure,
)
import matplotlib.pyplot as plt

from src.analysis import add_hum
from src.metrics import snr_db
from src.notch_filter import apply_notch, design_notch, plot_pole_zero, plot_response

FILE_KEYS = ("file_key", "clean", "fs", "info", "stem")
RESULT_KEYS = (
    "noisy",
    "noisy_params",
    "detection_done",
    "detection",
    "detection_png",
    "filtered",
    "filter_params",
    "coeffs",
)
FILTER_KEYS = ("filtered", "filter_params", "coeffs")
SKIP_S = 0.5  # extra SNR measurement that ignores the notch start-up transient

ss = st.session_state


def clear(keys):
    for k in keys:
        ss.pop(k, None)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def fmt_db(v):
    return "∞ dB (identical to clean)" if math.isinf(v) else f"{v:.2f} dB"


def notch_frequency(sidebar_f0):
    res = ss.get("detection")
    if res is not None and res["selected_f0"] is not None:
        return res["selected_f0"]
    return sidebar_f0


def snr_row(clean, noisy, filtered, label):
    try:
        before, after = snr_db(clean, noisy), snr_db(clean, filtered)
    except ValueError as e:
        st.error(f"SNR ({label}) could not be computed: {e}")
        return
    if math.isfinite(before) and math.isfinite(after):
        gain = f"{round(after - before, 2) + 0.0:+.2f} dB"  # + 0.0 turns -0.0 into 0.0
    elif math.isinf(after) and math.isfinite(before):
        gain = "∞"
    else:
        gain = None
    c1, c2, c3 = st.columns(3)
    c1.metric(f"SNR before ({label})", fmt_db(before))
    c2.metric(
        f"SNR after ({label})",
        fmt_db(after),
        delta=gain if gain not in (None, "∞") else None,
    )
    c3.metric(f"Improvement ({label})", gain if gain is not None else "—")


# ---------------------------------------------------------------- page + sidebar

st.set_page_config(page_title="Speech Hum Removal", layout="wide")
st.title("Speech Hum Removal")
st.caption("Add 50/60 Hz mains hum to speech, detect it, and remove it with a second-order IIR notch filter.")

sb = st.sidebar
sb.header("Controls")
uploaded = sb.file_uploader("WAV file", type=["wav"])
f0 = sb.radio("Mains frequency (Hz)", [50, 60], horizontal=True)
snr_target = sb.slider("Hum level (target SNR, dB)", min_value=0, max_value=30, value=10)
with sb.expander("Advanced"):
    Q = st.number_input("Notch quality factor Q", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
add_clicked = sb.button("Add hum")
detect_clicked = sb.button("Detect hum")
notch_clicked = sb.button("Apply notch")
reset_clicked = sb.button("Reset")

# ---------------------------------------------------------------- state: file

if uploaded is None:
    clear(FILE_KEYS + RESULT_KEYS)
    st.info("Upload a WAV file in the sidebar to begin.")
    st.stop()

file_key = f"{uploaded.name}:{uploaded.size}"
if ss.get("file_key") != file_key:
    clear(FILE_KEYS + RESULT_KEYS)
    try:
        clean, fs, info = read_upload(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()
    ss["file_key"], ss["clean"], ss["fs"], ss["info"] = file_key, clean, fs, info
    ss["stem"] = Path(uploaded.name).stem

clean, fs, info, stem = ss["clean"], ss["fs"], ss["info"], ss["stem"]

# ---------------------------------------------------------------- state: invalidation

if reset_clicked:
    clear(RESULT_KEYS)
    sb.info("Results cleared.")
if "noisy" in ss and ss["noisy_params"] != (f0, snr_target):
    clear(RESULT_KEYS)  # hum settings changed -> everything downstream is stale
if "filtered" in ss and ss["filter_params"][1] != Q:
    clear(FILTER_KEYS)

# ---------------------------------------------------------------- actions

if add_clicked:
    clear(RESULT_KEYS)
    ss["noisy"] = add_hum(clean, fs, f0=f0, snr_db=snr_target)
    ss["noisy_params"] = (f0, snr_target)

if detect_clicked:
    if "noisy" not in ss:
        sb.info("Add hum first.")
    else:
        res = safe_detect(ss["noisy"], fs)
        ss["detection_done"], ss["detection"] = True, res
        ss["detection_png"] = detection_png(ss["noisy"], fs, res) if res is not None else None
        clear(FILTER_KEYS)  # notch frequency may change

if notch_clicked:
    if "noisy" not in ss:
        sb.info("Add hum first.")
    else:
        f0_used = notch_frequency(f0)
        b, a = design_notch(f0_used, fs, Q)
        ss["coeffs"] = (b, a)
        ss["filtered"] = apply_notch(ss["noisy"], b, a)
        ss["filter_params"] = (f0_used, Q)

# ---------------------------------------------------------------- 1 Original

st.header("1 · Original audio")
c = st.columns(4)
c[0].metric("Sample rate", f"{info['fs']} Hz")
c[1].metric("Channels (file)", info["channels"])
c[2].metric("Duration", f"{info['duration_s']:.2f} s")
c[3].metric("Format", info["subtype"])
c = st.columns(4)
c[0].metric("Samples", f"{info['frames']:,}")
c[1].metric("Peak", f"{info['peak']:.3f}")
c[2].metric("RMS", f"{info['rms']:.4f}")
st.caption(f"{uploaded.name}: mixed to mono, DC removed, kept at the native sample rate.")
st.audio(to_wav(clean, fs), format="audio/wav")
show_fig(waveform_figure({"original": clean}, fs))

# ---------------------------------------------------------------- 2 Noisy

st.header("2 · Audio with hum")
if "noisy" not in ss:
    st.info("Choose the mains frequency and hum level, then press **Add hum**.")
else:
    noisy = ss["noisy"]
    hum_f0, hum_snr = ss["noisy_params"]
    c = st.columns(3)
    c[0].metric("Hum frequency", f"{hum_f0} Hz")
    c[1].metric("Target SNR", f"{hum_snr} dB")
    try:
        c[2].metric("Measured SNR before filtering", fmt_db(snr_db(clean, noisy)))
    except ValueError as e:
        c[2].metric("Measured SNR before filtering", "—")
        st.error(f"SNR could not be computed: {e}")
    st.audio(to_wav(noisy, fs), format="audio/wav")
    show_fig(waveform_figure({"clean": clean, "noisy": noisy}, fs))
    show_fig(spectrum_figure({"clean": clean, "noisy": noisy}, fs, fmax=200))

# ---------------------------------------------------------------- 3 Detection

st.header("3 · Hum detection")
if "noisy" not in ss:
    st.info("Add hum first, then press **Detect hum**.")
elif not ss.get("detection_done"):
    st.info("Press **Detect hum** to analyse the 40–70 Hz band.")
else:
    res = ss["detection"]
    if res is None:
        st.warning(f"Signal too short to analyse; the notch will use the sidebar frequency ({f0} Hz).")
    else:
        c = st.columns(3)
        c[0].metric("Measured peak", f"{res['measured_freq']:.2f} Hz")
        c[1].metric("Prominence", f"{res['prominence_db']:.1f} dB")
        c[2].metric("Selected mains", f"{res['selected_f0']} Hz" if res["selected_f0"] is not None else "none")
        st.write(res["message"])
        if res["selected_f0"] is None:
            st.warning(f"No clear hum detected; the notch will use the sidebar frequency ({f0} Hz).")
        else:
            st.success(f"The notch will be centred at the detected {res['selected_f0']} Hz.")
        st.image(ss["detection_png"], use_column_width=True)

# ---------------------------------------------------------------- 4 Filtered

st.header("4 · Filtered audio")
if "noisy" not in ss:
    st.info("Add hum first, then press **Apply notch**.")
elif "filtered" not in ss:
    st.info(f"Press **Apply notch** (it will use {notch_frequency(f0)} Hz, Q = {Q:g}).")
else:
    noisy, filtered = ss["noisy"], ss["filtered"]
    (b, a), (f0_used, q_used) = ss["coeffs"], ss["filter_params"]
    st.write(f"Notch at **{f0_used} Hz**, Q = **{q_used:g}**, fs = {fs} Hz")
    st.code(
        "b = [" + ", ".join(f"{v:.8f}" for v in b) + "]\n"
        "a = [" + ", ".join(f"{v:.8f}" for v in a) + "]",
        language="text",
    )
    st.audio(to_wav(filtered, fs), format="audio/wav")
    show_fig(plot_response(b, a, fs))
    show_fig(plot_pole_zero(b, a))
    show_fig(spectrum_figure({"noisy": noisy, "filtered": filtered}, fs, fmax=200))
    show_fig(spectrogram_figure({"noisy": noisy, "filtered": filtered}, fs))
    st.caption("Hum appears as horizontal lines; after the notch the 50/60 Hz line should disappear.")
    show_fig(waveform_figure({"noisy": noisy, "filtered": filtered}, fs))

# ---------------------------------------------------------------- 5 Results

st.header("5 · Results")
if "filtered" not in ss:
    st.info("Results appear after the notch filter has been applied.")
else:
    noisy, filtered = ss["noisy"], ss["filtered"]
    snr_row(clean, noisy, filtered, "full signal")
    if len(clean) > fs:
        skip = int(SKIP_S * fs)
        snr_row(clean[skip:], noisy[skip:], filtered[skip:], f"after skipping first {SKIP_S} s")
        st.caption(
            "The full-signal value is the primary result. The second row excludes the filter's "
            "start-up transient (poles close to the unit circle ring for a few tenths of a second)."
        )
    st.download_button(
        "Download filtered WAV",
        data=to_wav(filtered, fs),
        file_name=f"{stem}_filtered.wav",
        mime="audio/wav",
    )
