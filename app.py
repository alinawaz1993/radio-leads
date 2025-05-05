#!/usr/bin/env python3
import os
import glob
import json
import threading
import time
import datetime
import logging
import sys
import tempfile
from io import BytesIO

import streamlit as st
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────
STREAM_URL    = "https://broadcastify.cdnstream1.com/2846"
CHUNK_SEC     = 30
KBPS          = 16
BYTES_SEC     = (KBPS * 1000) // 8
CHUNK_SIZE    = 1024
TOTAL_CHUNKS  = BYTES_SEC * CHUNK_SEC // CHUNK_SIZE
MIN_DBFS      = -50
RETRY_TIMEOUT = 40  # seconds retry per chunk

# ─── Ephemeral storage setup ──────────────────────────────────────────────
tmp_dir = tempfile.mkdtemp()  # e.g. /tmp/tmpabcd1234
AUDIO_DIR = os.path.join(tmp_dir, "audio")
TRANS_DIR = os.path.join(tmp_dir, "transcripts")
LOG_FILE  = os.path.join(tmp_dir, "pipeline.log")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

# ─── Load secrets & client ─────────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)

STOP_EVENT  = threading.Event()
PAUSE_EVENT = threading.Event()
THREAD      = None

def log(msg):
    logging.info(msg)

def fetch_chunk():
    buf = BytesIO()
    resp = requests.get(STREAM_URL, stream=True, timeout=10)
    for i, ch in enumerate(resp.iter_content(CHUNK_SIZE)):
        buf.write(ch)
        if i >= TOTAL_CHUNKS:
            break
    buf.seek(0)
    return buf

def clean_and_level(buf):
    seg = AudioSegment.from_mp3(buf)
    if seg.dBFS < MIN_DBFS:
        raise RuntimeError("Silence/buffering")
    arr = np.array(seg.get_array_of_samples(), np.float32)
    arr /= np.iinfo(seg.array_type).max
    if seg.channels > 1:
        arr = arr.reshape((-1, seg.channels)).mean(axis=1)
    sr = seg.frame_rate
    try:
        rms = librosa.feature.rms(y=arr, frame_length=1024, hop_length=512, center=True)[0]
    except:
        rms = np.sqrt((arr**2).mean())
    gain = np.clip(0.1 / (rms + 1e-6), 0.5, 2.0)
    w = max(1, int((sr/512)*(100/1000)))
    k = np.ones(w)/w
    smooth = scipy.signal.filtfilt(k, [1.0], gain)
    ft = np.arange(len(smooth)) * 512
    env = np.interp(np.arange(len(arr)), ft, smooth)
    return arr * env, sr

def next_filename(prefix, ext, folder):
    pattern = os.path.join(folder, f"{prefix}_*.{ext}")
    n = len(glob.glob(pattern)) + 1
    return os.path.join(folder, f"{prefix}_{n:03d}.{ext}")

def extract_insights(transcript: str):
    prompt = f"""
You are an Emergency Response Assistant. Extract:

Summary:<one-sentence>
Locations:<comma-list or None>
Fire Location:<or None>
Priority:<High|Low>
Lead Type:<Medical|Fire|Police|Other>
Lead Quality:<Best|Good|Low>
Issues:<or None>
Recommendations:<or None>

Transcript:
\"\"\"{transcript}\"\"\"
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    out = resp.choices[0].message.content.strip()
    d = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":",1)
            d[k.strip().lower().replace(" ","_")] = v.strip()
    return d

def capture_loop():
    log("▶ Pipeline started")
    while not STOP_EVENT.is_set():
        if PAUSE_EVENT.is_set():
            time.sleep(0.5)
            continue

        # Fetch with retry
        buf = None
        start = time.time()
        while time.time() - start < RETRY_TIMEOUT and not STOP_EVENT.is_set():
            try:
                buf = fetch_chunk()
                log("Fetched chunk")
                break
            except Exception as e:
                log(f"Fetch error {e}, retry in 5s")
                time.sleep(5)
        if buf is None:
            log("Skipping after retries")
            continue

        # Clean
        try:
            audio, sr = clean_and_level(buf)
            log("Cleaned & leveled")
        except Exception as e:
            log(f"{e}, skipping")
            continue

        # Save WAV
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"clip_{ts}"
        wav    = next_filename(prefix, "wav", AUDIO_DIR)
        sf.write(wav, audio, sr)
        log(f"Saved WAV → {wav}")

        # Transcribe
        try:
            with open(wav, "rb") as f:
                tr = client.audio.transcriptions.create(
                    model="whisper-1", file=f,
                    response_format="text", language="en"
                )
            transcript = tr.strip()
            log("Transcription OK")
        except Exception as e:
            log(f"Transcription failed: {e}")
            continue

        # Insights
        try:
            insights = extract_insights(transcript)
            log("Insights OK")
        except Exception as e:
            log(f"Insights failed: {e}")
            insights = {}

        # Save TXT
        txt = next_filename(prefix, "txt", TRANS_DIR)
        with open(txt, "w") as tf:
            tf.write(transcript + "\n\n" + json.dumps(insights, indent=2))
        log(f"Saved TXT → {txt}")

    log("⏹ Pipeline stopped")

# ─── Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="🚒 AdjustLeads POC", layout="wide")
st.title("🚒 AdjustLeads POC")

# Controls
c1, c2, c3, c4, c5 = st.columns(5)
if c1.button("▶ Start Recording"):
    if THREAD is None or not THREAD.is_alive():
        STOP_EVENT.clear()
        PAUSE_EVENT.clear()
        t = threading.Thread(target=capture_loop, daemon=True)
        globals()["THREAD"] = t
        t.start()
        st.success("Recording started")
    else:
        st.warning("Already running")

if c2.button("⏸ Pause Recording"):
    PAUSE_EVENT.set()
    st.info("Paused")

if c3.button("▶ Resume Recording"):
    PAUSE_EVENT.clear()
    st.info("Resumed")

if c4.button("⏹ Stop Recording"):
    STOP_EVENT.set()
    st.info("Stopping...")

live = c5.checkbox("Live Mode", value=True)

if st.button("🔄 Refresh"):
    pass  # Streamlit reruns automatically

st.markdown("---")

# Playback
audio_files = sorted(glob.glob(f"{AUDIO_DIR}/clip_*.wav"))
if not audio_files:
    st.info("No recordings yet.")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0

if not live:
    pcol, ncol = st.columns(2)
    if pcol.button("◀ Previous") and st.session_state.idx > 0:
        st.session_state.idx -= 1
    if ncol.button("Next ▶") and st.session_state.idx < len(audio_files)-1:
        st.session_state.idx += 1

idx = len(audio_files)-1 if live else st.session_state.idx
wav_path = audio_files[idx]
st.markdown(f"**Clip {idx} of {len(audio_files)-1}:** `{os.path.basename(wav_path)}`")
st.audio(wav_path)

# Transcript
txt_path = wav_path.replace(os.path.basename(AUDIO_DIR), os.path.basename(TRANS_DIR)).replace(".wav", ".txt")
if os.path.exists(txt_path):
    raw, js = open(txt_path).read().split("\n\n", 1)
else:
    raw, js = "(no transcript)", "{}"

st.subheader("🎙 Full Transcript")
st.text_area("Transcript", raw, height=150)

# Insights
ins = json.loads(js)
st.subheader("🔧 Insights")
st.markdown(
    "<div style='background:#FFFBE6;border:1px solid #FFE58F;padding:12px;border-radius:8px;'>",
    unsafe_allow_html=True,
)
for key in ["summary","locations","fire_location","priority",
            "lead_type","lead_quality","issues","recommendations"]:
    st.write(f"**{key.replace('_',' ').title()}:** {ins.get(key,'')}")
st.markdown("</div>", unsafe_allow_html=True)
