import streamlit as st
import sounddevice as sd
import queue
import os
import time
import json
import pandas as pd
from vosk import Model, KaldiRecognizer

# === Configuration ===
MODEL_PATH = "model"
CSV_LOG = "factory_log.csv"
COMMANDS = {"start": "Production started",
            "stop": "Production stopped",
            "check": "Quality check triggered"}

# === State Initialization ===
def initialize_state():
    defaults = {
        "production_mode": False,
        "defect_count": 0,
        "batch_count": 0,
        "status": "STANDBY",
        "last_command": "None"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === Voice Command Logging ===
def log_batch(batch_size, defect_count, status):
    df = pd.DataFrame([{
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": batch_size,
        "defect_count": defect_count,
        "status": status
    }])
    if os.path.exists(CSV_LOG):
        df.to_csv(CSV_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_LOG, index=False)

# === Audio Stream Setup ===
def setup_audio_stream(callback):
    return sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                             channels=1, callback=callback)

# === Main App ===
def main():
    st.set_page_config(page_title="Voice-Controlled Factory", layout="centered")
    st.title("🛠️ Smart Factory Control via Voice")
    initialize_state()

    # Load Vosk model
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Vosk model not found at `{MODEL_PATH}`. Download and place it there.")
        return
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)
    audio_queue = queue.Queue()

    # Stream callback
    def audio_callback(indata, frames, time, status):
        if status:
            st.warning(f"Audio status: {status}")
        audio_queue.put(bytes(indata))

    # Start listening
    mic = setup_audio_stream(audio_callback)
    with mic:
        st.success("Microphone activated.")
        mic_status = st.empty()
        dashboard = st.empty()
        time_indicator = time.time()

        while True:
            if not audio_queue.empty():
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").lower()
                    for command in COMMANDS:
                        if command in text:
                            st.session_state["last_command"] = command

                            if command == "start" and not st.session_state.production_mode:
                                st.session_state.production_mode = True
                                st.session_state.status = "RUNNING"

                            elif command == "stop" and st.session_state.production_mode:
                                st.session_state.production_mode = False
                                st.session_state.status = "STOPPED"
                                log_batch(st.session_state.batch_count, st.session_state.defect_count, "STOPPED")
                                st.session_state.batch_count = 0
                                st.session_state.defect_count = 0

                            elif command == "check":
                                st.session_state.production_mode = False
                                st.session_state.status = "QUALITY CHECK"
                                log_batch(st.session_state.batch_count, st.session_state.defect_count, "QUALITY CHECK")
                                st.session_state.batch_count = 0
                                st.session_state.defect_count = 0

            # Simulate batch increment if in production mode
            if st.session_state.production_mode and time.time() - time_indicator > 2:
                st.session_state.batch_count += 1
                time_indicator = time.time()

            # Mic animation and dashboard update
            bar = "▇" * (int(time.time() * 10) % 10)
            mic_status.markdown(f"<h4 style='color: green;'>🎙️ Listening: {bar}</h4>", unsafe_allow_html=True)

            dashboard.markdown(f"""
            ### 🧾 Factory Dashboard
            - **Status**: `{st.session_state.status}`
            - **Current Batch Size**: `{st.session_state.batch_count}`
            - **Defect Count**: `{st.session_state.defect_count}`
            - **Last Command**: `{st.session_state.last_command}`
            """, unsafe_allow_html=True)

            time.sleep(0.1)  # Minimal sleep for responsiveness

if __name__ == "__main__":
    main()
