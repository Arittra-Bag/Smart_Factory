import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from new_fact_stream import SmartFactoryController
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# Set page config
st.set_page_config(page_title="Smart Factory Control", layout="wide")

# Auto-refresh every 2 seconds for metrics/alerts
st_autorefresh(interval=2000, key="refresh")

# Title
st.title("🤖 Smart Factory Control System")
st.markdown("""
### 🧠 Gesture Controls:
- ✊ **Fist**: Emergency Stop  
- ✌️ **Peace Sign**: Start Production  
- 🖐️ **Palm**: Trigger Quality Inspection / Reset Emergency  
""")

# -------------------------------------------------------
# Video Processor
# -------------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.controller = SmartFactoryController()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed = self.controller.process_frame(image)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

# Keep a shared instance
if "video_processor_instance" not in st.session_state:
    st.session_state.video_processor_instance = VideoProcessor()

video_processor_instance = st.session_state.video_processor_instance
controller = video_processor_instance.controller

# -------------------------------------------------------
# Webcam with better resolution & FPS
# -------------------------------------------------------
webrtc_streamer(
    key="smart-factory-stream",
    video_processor_factory=lambda: video_processor_instance,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    }
)

# -------------------------------------------------------
# Sidebar: Live Metrics
# -------------------------------------------------------
with st.sidebar:
    st.header("📊 System Metrics")
    st.metric("Status", controller.machine_status)
    st.metric("Production Count", controller.production_count)
    st.metric("Quality Score", f"{controller.quality_score:.2f}%")
    defect_rate = (controller.defect_count / controller.batch_size) * 100 if controller.batch_size > 0 else 0
    st.metric("Defect Rate", f"{defect_rate:.2f}%")

    if controller.emergency_mode:
        st.warning("🚨 Emergency Stop Activated")

    st.markdown("### 🧾 Recent Alerts")
    for alert in reversed(controller.alerts[-5:]):
        st.write(f"- {alert}")

# -------------------------------------------------------
# Batch Report + Conditional Download
# -------------------------------------------------------
st.markdown("### 📈 Batch Analytics")

# Track report generation
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

# Generate report button
if st.button("📊 Generate Batch vs. Defect Rate Report"):
    st.session_state.report_ready = True

# Show chart + download only if report has been generated
if st.session_state.report_ready:
    fig = controller.plot_defect_rate()
    if fig:
        st.pyplot(fig)

        try:
            df = pd.read_csv(controller.safety_check_records)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV Report", csv, "safety_check_records.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not load report: {e}")
