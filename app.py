import streamlit as st
import cv2
import tempfile
import numpy as np
from utils import process_video, speak, LANG_CODES, SAFE_MSG, UNSAFE_MSG

st.set_page_config(page_title="🦮 Smart Crosswalk Assistant", layout="centered")
st.title("🦯 Smart Crosswalk Assistant for Visually Impaired")

lang_choice = st.sidebar.selectbox("🌐 Choose Language", list(LANG_CODES.keys()))
lang_code = LANG_CODES[lang_choice]

st.subheader("Choose Input Method")
input_mode = st.radio("", ("Live Camera", "Upload Video"), horizontal=True)

# Initialize state variables
if 'last_status' not in st.session_state:
    st.session_state.last_status = None
if 'prev_positions' not in st.session_state:
    st.session_state.prev_positions = None

def process_frame(frame):
    detections, is_safe, current_positions = process_video(frame, st.session_state.prev_positions)
    st.session_state.prev_positions = current_positions

    # Speak only if status changed
    if is_safe != st.session_state.last_status:
        if is_safe:
            speak(SAFE_MSG[lang_code], lang_code=lang_code)
        else:
            speak(UNSAFE_MSG[lang_code], lang_code=lang_code)
        st.session_state.last_status = is_safe

    return detections, is_safe

if input_mode == "Live Camera":
    stframe = st.empty()
    cam_input = st.camera_input("📸 Point your camera to the crosswalk")

    if cam_input:
        # Convert image to OpenCV format
        file_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, (640, 384))

        detections, is_safe = process_frame(frame_resized)
        stframe.image(frame_resized)
        safety_text = "✅ Safe to Cross" if is_safe else "❌ Do Not Cross"
        st.write(f"Detections: {detections}")
        st.markdown(f"### {safety_text}")

else:
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mpeg4'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame, (640, 384))

                detections, is_safe = process_frame(frame_resized)
                stframe.image(frame_resized)
                safety_text = "✅ Safe to Cross" if is_safe else "❌ Do Not Cross"
                st.write(f"Detections: {detections}")
                st.markdown(f"### {safety_text}")

                
        finally:
            cap.release()
