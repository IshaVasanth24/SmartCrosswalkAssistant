import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from gtts import gTTS
import os
import pygame
import time
import tempfile
import threading

# Initialize pygame mixer for audio playback
pygame.mixer.init()

st.title("🚦 Smart Pedestrian Assistant for Visually Impaired")

# Indian language selection
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Konkani": "kok",
    "Tulu": None
}

selected_lang = st.selectbox("Select Alert Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[selected_lang]

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO("yolov11s_blindaid_best.pt")

model = load_model()

# Indian language alert messages
ALERT_MESSAGES = {
    "en": {
        "moving": "Warning! Vehicles are moving. Do not cross.",
        "stopped": "Caution. {count} vehicles detected but stopped. Proceed with care.",
        "clear": "The path is clear. Safe to cross now."
    },
    "hi": {
        "moving": "चेतावनी! वाहन चल रहे हैं। पार न करें।",
        "stopped": "सावधान। {count} वाहनों का पता चला है लेकिन वे रुके हुए हैं। सावधानी से आगे बढ़ें।",
        "clear": "रास्ता साफ है। अब सुरक्षित रूप से पार कर सकते हैं।"
    },
    "kn": {
        "moving": "ಎಚ್ಚರಿಕೆ! ವಾಹನಗಳು ಚಲಿಸುತ್ತಿವೆ. ದಾಟಬೇಡಿ.",
        "stopped": "ಜಾಗರೂಕರಾಗಿರಿ. {count} ವಾಹನಗಳು ಕಂಡುಬಂದಿವೆ ಆದರೆ ನಿಲ್ಲಿಸಲಾಗಿದೆ. ಜಾಗರೂಕತೆಯಿಂದ ಮುಂದುವರಿಯಿರಿ.",
        "clear": "ಮಾರ್ಗ ಸ್ಪಷ್ಟವಾಗಿದೆ. ಈಗ ಸುರಕ್ಷಿತವಾಗಿ ದಾಟಬಹುದು."
    },
    "ta": {
        "moving": "எச்சரிக்கை! வாகனங்கள் நகரும். கடக்க வேண்டாம்.",
        "stopped": "எச்சரிக்கை. {count} வாகனங்கள் கண்டறியப்பட்டன ஆனால் நிறுத்தப்பட்டன. கவனத்துடன் தொடரவும்.",
        "clear": "பாதை தெளிவாக உள்ளது. இப்போது பாதுகாப்பாக கடக்கலாம்."
    },
    "te": {
        "moving": "హెచ్చరిక! వాహనాలు కదులుతున్నాయి. దాటవద్దు.",
        "stopped": "జాగ్రత్త. {count} వాహనాలు కనుగొనబడ్డాయి కానీ ఆపబడ్డాయి. జాగ్రత్తగా ముందుకు సాగండి.",
        "clear": "మార్గం స్పష్టంగా ఉంది. ఇప్పుడు సురక్షితంగా దాటవచ్చు."
    },
    "kok": {
        "moving": "Xetavanni! Vaahan chalta. Poddunk naka.",
        "stopped": "Savdhani. {count} vaahan mell'l'le ani thamble asat. Savdhanean voch.",
        "clear": "Marg mullav. Ata surakshit poddunk zai."
    },
    None: {
        "moving": "Tulu: Warning! Vehicles moving. Barpandhe.",
        "stopped": "Tulu: {count} vehicles stopped. Savdhana.",
        "clear": "Tulu: Safe to cross. Barpuji."
    }
}

# Function to handle audio playback with temp files
def speak(text, lang):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # For Tulu, use Kannada TTS as fallback
        tts_lang = "kn" if lang is None else lang
        
        # For Konkani in Roman script, use English TTS
        if lang == "kok":
            tts_lang = "en"
            
        # Generate TTS
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(temp_path)
        
        # Wait if another audio is playing
        while pygame.mixer.get_busy():
            time.sleep(0.1)
            
        # Play the audio
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Clean up in a separate thread after playback completes
        def cleanup():
            start_time = time.time()
            while pygame.mixer.get_busy() and (time.time() - start_time) < 5:  # Timeout after 5 seconds
                time.sleep(0.1)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
        
        threading.Thread(target=cleanup, daemon=True).start()
        
    except Exception as e:
        st.warning(f"Audio error: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

# Track vehicle positions across frames
def detect_and_track_motion(frame, prev_tracks):
    results = model(frame, verbose=False)[0]  # Get prediction result

    motion_detected = False
    current_tracks = {}
    vehicle_count = 0

    # Loop through all detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label.lower() not in ["car", "truck", "bus", "motorcycle"]:  # Filter only vehicles
            continue

        vehicle_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        current_tracks[(cx, cy)] = (x1, y1, x2, y2)

        # Check movement by comparing with previous frame
        if prev_tracks:
            closest = min(prev_tracks.keys(), key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2, default=None)
            if closest:
                old_cx, old_cy = closest
                distance = ((old_cx - cx) ** 2 + (old_cy - cy) ** 2) ** 0.5
                if distance > 20:  # Movement threshold
                    motion_detected = True

    return current_tracks, motion_detected, vehicle_count

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Create temp file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    prev_tracks = {}
    last_alert_time = 0
    last_status = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))  # Resize for speed

        current_tracks, motion, vehicle_count = detect_and_track_motion(frame, prev_tracks)
        prev_tracks = current_tracks

        # Draw bounding boxes
        for (cx, cy), (x1, y1, x2, y2) in current_tracks.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Decision logic
        if motion:
            status_text = "🚫 Don't Cross - Vehicles Moving"
            status_audio = ALERT_MESSAGES[lang_code]["moving"] if lang_code else ALERT_MESSAGES[None]["moving"]
            color = (0, 0, 255)
        elif current_tracks:
            status_text = "✅ Safe to Walk - Vehicles Stopped"
            status_audio = (ALERT_MESSAGES[lang_code]["stopped"] if lang_code else ALERT_MESSAGES[None]["stopped"]).format(count=vehicle_count)
            color = (0, 255, 0)
        else:
            status_text = "✅ Safe to Walk - No Vehicles"
            status_audio = ALERT_MESSAGES[lang_code]["clear"] if lang_code else ALERT_MESSAGES[None]["clear"]
            color = (0, 255, 0)

        # Update status text on frame
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Audio alerts (throttled to avoid too frequent playback)
        current_time = time.time()
        if (status_audio != last_status or current_time - last_alert_time > 5) and not pygame.mixer.get_busy():
            speak(status_audio, lang_code)
            last_alert_time = current_time
            last_status = status_audio

        # Show in Streamlit
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    
    # Clean up temporary video file
    try:
        os.remove(video_path)
        os.rmdir(temp_dir)
    except:
        pass