import torch
import pandas as pd
from ultralytics import YOLO
import pyttsx3
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import tempfile
import os

# Load YOLO model
model = YOLO(r"C:\Users\hp\OneDrive\Desktop\CrosswalkAssistant\yolov11s_blind_aid_best.pt")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def detect_objects(image):
    results = model(image)
    boxes = results[0].boxes

    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'name', 'confidence'])

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    class_names = model.names

    data = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        label = class_names[cls[i]] if cls[i] in class_names else str(cls[i])
        confidence = conf[i]
        data.append([x1, y1, x2, y2, label, confidence])

    return pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'name', 'confidence'])

def process_video(frame, prev_positions=None):
    detections_df = detect_objects(frame)
    height, width = frame.shape[:2]
    third_width = width // 3

    detections = {"left": [], "center": [], "right": []}
    vehicle_classes = ["Car", "Bus", "Truck", "Motorcycle", "Bicycle"]
    crosswalk_class = "Pedestrian Crossing"

    current_positions = []
    vehicle_boxes = []

    for idx, row in detections_df.iterrows():
        x_center = (row['xmin'] + row['xmax']) / 2
        label = row['name']

        if x_center < third_width:
            detections["left"].append(label)
        elif x_center < 2 * third_width:
            detections["center"].append(label)
        else:
            detections["right"].append(label)

        if label in vehicle_classes:
            vehicle_boxes.append((row['xmin'], row['ymin'], row['xmax'], row['ymax']))

    crosswalk_present = 'name' in detections_df.columns and any(row['name'] == crosswalk_class for _, row in detections_df.iterrows())
    crosswalk_boxes = (
        detections_df[detections_df['name'] == crosswalk_class][['xmin', 'ymin', 'xmax', 'ymax']].values
        if 'name' in detections_df.columns else []
    )

    vehicle_on_crosswalk = False
    for vxmin, vymin, vxmax, vymax in vehicle_boxes:
        for cxmin, cymin, cxmax, cymax in crosswalk_boxes:
            ixmin = max(vxmin, cxmin)
            iymin = max(vymin, cymin)
            ixmax = min(vxmax, cxmax)
            iymax = min(vymax, cymax)
            iw = max(0, ixmax - ixmin)
            ih = max(0, iymax - iymin)
            if iw * ih > 0:
                vehicle_on_crosswalk = True
                break
        if vehicle_on_crosswalk:
            break

    moving_vehicle = False
    if prev_positions is not None:
        for i, (xmin, ymin, xmax, ymax) in enumerate(vehicle_boxes):
            curr_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            if i < len(prev_positions):
                prev_center = prev_positions[i]
                dist = ((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2) ** 0.5
                if dist > 5:
                    moving_vehicle = True
                    break

    current_positions = [((xmin + xmax) / 2, (ymin + ymax) / 2) for xmin, ymin, xmax, ymax in vehicle_boxes]
    is_safe = crosswalk_present and not vehicle_on_crosswalk and not moving_vehicle

    return detections, is_safe, current_positions

def speak(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_audio = fp.name
            tts.save(temp_audio)
        playsound(temp_audio)
        os.remove(temp_audio)
    except Exception as e:
        print("Speech Error:", e)

LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Kannada": "kn",
    "Tamil": "ta",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
}

SAFE_MSG = {
    "en": "Safe to cross",
    "hi": "सड़क पार करना सुरक्षित है",
    "te": "దాటడానికి సురక్షితం",
    "kn": "ಹಾದು ಹೋಗಲು ಸುರಕ್ಷಿತವಾಗಿದೆ",
    "ta": "தாண்டுவதற்கு பாதுகாப்பானது",
    "mr": "रस्ता ओलांडण्यासाठी सुरक्षित आहे",
    "bn": "পার হওয়ার জন্য নিরাপদ",
    "gu": "અન્ય પાર કરવા માટે સલામત છે",
    "ml": "കടക്കാൻ സുരക്ഷിതമാണ്",
    "pa": "ਸੁਰੱਖਿਅਤ ਹੈ ਪਾਰ ਕਰਨ ਲਈ",
    "ur": "پار کرنا محفوظ ہے"
}

UNSAFE_MSG = {
    "en": "Do not cross now",
    "hi": "अब सड़क पार न करें",
    "te": "ఇప్పుడు దాటవద్దు",
    "kn": "ಈಗ ದಾಟಬೇಡಿ",
    "ta": "இப்போது தாண்ட வேண்டாம்",
    "mr": "आता रस्ता ओलांडू नका",
    "bn": "এখন পার হবেন না",
    "gu": "હમણાં પાર ન કરો",
    "ml": "ഇപ്പോൾ കടക്കരുത്",
    "pa": "ਹੁਣੇ ਪਾਰ ਨਾ ਕਰੋ",
    "ur": "ابھی پار نہ کریں"
}
