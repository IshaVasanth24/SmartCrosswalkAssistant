import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

model = YOLO(r"C:\Users\hp\OneDrive\Desktop\CrosswalkAssistant\yolov11s_blind_aid_best.pt")

def detect_objects(image):
    results = model(image)
    boxes = results[0].boxes
    if boxes is None:
        return pd.DataFrame()

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    class_names = model.names

    data = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        label = class_names[cls[i]]
        confidence = conf[i]
        data.append([x1, y1, x2, y2, label, confidence])
    
    return pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'name', 'confidence'])
