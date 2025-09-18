import cv2
import face_recognition
import numpy as np
import pickle
import threading
import time
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from datetime import datetime
import paho.mqtt.client as mqtt   # MQTT

# --- MQTT Setup ---
MQTT_BROKER = "192.168.196.155"   # ganti dengan IP Raspberry Pi (broker Mosquitto)
MQTT_PORT = 1883
MQTT_TOPIC = "led/display"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

app = Flask(__name__)

# ================= Load database wajah =================
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
    print(f"✅ Data wajah berhasil dimuat: {set(data['names'])}")
except FileNotFoundError:
    print("⚠️ File encodings.pickle tidak ditemukan!")
    data = {"encodings": [], "names": []}

TOLERANCE = 0.40
PROCESS_EVERY_N_FRAMES = 2
DOWNSCALE = 0.6

frame = None
output_frame = None
lock = threading.Lock()
stop_thread = False

LINE_Y = 200
last_trigger_time = 0
TRIGGER_DELAY = 3
visitors_log = []

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.4
)

def detection_thread():
    global frame, output_frame, stop_thread, last_trigger_time, visitors_log
    frame_count = 0
    
    while not stop_thread:
        with lock:
            if frame is None:
                continue
            f = frame.copy()

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        small_frame = cv2.resize(f, None, fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb_small)

        if results.detections:
            for det in results.detections:
                bboxC = det.location_data.relative_bounding_box
                h, w, _ = small_frame.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = x1 + int(bboxC.width * w)
                y2 = y1 + int(bboxC.height * h)
                x1, y1, x2, y2 = [int(val / DOWNSCALE) for val in (x1, y1, x2, y2)]

                face_crop = f[y1:y2, x1:x2]
                name = "Tamu"

                if face_crop.size > 0:
                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb_face)
                    if encs:
                        enc = encs[0]
                        matches = face_recognition.compare_faces(data["encodings"], enc, tolerance=TOLERANCE)
                        face_distances = face_recognition.face_distance(data["encodings"], enc)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = data["names"][best_match_index]

                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(f, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if y2 >= LINE_Y:
                    now = time.time()
                    if now - last_trigger_time > TRIGGER_DELAY:
                        print(f"Selamat Datang {name}")
                        visitors_log.append({
                            "name": name,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        last_trigger_time = now

                        # --- Kirim ke LED Matrix lewat MQTT ---
                        mqtt_message = f"Selamat Datang {name}"
                        mqtt_client.publish(MQTT_TOPIC, mqtt_message)
                        print(f"📡 MQTT Publish: {mqtt_message}")

        cv2.line(f, (0, LINE_Y), (f.shape[1], LINE_Y), (0, 0, 255), 2)
        cv2.putText(f, "Face Recognition System", (10, f.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(f, f"Visitors: {len(visitors_log)}", (10, f.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        with lock:
            output_frame = f
