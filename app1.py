#!/usr/bin/env python3
import cv2
import face_recognition
import pickle
import paho.mqtt.client as mqtt

MQTT_BROKER = "192.168.196.155"
MQTT_PORT = 1883
MQTT_TOPIC = "led/display"

with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
print(f"✅ Data wajah berhasil dimuat: {set(data['names'])}")

video_capture = cv2.VideoCapture(0)

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matched_idx:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        if name != "Unknown":
            client.publish(MQTT_TOPIC, f"Wajah terdeteksi: {name}")
            print(f"📡 Mengirim ke MQTT: {name}")

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
