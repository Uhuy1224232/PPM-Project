import cv2
import pickle
import time
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
import threading
import paho.mqtt.client as mqtt

# =========================
# KONFIGURASI
# =========================
RTSP_URL = "rtsp://admin:BABKQU@192.168.196.93:554/h264/ch1/main/av_stream"
LOG_FILE = "face_log.txt"
RETRY_LIMIT = 5
FPS_INTERVAL = 2.0
SIMILARITY_THRESHOLD = 0.4

MQTT_BROKER = "192.168.196.195"
MQTT_PORT = 1883
MQTT_TOPIC = "led/display"

# =========================
# LOAD DATA WAJAH
# =========================
with open("face_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_names = list(data.keys())
known_encodings = np.array(list(data.values()))
print(f"✅ Dataset wajah terload: {known_names}")

# =========================
# INIT INSIGHTFACE
# =========================
print("🔍 Memuat model InsightFace...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# =========================
# INIT MQTT
# =========================
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

def send_mqtt(message):
    client.publish(MQTT_TOPIC, message)

# =========================
# FUNGSI LOG (UTF-8 aman)
# =========================
def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

# =========================
# CLASS RTSP READER THREAD
# =========================
class RTSPReader:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# =========================
# MAIN LOOP
# =========================
reader = RTSPReader(RTSP_URL)
time.sleep(1)  # beri waktu awal

frame_count = 0
fps_start = time.time()
retry_count = 0

log_event("📡 Streaming dimulai... Tekan Ctrl+C untuk berhenti.")
try:
    while True:
        frame = reader.read()
        if frame is None:
            log_event("⚠️ Tidak bisa membaca frame dari kamera.")
            retry_count += 1
            if retry_count >= RETRY_LIMIT:
                log_event("🔄 Mencoba reconnect ke RTSP...")
                reader.stop()
                time.sleep(3)
                reader = RTSPReader(RTSP_URL)
                retry_count = 0
            continue

        retry_count = 0
        frame_count += 1

        # Hitung FPS setiap beberapa detik
        if time.time() - fps_start >= FPS_INTERVAL:
            fps = frame_count / (time.time() - fps_start)
            log_event(f"FPS: {fps:.2f}")
            frame_count = 0
            fps_start = time.time()

        # Deteksi wajah
        faces = app.get(frame)
        for face in faces:
            embedding = face.normed_embedding
            similarities = np.dot(known_encodings, embedding)
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            if best_similarity > SIMILARITY_THRESHOLD:
                name = known_names[best_idx]
                log_event(f"👤 Deteksi: {name} (similarity={best_similarity:.3f})")
                send_mqtt(f"Selamat datang {name}")
            else:
                log_event(f"👤 Wajah tidak dikenali (similarity={best_similarity:.3f})")
                send_mqtt("Selamat datang Tamu")

except KeyboardInterrupt:
    log_event("⏹ Dihentikan oleh pengguna.")
    reader.stop()
    client.loop_stop()
