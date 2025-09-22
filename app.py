import cv2
import face_recognition
import numpy as np
import pickle
import threading
import time
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from datetime import datetime
import subprocess

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

# ================= Line detector =================
LINE_Y = 200
last_trigger_time = 0
TRIGGER_DELAY = 3
visitors_log = []  # Log pengunjung

# ================= Init MediaPipe =================
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.4
)

# ================= FFmpeg RTSP Capture =================
class FFmpegCamera:
    def __init__(self, url, w=1920, h=1080):
        self.url = url
        self.w = w
        self.h = h
        self.frame_size = w * h * 3
        self.proc = None

    def start(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vsync", "0",
            "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return None
        frame = np.frombuffer(raw, np.uint8).reshape((self.h, self.w, 3))
        return frame

    def release(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None

# ================= Thread untuk capture dari kamera (RTSP) =================
def capture_thread():
    global frame, stop_thread
    camera = FFmpegCamera("rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream")
    camera.start()

    while not stop_thread:
        f = camera.read()
        if f is not None:
            with lock:
                frame = f
        else:
            print("⚠️ Gagal ambil frame dari kamera")
            time.sleep(0.1)
    camera.release()

# ================= Thread untuk face detection + recognition =================
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

        cv2.line(f, (0, LINE_Y), (f.shape[1], LINE_Y), (0, 0, 255), 2)
        cv2.putText(f, "Face Recognition System", (10, f.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(f, f"Visitors: {len(visitors_log)}", (10, f.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        with lock:
            output_frame = f

# ================= Flask Routes =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/visitors')
def get_visitors():
    return jsonify(visitors_log[-10:])

@app.route('/api/stats')
def get_stats():
    return jsonify({
        "total_visitors": len(visitors_log),
        "unique_visitors": len(set(v["name"] for v in visitors_log)),
        "known_persons": len(set(data["names"])) if data["names"] else 0
    })

def generate_frames():
    fps = 0
    frame_count = 0
    prev_t = time.time()
    
    while True:
        with lock:
            if output_frame is None:
                continue
            f = output_frame.copy()

        frame_count += 1
        now = time.time()
        if now - prev_t >= 1.0:
            fps = frame_count / (now - prev_t)
            prev_t = now
            frame_count = 0

        cv2.putText(f, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

        ret, buffer = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================= Start Threads =================
def start_camera_threads():
    global t1, t2
    t1 = threading.Thread(target=capture_thread)
    t2 = threading.Thread(target=detection_thread)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

if __name__ == '__main__':
    start_camera_threads()
    time.sleep(2)
    print("🚀 Starting Face Recognition Web Server...")
    print("📱 Local access: http://localhost:5000")
    print("🌐 Network access: http://YOUR_IP_ADDRESS:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        stop_thread = True
        if 't1' in globals(): t1.join()
        if 't2' in globals(): t2.join()
