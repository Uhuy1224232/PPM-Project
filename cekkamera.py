import cv2
import subprocess
import numpy as np
import threading
import time

frame = None
stop_thread = False
lock = threading.Lock()

def capture_thread():
    global frame, stop_thread
    rtsp_url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"

    # ffmpeg command
    command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo", "-"
    ]

    # jalankan ffmpeg sebagai subprocess
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    w, h = 1920, 1080  # resolusi stream main, kalau sub biasanya 640x480
    while not stop_thread:
        raw_frame = proc.stdout.read(w * h * 3)
        if not raw_frame:
            print("⚠️ Stream berhenti, ffmpeg tidak kirim data")
            break
        f = np.frombuffer(raw_frame, np.uint8).reshape((h, w, 3))
        with lock:
            frame = f
    proc.terminate()

# ================= TESTING =================
if __name__ == "__main__":
    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    for _ in range(100):  # ambil 100 frame buat tes
        with lock:
            if frame is not None:
                print("✅ Frame diterima:", frame.shape)
        time.sleep(0.1)

    stop_thread = True
    t.join()
