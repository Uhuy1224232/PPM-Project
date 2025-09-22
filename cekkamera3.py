import cv2
import subprocess
import numpy as np

url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"

def try_opencv(url):
    print("▶️ Coba dengan OpenCV + FFmpeg...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ OpenCV gagal buka stream")
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ OpenCV gagal ambil frame")
        cap.release()
        return None
    print("✅ OpenCV berhasil, frame:", frame.shape)
    return cap

def try_ffmpeg_pipe(url, w=1920, h=1080):
    print("▶️ Fallback ke FFmpeg subprocess...")
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vsync", "0",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("⚠️ FFmpeg stream habis / error")
            break
        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
        print("✅ FFmpeg frame:", frame.shape)

# --- Main ---
cap = try_opencv(url)

if cap:
    # ambil 10 frame pakai OpenCV
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame hilang, stop.")
            break
        print(f"✅ Frame {i}:", frame.shape)
    cap.release()
else:
    # kalau gagal, pakai FFmpeg pipe
    try_ffmpeg_pipe(url)
