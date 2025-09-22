import cv2
import subprocess
import numpy as np

url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"

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

w, h = 1920, 1080  # sesuai output kameramu
frame_size = w * h * 3

while True:
    raw = proc.stdout.read(frame_size)
    if len(raw) != frame_size:
        break
    frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
    print("✅ Frame diterima:", frame.shape)
