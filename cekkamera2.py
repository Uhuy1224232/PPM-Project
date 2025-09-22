import subprocess
import numpy as np

rtsp_url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"

command = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", rtsp_url,
    "-frames:v", "1",          # ambil 1 frame saja
    "-f", "image2pipe",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo", "-"
]

try:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_frame = proc.stdout.read(1920 * 1080 * 3)  # coba resolusi FullHD
    if raw_frame:
        frame = np.frombuffer(raw_frame, np.uint8)
        print("✅ RTSP stream OK, panjang data:", len(frame))
    else:
        print("❌ Tidak ada frame yang diterima")
    proc.terminate()
except Exception as e:
    print("⚠️ Error:", e)
