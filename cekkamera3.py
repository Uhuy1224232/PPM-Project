import cv2

url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"

# Pakai backend FFmpeg
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Gagal buka stream")
else:
    for i in range(20):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Frame {i} gagal terbaca")
            break
        print(f"✅ Frame {i} OK, shape: {frame.shape}")

cap.release()
