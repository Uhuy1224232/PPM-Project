import cv2

url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # pakai backend ffmpeg

if not cap.isOpened():
    print("❌ Tidak bisa membuka stream")
else:
    for i in range(50):  # baca 50 frame pertama
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Gagal membaca frame ke-", i)
            break
        print(f"✅ Frame {i} terbaca: {frame.shape}")
cap.release()
