import cv2

rtsp_url = "rtsp://admin:BABKQU@192.168.196.110:554/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Gagal membuka RTSP stream")
else:
    ret, frame = cap.read()
    if ret:
        print("✅ RTSP stream berhasil dibuka, ukuran frame:", frame.shape)
    else:
        print("⚠️ RTSP terbuka tapi tidak ada frame")
cap.release()
