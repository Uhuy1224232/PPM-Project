import face_recognition
import cv2

# Load gambar
image = face_recognition.load_image_file("foto.jpg")
face_locations = face_recognition.face_locations(image)

print(f"Jumlah wajah terdeteksi: {len(face_locations)}")

# Tampilkan dengan OpenCV (opsional kalau ada GUI/forwarding X11)
img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imwrite("output.jpg", img)
