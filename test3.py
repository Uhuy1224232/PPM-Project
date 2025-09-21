import face_recognition
import cv2
import pickle

# Load data encoding
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

print("Data wajah berhasil dimuat:", set(data["names"]))

# Load gambar uji
image = cv2.imread("foto.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Deteksi wajah
boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)

for (box, enc) in zip(boxes, encodings):
    matches = face_recognition.compare_faces(data["encodings"], enc, tolerance=0.4)
    name = "Tamu"

    if True in matches:
        best_match = matches.index(True)
        name = data["names"][best_match]

    # Gambar kotak di wajah
    (top, right, bottom, left) = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
