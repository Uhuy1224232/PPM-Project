import face_recognition

# Load gambar
image = face_recognition.load_image_file("foto.jpg")

# Cari lokasi wajah
face_locations = face_recognition.face_locations(image)

# Ambil encoding wajah
face_encodings = face_recognition.face_encodings(image, face_locations)

# Cetak hasil
print(f"Jumlah wajah terdeteksi: {len(face_encodings)}")
for i, encoding in enumerate(face_encodings):
    print(f"\nEncoding wajah ke-{i+1}:")
    print(encoding)
