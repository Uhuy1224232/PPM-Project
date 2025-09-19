import cv2

def tampilkan_gambar(path_gambar, window_name="Tampilan Gambar"):
    """
    Menampilkan gambar menggunakan OpenCV.

    Args:
        path_gambar (str): Path file gambar yang ingin ditampilkan.
        window_name (str): Nama jendela OpenCV.
    """
    # Baca gambar
    img = cv2.imread(path_gambar)

    if img is None:
        print(f"❌ Gagal membuka gambar: {path_gambar}")
        return

    # Tampilkan gambar
    cv2.imshow(window_name, img)
    print("Tekan tombol apapun untuk menutup jendela...")
    cv2.waitKey(0)  # Tunggu sampai ada tombol ditekan
    cv2.destroyAllWindows()

# Contoh penggunaan
if __name__ == "__main__":
    tampilkan_gambar("contoh.jpg")  # ganti dengan path file gambar kamu
