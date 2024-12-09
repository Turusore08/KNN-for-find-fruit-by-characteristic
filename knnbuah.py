import math
from collections import Counter

def calculate_distance(point1, point2):
    
    
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def knn_predict(dataset, query, k):
    
    distances = []

    # Hitung jarak dari query ke setiap data dalam dataset
    for data in dataset:
        data_point, label = data[:-1], data[-1]
        distance = calculate_distance(data_point, query)
        distances.append((distance, label, data_point))

    # Urutkan berdasarkan jarak dan pilih k tetangga terdekat
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    # Periksa jika jarak terdekat terlalu jauh
    if k_nearest[0][0] > 50:  # Ambang batas jarak 50
        return "Tidak dapat diprediksi", k_nearest

    # Voting berdasarkan label dari k tetangga terdekat
    labels = [label for _, label, _ in k_nearest]
    predicted_label = Counter(labels).most_common(1)[0][0]

    return predicted_label, k_nearest

def main():
    # Dataset: (berat, diameter, warna, jenis buah)
    dataset = [
        (150, 7, 3, "Apel"),
        (180, 8, 3, "Apel"),
        (120, 6, 2, "Jeruk"),
        (130, 6, 2, "Jeruk"),
        (200, 10, 1, "Pisang"),
        (210, 9, 3, "Pisang"),
        (100, 5, 2, "Lemon"),
        (110, 6, 2, "Lemon"),
        (250, 11, 1, "Semangka"),
        (300, 12, 1, "Semangka"),
        (50, 4, 2, "Anggur"),
        (60, 4, 2, "Anggur"),
        (90, 5, 3, "Stroberi"),
        (100, 5, 3, "Stroberi"),
        (220, 9, 1, "Melon"),
        (230, 10, 1, "Melon"),
        (150, 7, 2, "Mangga"),
        (160, 7, 2, "Mangga"),
        (80, 6, 3, "Ceri"),
        (85, 6, 3, "Ceri"),
        (240, 12, 2, "Nanas"),
        (250, 13, 2, "Nanas"),
        (170, 8, 3, "Apel"),
        (140, 7, 2, "Apel"),
        (190, 9, 1, "Pisang"),
        (115, 6, 2, "Jeruk"),
        (260, 11, 1, "Semangka"),
        (95, 5, 3, "Stroberi"),
        (70, 4, 2, "Anggur"),
        (240, 10, 1, "Melon"),
        (160, 7, 2, "Mangga"),
        (105, 5, 2, "Lemon"),
        (215, 9, 3, "Pisang"),
        (265, 12, 1, "Semangka"),
        (150, 7, 3, "Apel"),
        (155, 7, 3, "Apel"),
        (135, 6, 2, "Jeruk"),
        (250, 12, 2, "Nanas"),
        (245, 11, 1, "Semangka"),
        (85, 5, 3, "Ceri"),
        (260, 13, 2, "Nanas"),
        (155, 7, 3, "Mangga"),
        (95, 4, 2, "Lemon"),
    ]

    # Input dari pengguna
    try:
        berat = float(input("Masukkan berat buah (gram): "))
        diameter = float(input("Masukkan diameter buah (cm): "))
        warna = int(input("Masukkan kode warna buah (1: Hijau, 2: Kuning, 3: Merah): "))
        query = (berat, diameter, warna)
    except ValueError:
        print("Input tidak valid. Pastikan memasukkan angka yang sesuai.")
        return

    k = 3  # Jumlah tetangga terdekat

    # Prediksi jenis buah
    result, k_nearest = knn_predict(dataset, query, k)

    print("\nDataset:")
    for data in dataset:
        print(f"{data[:-1]} -> {data[-1]}")

    print("\nData yang diprediksi:", query)
    if result == "Tidak dapat diprediksi":
        print("Jenis buah tidak dapat diprediksi karena terlalu jauh dari dataset.")
    else:
        print(f"Jenis buah yang diprediksi: {result}")

    print("\nTetangga terdekat:")
    for i, (distance, label, data_point) in enumerate(k_nearest, start=1):
        print(f"{i}. Data: {data_point}, Label: {label}, Jarak: {distance:.2f}")

if __name__ == "__main__":
    main()
