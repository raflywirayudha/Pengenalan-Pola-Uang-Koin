import os
import cv2
import numpy as np
import csv

# Tentukan lokasi folder gambar
image_folders = {
    '500_depan': 'data/500DEPAN',
    '500_belakang': 'data/500BELAKANG',
    '1000_depan': 'data/1000DEPAN',
    '1000_belakang': 'data/1000BELAKANG'
}

# Buat folder baru untuk menyimpan hasil Canny
canny_folder = 'data/canny'
if not os.path.exists(canny_folder):
    os.makedirs(canny_folder)

# Simpan hasil ekstraksi fitur ke dalam file CSV
feature_file = 'coin_features.csv'

def extract_canny_features(image_path):
    """
    Melakukan ekstraksi fitur menggunakan metode Canny yang lebih halus
    """
    # Baca citra
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Tidak dapat membaca gambar pada path {image_path}")
        return None, None, None
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan filter bilateral untuk mereduksi noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Terapkan algoritma Canny
    canny = cv2.Canny(bilateral, 50, 100)
    
    # Simpan citra hasil Canny
    canny_path = os.path.join(canny_folder, os.path.basename(image_path))
    cv2.imwrite(canny_path, canny)
    
    # Hitung jumlah tepi
    edge_count = np.sum(canny)
    
    # Hitung luas area tepi
    edge_area = np.count_nonzero(canny)
    
    # Hitung rasio aspek
    height, width = canny.shape
    aspect_ratio = width / height
    
    return edge_count, edge_area, aspect_ratio

def process_coin_dataset():
    """
    Proses seluruh dataset koin
    """
    with open(feature_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Coin', 'Edge Count', 'Edge Area', 'Aspect Ratio'])
        
        for coin_type, folder_path in image_folders.items():
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                edge_count, edge_area, aspect_ratio = extract_canny_features(image_path)
                if edge_count is not None:
                    writer.writerow([coin_type, edge_count, edge_area, aspect_ratio])

# Proses seluruh dataset koin
process_coin_dataset()