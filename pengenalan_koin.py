import os
import cv2
import numpy as np
import csv
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Mendefinisikan Folder
image_folders = {
    '500A': 'data/500DEPAN',
    '500B': 'data/500BELAKANG',
    '1000A': 'data/1000DEPAN',
    '1000B': 'data/1000BELAKANG'
}

# Membuat output folder untuk hasil preprocessing
output_folders = {
    'grayscale': 'data/preprocessing/grayscale',
    'bilateral': 'data/preprocessing/bilateral',
    'gaussian': 'data/preprocessing/gaussian',
    'canny': 'data/preprocessing/canny',
    'morphology': 'data/preprocessing/morphology'
}

# Membuat folder jika belum ada
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

def extract_advanced_features(image_path):
    # [Preprocessing] Grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # [Preprocessing] Bilateral filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    # [Preprocessing] Gaussian blur
    blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
    # [Preprocessing] Deteksi Tepi Canny 
    canny = cv2.Canny(blurred, 50, 100)
    # [Preprocessing] Morphological operation
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    # Ekstraksi Fitur GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Ekstraksi Fitur Edge-based
    edge_count = np.sum(morph)
    edge_area = np.count_nonzero(morph)
    edge_mean = np.mean(morph)
    edge_std = np.std(morph)

    # Ekstraksi Fitur Contour-based
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

    # Simpan hasil preprocessing
    cv2.imwrite(os.path.join(output_folders['grayscale'], os.path.basename(image_path)), image)
    cv2.imwrite(os.path.join(output_folders['bilateral'], os.path.basename(image_path)), bilateral)
    cv2.imwrite(os.path.join(output_folders['gaussian'], os.path.basename(image_path)), blurred)
    cv2.imwrite(os.path.join(output_folders['canny'], os.path.basename(image_path)), canny)
    cv2.imwrite(os.path.join(output_folders['morphology'], os.path.basename(image_path)), morph)

    # Menggabungkan semua fitur untuk dikembalikan
    return {
        'image_name': os.path.basename(image_path),
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'edge_count': edge_count,
        'edge_area': edge_area,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'contour_count': contour_count,
        'total_contour_area': total_contour_area
    }

    

def process_coin_dataset():
    # File CSV
    all_features_file = 'hasil_ekstraksi_fitur_koin.csv'
    glcm_file = 'fitur_glcm.csv'
    edge_file = 'fitur_edge_based.csv'
    contour_file = 'fitur_contour_based.csv'

    # List untuk menyimpan data
    all_features_data = []
    glcm_data = []
    edge_data = []
    contour_data = []

    # Proses ekstraksi fitur
    for coin_type, folder_path in image_folders.items():
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            features = extract_advanced_features(image_path)

            if features:
                # Menyimpan semua fitur ke `hasil_ekstraksi_fitur_koin.csv`
                all_features_data.append({
                    'Coin': coin_type,
                    'Image_Name': features['image_name'],
                    'Contrast': features['contrast'],
                    'Dissimilarity': features['dissimilarity'],
                    'Homogeneity': features['homogeneity'],
                    'Edge Count': features['edge_count'],
                    'Edge Area': features['edge_area'],
                    'Edge Mean': features['edge_mean'],
                    'Edge Std': features['edge_std'],
                    'Contour Count': features['contour_count'],
                    'Total Contour Area': features['total_contour_area']
                })

                # Menyimpan fitur GLCM
                glcm_data.append({
                    'Coin': coin_type,
                    'Image_Name': features['image_name'],
                    'Contrast': features['contrast'],
                    'Dissimilarity': features['dissimilarity'],
                    'Homogeneity': features['homogeneity']
                })

                # Menyimpan fitur Edge-based
                edge_data.append({
                    'Coin': coin_type,
                    'Image_Name': features['image_name'],
                    'Edge Count': features['edge_count'],
                    'Edge Area': features['edge_area'],
                    'Edge Mean': features['edge_mean'],
                    'Edge Std': features['edge_std']
                })

                # Menyimpan fitur Contour-based
                contour_data.append({
                    'Coin': coin_type,
                    'Image_Name': features['image_name'],
                    'Contour Count': features['contour_count'],
                    'Total Contour Area': features['total_contour_area']
                })

    # Menyimpan data ke masing-masing file CSV
    pd.DataFrame(all_features_data).to_csv(all_features_file, index=False)
    pd.DataFrame(glcm_data).to_csv(glcm_file, index=False)
    pd.DataFrame(edge_data).to_csv(edge_file, index=False)
    pd.DataFrame(contour_data).to_csv(contour_file, index=False)

    print(f"Semua fitur disimpan di {all_features_file}")
    print(f"Fitur GLCM disimpan di {glcm_file}")
    print(f"Fitur Edge-based disimpan di {edge_file}")
    print(f"Fitur Contour-based disimpan di {contour_file}")

# Menjalankan proses
process_coin_dataset()
