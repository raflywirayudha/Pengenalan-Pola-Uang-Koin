import os
import cv2
import numpy as np
import csv
from skimage.feature import graycomatrix, graycoprops
from scipy import stats

# Lokasi folder gambar (sama seperti sebelumnya)
image_folders = {
    '500A': 'data/500DEPAN',
    '500B': 'data/500BELAKANG',
    '1000A': 'data/1000DEPAN',
    '1000B': 'data/1000BELAKANG'
}

# Buat folder untuk menyimpan hasil
canny_folder = 'data/canny'
os.makedirs(canny_folder, exist_ok=True)

feature_file = 'coin_features.csv'

def extract_advanced_features(image_path):
    """
    Ekstraksi fitur canggih menggunakan berbagai metode
    """
    # Baca citra
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membaca gambar pada path {image_path}")
        return None

    # Filter Bilateral untuk noise reduction
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Canny Edge Detection
    canny = cv2.Canny(bilateral, 50, 100)
    
    # Simpan citra Canny
    canny_path = os.path.join(canny_folder, os.path.basename(image_path))
    cv2.imwrite(canny_path, canny)
    
    # Fitur dari Canny
    edge_count = np.sum(canny)
    edge_area = np.count_nonzero(canny)
    
    # Fitur Geometri
    height, width = image.shape
    aspect_ratio = width / height
    
    # Matriks Ko-Okurensi Keabuan (GLCM)
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    # Properti GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    # Statistik Canny Edge
    edge_mean = np.mean(canny)
    edge_std = np.std(canny)
    edge_skewness = stats.skew(canny.flatten())
    edge_kurtosis = stats.kurtosis(canny.flatten())
    
    # Deteksi kontur
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    
    # Fitur kontur
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)
    
    return {
        'edge_count': edge_count,
        'edge_area': edge_area,
        'aspect_ratio': aspect_ratio,
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'edge_skewness': edge_skewness,
        'edge_kurtosis': edge_kurtosis,
        'contour_count': contour_count,
        'total_contour_area': total_contour_area
    }

def process_coin_dataset():
    """
    Proses seluruh dataset koin
    """
    with open(feature_file, 'w', newline='') as csvfile:
        # Perbarui fieldnames untuk mencakup semua fitur
        fieldnames = [
            'Coin', 
            'Edge Count', 
            'Edge Area', 
            'Aspect Ratio',
            'Contrast',
            'Dissimilarity', 
            'Homogeneity', 
            'Energy', 
            'Correlation', 
            'Edge Mean', 
            'Edge Std', 
            'Edge Skewness', 
            'Edge Kurtosis', 
            'Contour Count', 
            'Total Contour Area'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for coin_type, folder_path in image_folders.items():
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                features = extract_advanced_features(image_path)
                
                if features:
                    # Gabungkan fitur dengan nama koin
                    result = {
                        'Coin': coin_type,
                        'Edge Count': features['edge_count'],
                        'Edge Area': features['edge_area'],
                        'Aspect Ratio': features['aspect_ratio'],
                        'Contrast': features['contrast'],
                        'Dissimilarity': features['dissimilarity'],
                        'Homogeneity': features['homogeneity'],
                        'Energy': features['energy'],
                        'Correlation': features['correlation'],
                        'Edge Mean': features['edge_mean'],
                        'Edge Std': features['edge_std'],
                        'Edge Skewness': features['edge_skewness'],
                        'Edge Kurtosis': features['edge_kurtosis'],
                        'Contour Count': features['contour_count'],
                        'Total Contour Area': features['total_contour_area']
                    }
                    writer.writerow(result)

# Proses seluruh dataset koin
process_coin_dataset()

print(f"Fitur koin telah diekstraksi dan disimpan di {feature_file}")