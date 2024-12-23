import os
import cv2
import numpy as np
import csv
from skimage.feature import graycomatrix, graycoprops
from scipy import stats

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

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

def extract_advanced_features(image_path):
  
    # [Preprocessing] grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot read image at {image_path}")
        return None
    
    grayscale_path = os.path.join(output_folders['grayscale'], os.path.basename(image_path))
    cv2.imwrite(grayscale_path, image)
    
    # [Preprocessing] Bilateral filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    bilateral_path = os.path.join(output_folders['bilateral'], os.path.basename(image_path))
    cv2.imwrite(bilateral_path, bilateral)
    
    # [Preprocessing] Gaussian blur
    blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
    gaussian_path = os.path.join(output_folders['gaussian'], os.path.basename(image_path))
    cv2.imwrite(gaussian_path, blurred)
    
    # [Preprocessing] Deteksi Tepi Canny 
    canny = cv2.Canny(blurred, 50, 100)
    canny_path = os.path.join(output_folders['canny'], os.path.basename(image_path))
    cv2.imwrite(canny_path, canny)
    
    # [Preprocessing] Morphological operation = diterapkan pada hasil Canny edge detection untuk memperbaiki hasil deteksi tepi dengan menghubungkan garis-garis yang terputus
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    morph_path = os.path.join(output_folders['morphology'], os.path.basename(image_path))
    cv2.imwrite(morph_path, morph)
    
    # Ekstraksi Fitur

    # Fitur dari Canny
    edge_count = np.sum(morph)
    edge_area = np.count_nonzero(morph)
    
    # Matriks Ko-Okurensi Keabuan (GLCM) untuk analisis tekstur
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    # Properti GLCM yang relevan untuk tekstur koin
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Statistik Canny Edge
    edge_mean = np.mean(morph)
    edge_std = np.std(morph)

    # Deteksi kontur untuk analisis bentuk
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)
    
    return {
        'edge_count': edge_count,
        'edge_area': edge_area,
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'contour_count': contour_count,
        'total_contour_area': total_contour_area
    }

feature_file = 'hasil_ekstraksi_fitur_koin.csv'

def process_coin_dataset():
    with open(feature_file, 'w', newline='') as csvfile:
        fieldnames = [
            'Coin', 
            'Edge Count', 
            'Edge Area', 
            'Contrast',
            'Dissimilarity', 
            'Homogeneity',
            'Edge Mean', 
            'Edge Std', 
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
                    result = {
                        'Coin': coin_type,
                        'Edge Count': features['edge_count'],
                        'Edge Area': features['edge_area'],
                        'Contrast': features['contrast'],
                        'Dissimilarity': features['dissimilarity'],
                        'Homogeneity': features['homogeneity'],
                        'Edge Mean': features['edge_mean'],
                        'Edge Std': features['edge_std'],
                        'Contour Count': features['contour_count'],
                        'Total Contour Area': features['total_contour_area']
                    }
                    writer.writerow(result)

process_coin_dataset()
print(f"Fitur yang telah diekstraksi disimpan di {feature_file}")