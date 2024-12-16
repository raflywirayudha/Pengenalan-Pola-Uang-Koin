import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Buat folder untuk menyimpan hasil visualisasi
output_folder = 'data/classification_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Baca data dari CSV
def load_and_prepare_data():
    df = pd.read_csv('coin_features.csv')
    
    # Pisahkan fitur dan label
    X = df[['Edge Count', 'Edge Area', 'Aspect Ratio']].values
    y = df['Coin'].values
    
    # Encode label kategorikal menjadi numerik
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified split dengan mempertahankan proporsi kelas
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.3,          # 30% untuk testing
        random_state=42,        # untuk reproducibility
        stratify=y_encoded      # memastikan proporsi kelas seimbang
    )
    
    # Verifikasi jumlah sampel per kelas
    train_distribution = pd.Series(label_encoder.inverse_transform(y_train)).value_counts()
    test_distribution = pd.Series(label_encoder.inverse_transform(y_test)).value_counts()
    
    print("\nDistribusi data training:")
    print(train_distribution)
    print("\nDistribusi data testing:")
    print(test_distribution)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, label_encoder):
    # Inisialisasi dan latih model KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Prediksi
    y_pred = knn.predict(X_test)
    
    # Transform kembali ke label asli untuk hasil
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Simpan hasil klasifikasi
    results_df = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_pred_original
    })
    results_df.to_csv('data/classification_results/classification_results.csv', index=False)
    
    # Buat confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('data/classification_results/confusion_matrix.png')
    plt.close()
    
    # Visualisasi hasil klasifikasi untuk 2 fitur pertama
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
    plt.title('Hasil Klasifikasi KNN')
    plt.xlabel('Edge Count (Normalized)')
    plt.ylabel('Edge Area (Normalized)')
    plt.colorbar(scatter, label='Kelas Koin')
    plt.savefig('data/classification_results/classification_visualization.png')
    plt.close()
    
    # Simpan report klasifikasi
    report = classification_report(y_test_original, y_pred_original)
    with open('data/classification_results/classification_report.txt', 'w') as f:
        f.write(report)

     # Tambahkan perhitungan akurasi
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100
    
    print(f"\nAkurasi Model: {accuracy_percentage:.2f}%")
    
    # Simpan akurasi ke dalam file txt
    with open('data/classification_results/accuracy.txt', 'w') as f:
        f.write(f"Akurasi Model: {accuracy_percentage:.2f}%")
    
    return knn, results_df, accuracy_percentage

def main():
    # Load dan prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_prepare_data()
    
    # Train dan evaluasi model
    model, results, accuracy = train_and_evaluate_knn(X_train, X_test, y_train, y_test, label_encoder)
    
    print("Klasifikasi selesai! Hasil tersimpan di folder 'data/classification_results'")
    print("\nSample hasil klasifikasi:")
    print(results.head())
    print(f"\nAkurasi Model: {accuracy:.2f}%")

if __name__ == "__main__":
    main()