import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Buat folder untuk menyimpan hasil visualisasi
output_folder = 'data/hasil_klasifikasi'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def load_and_prepare_data(test_size):
    """
    Load data dan persiapkan dengan rasio train-test yang berbeda
    """
    # Baca data dari CSV
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
        test_size=test_size,         # Rasio yang akan diuji
        random_state=42,             # untuk reproducibility
        stratify=y_encoded           # memastikan proporsi kelas seimbang
    )
    
    # Verifikasi jumlah sampel per kelas
    train_distribution = pd.Series(label_encoder.inverse_transform(y_train)).value_counts()
    test_distribution = pd.Series(label_encoder.inverse_transform(y_test)).value_counts()
    
    print(f"\nDistribusi data training (test_size={test_size}):")
    print(train_distribution)
    print(f"\nDistribusi data testing (test_size={test_size}):")
    print(test_distribution)
    
    return X_train, X_test, y_train, y_test, X_scaled, y_encoded, scaler, label_encoder

def evaluate_model(model_name, y_test, y_pred, label_encoder, test_size):
    """
    Evaluasi model dengan menyimpan hasil dan visualisasi
    """
    # Buat subfolder untuk setiap rasio split
    split_folder = os.path.join(output_folder, f'datauji_{int(test_size*100)}')
    os.makedirs(split_folder, exist_ok=True)
    
    # Transform kembali ke label asli untuk hasil
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Simpan hasil klasifikasi
    results_df = pd.DataFrame({
        'Asli': y_test_original,
        'Prediksi': y_pred_original
    })
    results_df.to_csv(os.path.join(split_folder, f'{model_name}_results.csv'), index=False)
    
    # Buat confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name} (Data Uji: {int(test_size*100)}%)')
    plt.xlabel('Prediksi')
    plt.ylabel('Asli')
    plt.tight_layout()
    plt.savefig(os.path.join(split_folder, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100
    
    print(f"\n{model_name} Hasil (Data Uji: {int(test_size*100)}%):")
    print(f"Akurasi Model: {accuracy_percentage:.2f}%")
    
    # Simpan laporan klasifikasi
    report = classification_report(y_test_original, y_pred_original, zero_division=1)
    with open(os.path.join(split_folder, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(f"Akurasi Model: {accuracy_percentage:.2f}%\n\n")
        f.write(report)
    
    return accuracy_percentage, results_df

def main():
    # Rasio split yang akan diuji
    test_sizes = [0.3, 0.2, 0.1]
    
    # Simpan hasil untuk semua percobaan
    all_results = {}
    
    for test_size in test_sizes:
        print(f"\n--- Pengujian dengan Test Size {int(test_size*100)}% ---")
        
        # Load dan prepare data
        X_train, X_test, y_train, y_test, X_scaled, y_encoded, scaler, label_encoder = load_and_prepare_data(test_size)
        
        # Gunakan KNN dengan berbagai parameter
        knn_params = [
            ('KNN-3', KNeighborsClassifier(n_neighbors=3)),
            ('KNN-5', KNeighborsClassifier(n_neighbors=5)),
            ('KNN-7', KNeighborsClassifier(n_neighbors=7))
        ]
        
        # Evaluasi KNN
        results = {}
        for name, model in knn_params:
            # Latih model
            model.fit(X_train, y_train)
            
            # Prediksi
            y_pred = model.predict(X_test)
            
            # Evaluasi
            accuracy, result_df = evaluate_model(name, y_test, y_pred, label_encoder, test_size)
            results[name] = {'Akurasi': accuracy, 'Hasil': result_df}
        
        # Temukan model terbaik untuk split ini
        best_model = max(results, key=lambda x: results[x]['Akurasi'])
        print(f"\nModel terbaik untuk split {int(test_size*100)}%: {best_model} dengan akurasi {results[best_model]['Akurasi']:.2f}%")
        
        # Simpan hasil untuk keseluruhan pengujian
        all_results[f'datauji_{int(test_size*100)}'] = results
    
    # Buat ringkasan hasil
    summary_results = []
    for split, models in all_results.items():
        for model, data in models.items():
            summary_results.append({
                'Split': split,
                'Model': model,
                'Akurasi': data['Akurasi']
            })
    
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(output_folder, 'rangkuman_akurasi.csv'), index=False)
    
    print("\nKlasifikasi selesai! Hasil tersimpan di folder 'data/hasil_klasifikasi'")

if __name__ == "__main__":
    main()