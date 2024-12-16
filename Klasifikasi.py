import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Tambahkan library untuk Neural Networks
from sklearn.neural_network import MLPClassifier  # BPNN
from minisom import MiniSom  # LVQ (menggunakan MiniSom sebagai implementasi serupa)

# Buat folder untuk menyimpan hasil visualisasi
output_folder = 'data/classification_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def load_and_prepare_data():
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
        test_size=0.3,          # 20% untuk testing
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
    
    return X_train, X_test, y_train, y_test, X_scaled, y_encoded, scaler, label_encoder

def custom_lvq(X_train, y_train, X_test, learning_rate=0.1, iterations=100):
    """
    Implementasi sederhana Learning Vector Quantization (LVQ)
    """
    # Inisialisasi prototype vectors (mengambil sampel dari setiap kelas)
    unique_classes = np.unique(y_train)
    prototypes = []
    prototype_labels = []
    
    for cls in unique_classes:
        # Ambil sampel dari kelas tersebut
        class_samples = X_train[y_train == cls]
        # Pilih prototype sebagai centroid
        prototype = np.mean(class_samples, axis=0)
        prototypes.append(prototype)
        prototype_labels.append(cls)
    
    prototypes = np.array(prototypes)
    prototype_labels = np.array(prototype_labels)
    
    # Proses training
    for _ in range(iterations):
        for i, sample in enumerate(X_train):
            # Cari prototype terdekat
            distances = np.linalg.norm(prototypes - sample, axis=1)
            closest_prototype_idx = np.argmin(distances)
            
            # Update prototype
            if prototype_labels[closest_prototype_idx] == y_train[i]:
                # Gerakkan prototype mendekati sampel
                prototypes[closest_prototype_idx] += learning_rate * (sample - prototypes[closest_prototype_idx])
            else:
                # Gerakkan prototype menjauhi sampel
                prototypes[closest_prototype_idx] -= learning_rate * (sample - prototypes[closest_prototype_idx])
    
    # Prediksi
    predictions = []
    for sample in X_test:
        distances = np.linalg.norm(prototypes - sample, axis=1)
        closest_prototype_idx = np.argmin(distances)
        predictions.append(prototype_labels[closest_prototype_idx])
    
    return np.array(predictions)

def evaluate_model(model_name, y_test, y_pred, label_encoder):
    # Transform kembali ke label asli untuk hasil
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Simpan hasil klasifikasi
    results_df = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_pred_original
    })
    results_df.to_csv(f'data/classification_results/{model_name}_results.csv', index=False)
    
    # Buat confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'data/classification_results/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100
    
    print(f"\n{model_name} Results:")
    print(f"Akurasi Model: {accuracy_percentage:.2f}%")
    
    # Simpan laporan klasifikasi
    report = classification_report(y_test_original, y_pred_original, zero_division=1)
    with open(f'data/classification_results/{model_name}_classification_report.txt', 'w') as f:
        f.write(f"Akurasi Model: {accuracy_percentage:.2f}%\n\n")
        f.write(report)
    
    return accuracy_percentage, results_df

def main():
    # Load dan prepare data
    X_train, X_test, y_train, y_test, X_scaled, y_encoded, scaler, label_encoder = load_and_prepare_data()
    
    # Model-model yang akan dievaluasi
    models = [
        ('KNN', KNeighborsClassifier(n_neighbors=3)),
        ('BPNN', MLPClassifier(
            hidden_layer_sizes=(10, 5),  # 2 hidden layers
            max_iter=2000, 
            random_state=42
        ))
    ]
    
    # Evaluasi KNN dan BPNN
    results = {}
    for name, model in models:
        # Latih model
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Evaluasi
        accuracy, result_df = evaluate_model(name, y_test, y_pred, label_encoder)
        results[name] = {'accuracy': accuracy, 'results': result_df}
    
    # Evaluasi LVQ
    y_pred_lvq = custom_lvq(X_train, y_train, X_test)
    accuracy_lvq, result_df_lvq = evaluate_model('LVQ', y_test, y_pred_lvq, label_encoder)
    results['LVQ'] = {'accuracy': accuracy_lvq, 'results': result_df_lvq}
    
    # Temukan model terbaik
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nModel terbaik: {best_model} dengan akurasi {results[best_model]['accuracy']:.2f}%")
    
    print("\nKlasifikasi selesai! Hasil tersimpan di folder 'data/classification_results'")

if __name__ == "__main__":
    main()