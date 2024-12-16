import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns

class BackpropagationNeuralNetwork:
    def __init__(self, output_folder='data/classification_results'):
        # Buat folder untuk menyimpan hasil
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def load_and_prepare_data(self, file_path='coin_features.csv'):
        """
        Memuat dan mempersiapkan data untuk klasifikasi
        """
        try:
            # Baca data dari CSV
            df = pd.read_csv(file_path)

            # Pisahkan fitur dan label
            X = df[['Edge Count', 'Edge Area', 'Aspect Ratio']].values
            y = df['Coin'].values

            # Encode label kategorikal
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Normalisasi fitur
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_encoded
            )

            # Cetak distribusi kelas
            print("\nDistribusi Data Training:")
            print(pd.Series(label_encoder.inverse_transform(y_train)).value_counts())
            print("\nDistribusi Data Testing:")
            print(pd.Series(label_encoder.inverse_transform(y_test)).value_counts())

            return X_train, X_test, y_train, y_test, scaler, label_encoder

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def build_and_train_model(self, X_train, y_train):
        """
        Membuat dan melatih model Backpropagation Neural Network menggunakan scikit-learn
        """
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), 
            activation='relu', 
            solver='adam', 
            max_iter=500, 
            random_state=42
        )

        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, label_encoder):
        """
        Mengevaluasi model dan menyimpan hasil
        """
        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAkurasi Model: {test_accuracy * 100:.2f}%")

        # Konversi kembali ke label asli
        y_test_original = label_encoder.inverse_transform(y_test)
        y_pred_original = label_encoder.inverse_transform(y_pred)

        # Simpan hasil klasifikasi
        results_df = pd.DataFrame({
            'Actual': y_test_original,
            'Predicted': y_pred_original
        })
        results_df.to_csv(os.path.join(self.output_folder, 'classification_results.csv'), index=False)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.output_folder, 'confusion_matrix.png'))
        plt.close()

        # Simpan classification report
        report = classification_report(y_test_original, y_pred_original)
        with open(os.path.join(self.output_folder, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # Simpan akurasi
        with open(os.path.join(self.output_folder, 'accuracy.txt'), 'w') as f:
            f.write(f"Akurasi Model: {test_accuracy * 100:.2f}%")

        return results_df, test_accuracy * 100

    def main(self):
        """
        Fungsi utama untuk menjalankan klasifikasi
        """
        # Load dan prepare data
        data = self.load_and_prepare_data()

        if data is None:
            print("Gagal memuat data. Pastikan file CSV benar.")
            return

        X_train, X_test, y_train, y_test, scaler, label_encoder = data

        # Bangun dan latih model
        model = self.build_and_train_model(X_train, y_train)

        # Evaluasi model
        results, accuracy = self.evaluate_model(model, X_test, y_test, label_encoder)

        print("\nKlasifikasi selesai!")
        print("Hasil tersimpan di folder:", self.output_folder)
        print("\nSample hasil klasifikasi:")
        print(results.head())

# Jalankan klasifikasi
if __name__ == "__main__":
    bpnn = BackpropagationNeuralNetwork()
    bpnn.main()
