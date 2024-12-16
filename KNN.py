import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Baca dataset dari file CSV
feature_file = 'coin_features.csv'
data = pd.read_csv(feature_file)

# Pisahkan fitur (X) dan label (y)
X = data[['Edge Count', 'Edge Area', 'Aspect Ratio']]
y = data['Coin']

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN dengan jumlah tetangga (k)
k = 4  # Anda dapat mencoba nilai k yang berbeda
knn = KNeighborsClassifier(n_neighbors=k)

# Latih model dengan data latih
knn.fit(X_train, y_train)

# Prediksi data uji
y_pred = knn.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi KNN dengan k={k}: {accuracy * 100:.2f}%")
