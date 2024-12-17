# 📊 **KNN Classifier Performance Evaluation**

## 🔍 **Overview**
This project evaluates the performance of the **K-Nearest Neighbors (KNN)** algorithm with different values of *K* (3, 5, 7) on varying test sizes: **30%**, **20%**, and **10%**.

The dataset was split into training and testing sets, and the KNN algorithm's accuracy was measured for each test size configuration. The aim is to determine the best-performing model for each split.

---

## 🗂 **Data Distribution**
### 📉 **Test Size: 30%**
**Training Data Distribution:**
```
1000B    11
500B     11
500A     10
1000A    10
```
**Testing Data Distribution:**
```
500A     5
1000A    5
500B     4
1000B    4
```

### 📊 **Test Size: 20%**
**Training Data Distribution:**
```
500A     12
500B     12
1000A    12
1000B    12
```
**Testing Data Distribution:**
```
500A     3
1000B    3
1000A    3
500B     3
```

### 📈 **Test Size: 10%**
**Training Data Distribution:**
```
500B     14
1000B    14
1000A    13
500A     13
```
**Testing Data Distribution:**
```
500A     2
1000A    2
500B     1
1000B    1
```

---

## ⚙️ **Model Accuracy Results**
### 🧪 **Test Size: 30%**
| KNN (K) | Accuracy |
|---------|----------|
| K = 3   | **50.00%** |
| K = 5   | 50.00%   |
| K = 7   | 27.78%   |
**Best Model:** KNN-3 with **50.00%** accuracy.

---
### 🧪 **Test Size: 20%**
| KNN (K) | Accuracy |
|---------|----------|
| K = 3   | **58.33%** |
| K = 5   | 41.67%   |
| K = 7   | 50.00%   |
**Best Model:** KNN-3 with **58.33%** accuracy.

---
### 🧪 **Test Size: 10%**
| KNN (K) | Accuracy |
|---------|----------|
| K = 3   | **83.33%** |
| K = 5   | 66.67%   |
| K = 7   | 50.00%   |
**Best Model:** KNN-3 with **83.33%** accuracy.

---

## 🏆 **Conclusion**
From the tests conducted:
- **KNN-3** consistently outperforms other configurations across all test sizes.
- The highest accuracy of **83.33%** was achieved with **Test Size: 10%**.

| Test Size | Best KNN | Accuracy |
|-----------|----------|----------|
| 30%       | KNN-3    | 50.00%   |
| 20%       | KNN-3    | 58.33%   |
| 10%       | KNN-3    | **83.33%** |

---

## 🚀 **Key Insights**
1. As the test size decreases, the model's accuracy increases, indicating that more training data improves KNN performance.
2. **K = 3** is the optimal parameter for this dataset.

---

## 🛠 **Technologies Used**
- Python
- Scikit-Learn
- Pandas
- Jupyter Notebook

---

## 📬 **Authors**
- **Ahmad Kurniawan**  
- **Farras Latief**  
- **Muhammad Rafly Wirayudha**  
*Informatics Engineering, UIN Suska Riau*

---

## 🌟 **Future Improvements**
- Hyperparameter tuning for better results.
- Evaluation with larger datasets.
- Comparison with other machine learning models.

---

## 🔗 **License**
This project is licensed under the MIT License.
