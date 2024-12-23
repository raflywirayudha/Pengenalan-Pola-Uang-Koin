# 🔊 **KNN Classifier Performance Evaluation**

## 🔍 **Overview**
This project evaluates the performance of the **K-Nearest Neighbors (KNN)** algorithm with **K=3** on varying test sizes: **30%**, **20%**, and **10%**.

The dataset was split into training and testing sets, and the KNN algorithm's accuracy was measured for each test size configuration. The aim is to determine the best-performing model for each split.

---

## 🔂 **Data Distribution**
### 🔁 **Test Size: 30%**
**Training Data Distribution:**
```
1000A    7
500A     7
500B     7
1000B    7
```
**Testing Data Distribution:**
```
500A     3
500B     3
1000B    3
1000A    3
```

### 🔀 **Test Size: 20%**
**Training Data Distribution:**
```
1000B    8
1000A    8
500A     8
500B     8
```
**Testing Data Distribution:**
```
500A     2
500B     2
1000A    2
1000B    2
```

### 📈 **Test Size: 10%**
**Training Data Distribution:**
```
500A     9
500B     9
1000B    9
1000A    9
```
**Testing Data Distribution:**
```
1000A    1
500B     1
1000B    1
500A     1
```

---

## ⚙️ **Model Accuracy Results**
### 🪪 **Test Size: 30%**
| Test Size | Accuracy |
|-----------|----------|
| 30%       | **50.00%** |

### 🪪 **Test Size: 20%**
| Test Size | Accuracy |
|-----------|----------|
| 20%       | **50.00%** |

### 🪪 **Test Size: 10%**
| Test Size | Accuracy |
|-----------|----------|
| 10%       | **75.00%** |

---

## 🏆 **Conclusion**
From the tests conducted:
- **KNN-3** consistently delivers stable performance across all test sizes.
- The highest accuracy of **75.00%** was achieved with **Test Size: 10%**.

| Test Size | Best KNN | Accuracy |
|-----------|----------|----------|
| 30%       | KNN-3    | 50.00%   |
| 20%       | KNN-3    | 50.00%   |
| 10%       | KNN-3    | **75.00%** |

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

## 📨 **Authors**
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

