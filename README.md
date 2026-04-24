# House Price Prediction Using Linear Regression

Proyek machine learning untuk memprediksi harga rumah menggunakan metode **Regresi Linear** berbasis dataset [House Price Prediction Dataset](https://www.kaggle.com/datasets/muhamedumarjamil/house-price-prediction-dataset).

---

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun model prediksi harga rumah menggunakan algoritma **Linear Regression**. Model dilatih menggunakan berbagai fitur seperti luas rumah, jumlah kamar, usia bangunan, dan jarak ke pusat kota untuk memprediksi harga jual properti.

---

## Struktur Dataset

Dataset terdiri dari **10.000 baris** dengan fitur-fitur berikut:

| Fitur | Deskripsi |
|---|---|
| `square_feet` | Luas rumah (satuan ft²) |
| `num_rooms` | Jumlah kamar |
| `age` | Usia rumah |
| `distance_to_city(km)` | Jarak rumah ke kota dalam kilometer |
| `price` | **Target variabel**: Harga rumah |

---

## Library yang Digunakan

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

---

## Alur Analisis

### Import Library dan Load Dataset
Memuat dataset dari file CSV (`house_prices_dataset.csv`) dan menampilkan 5 baris pertama untuk validasi data.

### Informasi Dataset dan Fitur
- Memeriksa struktur dan tipe data dengan `data.info()` dan `data.describe()`
- Filter fitur yang relevan: `square_feet`, `num_rooms`, `age`, `distance_to_city`
- Membuang nilai kosong (*missing values*)

### Pembagian Data Training dan Testing
```python
X = data[['square_feet']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **Data Training (80%)**: Digunakan untuk melatih model
- **Data Testing (20%)**: Digunakan untuk mengevaluasi model

### Training Model Linear Regression
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

Output parameter model:
- **Intercept (β₀)**: Nilai konstanta
- **Koefisien (β₁)**: Pengaruh luas rumah terhadap harga

### Prediksi Menggunakan Model
```python
y_pred = model.predict(X_test)
```

### Evaluasi Model (Mean Squared Error)
```python
mse = mean_squared_error(y_test, y_pred)
```

## Hasil dan Kesimpulan

- Model berhasil menemukan hubungan linear antara luas rumah dan harga
- **Intercept** menunjukkan harga dasar sebelum mempertimbangkan fitur lainnya
- **Koefisien** menunjukkan kenaikan harga per satuan luas rumah
- Evaluasi menggunakan **MSE (Mean Squared Error)** untuk mengukur akurasi prediksi
- Visualisasi regresi menunjukkan garis tren yang sesuai dengan sebaran data

---

## Cara Menjalankan

### Google Colab
1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload file notebook `.ipynb`
3. Upload dataset `house_prices_dataset.csv`
4. Jalankan semua cell secara berurutan (`Runtime → Run All`)

### Lokal
```bash
# Clone repository
git clone https://github.com/username/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Jalankan notebook
jupyter notebook house_price_prediction.ipynb
```

---

## File dalam Repository

```
house-price-prediction/
│
├── house_price_prediction.ipynb   # Notebook utama
├── house_prices_dataset.csv       # Dataset
├── README.md                      # Dokumentasi proyek
└── requirements.txt               # Daftar library
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Author

**[Qanzul Arays]**
- GitHub: [@Egyy24](https://github.com/Egyy24)

**[Muhammad Ibnu Rasyid]**
- GitHub: [@Ibnurasyid15](https://github.com/Ibnurasyid15)

---
