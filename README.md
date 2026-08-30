# Prediksi Penjualan Mobil Bekas

Aplikasi ini berfungsi untuk memprediksi harga jual mobil bekas berdasarkan spesifikasi kendaraan menggunakan model machine learning berbasis XGBoost. Aplikasi dibuat dengan Streamlit dan dilengkapi dengan halaman analisis data, evaluasi model, serta fitur prediksi harga secara real-time.

## Fitur Utama

- Prediksi harga mobil bekas berdasarkan input pengguna
- Pemilihan merek, model, tahun, bahan bakar, transmisi, dan spesifikasi teknis
- Menampilkan estimasi harga beserta rentang nilai wajar sekitar +/- 10%
- Evaluasi performa model menggunakan metrik seperti R2, MAE, RMSE, dan MAPE
- Visualisasi performa model dan distribusi data
- Eksplorasi dataset historis mobil bekas

## Teknologi yang Digunakan

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

## Struktur Project

```text
Prediksi Penjualan Mobil Bekas/
├── app.py                          # Aplikasi Streamlit utama
├── cleaned_used_car_dataset.csv    # Dataset hasil pembersihan dan preprocessing
├── requirements.txt                # Daftar dependency Python
├── pipeline_prediksi_mobil_bekas.ipynb  # Notebook untuk pipeline data dan training model
├── model/
│   ├── xgb_best_model.pkl         # Model XGBoost terbaik
│   ├── preprocessor.pkl           # Preprocessor untuk fitur kategorikal dan numerik
│   ├── model_metadata.json        # Metadata model dan metric evaluasi
│   ├── feature_importance.csv     # Data pentingnya fitur
│   ├── test_predictions.pkl       # Hasil prediksi data uji untuk evaluasi
│   └── ...
└── README.md
```

## Persyaratan Sistem

Pastikan Anda sudah memiliki Python 3.9+ dan pip terinstal di komputer Anda.

## Cara Menjalankan Aplikasi

1. Buka terminal atau command prompt
2. Masuk ke direktori project
3. Install dependency:

```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi:

```bash
streamlit run app.py
```

5. Buka browser sesuai URL yang ditampilkan, biasanya:

```text
http://localhost:8501
```

## Deskripsi Model

Model yang digunakan adalah XGBoost Regressor untuk memprediksi target `Selling_Price_IDR`.

Beberapa metadata model yang dihasilkan meliputi:

- R2 Score: 0.9451
- MAE: Rp 11.800.164,91
- RMSE: Rp 18.565.647,04
- MAPE: 9.79%

## Halaman pada Aplikasi

### 1. Beranda
Menampilkan ringkasan performa model dan dataset.

### 2. Prediksi Harga
User dapat memasukkan spesifikasi kendaraan dan mendapatkan estimasi harga pasar.

### 3. Evaluasi Model
Menampilkan metrik evaluasi, plot aktual vs prediksi, residual, dan feature importance.

### 4. Tentang Dataset
Menampilkan struktur dataset, statistik deskriptif, serta beberapa visualisasi data.

## Catatan

- File model berada di folder `model/` dan harus tetap tersedia agar aplikasi dapat berjalan.
- Jika file model tidak ditemukan, pastikan proses training atau pipeline data telah selesai dijalankan terlebih dahulu.
- Aplikasi ini dibuat untuk kebutuhan analisis dan estimasi harga mobil bekas di pasar Indonesia dengan format dataset lokal.

## Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan pengembangan aplikasi prediksi harga kendaraan bekas.
