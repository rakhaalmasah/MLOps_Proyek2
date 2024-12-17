# Submission 1: Prediksi Indeks Nasdaq

**Nama:** Muhammad Rakha Almasah  
**Username Dicoding:** muhrakhaal  

## **Deskripsi Proyek**  
Proyek ini bertujuan untuk memprediksi nilai **Indeks Nasdaq (IXIC)** berdasarkan data harga saham dari beberapa perusahaan besar seperti **Apple (AAPL)**, **Microsoft (MSFT)**, **Amazon (AMZN)**, dan **Berkshire Hathaway (BRK_B)**. Pipeline ini dibangun menggunakan **TensorFlow Extended (TFX)** dan hasil model diterapkan melalui **Flask API** yang diakses menggunakan **Ngrok**.

---

| **Kategori**         | **Deskripsi**                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| **Dataset**           | [cleaned_merged_stock_data.csv](https://raw.githubusercontent.com/rakhaalmasah/MLOps_Proyek1/9475b7b9259adff80d5b639dc54ff9c8447db4be/cleaned_merged_stock_data.csv) <br> Dataset ini diperoleh dari proyek di kelas **Machine Learning Terapan (Predictive Analytics)**. Dataset mencakup harga saham harian dari perusahaan-perusahaan besar dan indeks Nasdaq (IXIC).   |
| **Masalah**           | Permasalahan utama adalah **memprediksi nilai indeks Nasdaq (IXIC)** berdasarkan harga saham dari perusahaan terkemuka. Ini merupakan masalah **regresi** di mana nilai indeks diprediksi sebagai nilai numerik berkelanjutan.      |
| **Solusi Machine Learning** | Solusi yang dibuat melibatkan pengembangan pipeline **Machine Learning** end-to-end menggunakan **TFX**. Pipeline ini mencakup proses pengolahan data, pelatihan model dengan **Keras Tuner**, evaluasi model menggunakan **TFMA**, dan penerapan model melalui **Flask API**. Target akhir adalah meminimalkan error prediksi dan membuat model yang siap digunakan dalam produksi.   |
| **Metode Pengolahan** | 1. **Pembagian Data**: Data dibagi menjadi train (80%) dan eval (20%) menggunakan `CsvExampleGen`. <br> 2. **Statistik Data**: Menggunakan `StatisticsGen` untuk memahami distribusi data. <br> 3. **Validasi Skema**: `SchemaGen` digunakan untuk membuat skema dan memvalidasi anomali data menggunakan `ExampleValidator`. <br> 4. **Transformasi Data**: Normalisasi fitur dilakukan dengan **Z-score scaling** menggunakan **TensorFlow Transform (TFT)**. <br> 5. **Feature Engineering**: Fitur masukan **AAPL**, **MSFT**, **AMZN**, dan **BRK_B** digunakan, sedangkan **IXIC** menjadi target prediksi.   |
| **Arsitektur Model**  | Model dibangun menggunakan **Neural Network** sederhana dengan arsitektur sebagai berikut: <br> - **Input Layer**: Fitur numerik dengan bentuk `(batch_size, 1)`. <br> - **Hidden Layers**: <br>    - Hidden Layer 1: 192 neuron dengan **ReLU activation** (hasil tuning terbaik). <br>    - Hidden Layer 2: 128 neuron dengan **ReLU activation**. <br> - **Output Layer**: 1 neuron untuk memprediksi nilai kontinu (IXIC). <br> Model menggunakan **optimizer Adam** dan **loss function MSE**. |
| **Metrik Evaluasi**   | - **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat error. <br> - **Mean Absolute Error (MAE)**: Mengukur rata-rata absolut error. <br> Target yang ditetapkan adalah **MSE < 500.0** dan **MAE < 50.0**. |
| **Performa Model**    | Berdasarkan hasil evaluasi: <br> - **MAE**: 238.13 <br> - **MSE**: 105652.79 <br> Performa model sudah cukup baik namun dapat ditingkatkan dengan optimasi lebih lanjut. Model ini diuji menggunakan **Flask API** dan memberikan prediksi yang stabil. |

---

## **Tahapan Pipeline**
1. **Pengolahan Data**:  
   - Data diproses menggunakan **TFX** dengan komponen **CsvExampleGen**, **StatisticsGen**, **SchemaGen**, dan **Transform**.  
   - Transformasi data melibatkan **Z-score normalization** pada fitur-fitur numerik.

2. **Pengembangan Model**:  
   - Hyperparameter Tuning dilakukan menggunakan **Keras Tuner** dengan metode **Random Search**.  
   - Arsitektur model terbaik memiliki **192** neuron di hidden layer pertama dan **128** neuron di hidden layer kedua.

3. **Evaluasi Model**:  
   - Evaluasi dilakukan menggunakan **TFMA** dengan metrik **MAE** dan **MSE**.  
   - Hasil evaluasi:  
     - **MAE**: 238.13  
     - **MSE**: 105652.79  

4. **Deployment**:  
   - Model diterapkan menggunakan **Flask API** dan diakses melalui **Ngrok** untuk menyediakan endpoint publik.  

5. **Testing Model**:  
   - Model diuji dengan POST request ke endpoint Flask dan memberikan prediksi sebagai berikut:  

### Contoh Request:  
```json
POST /predict
{
  "features": {
    "AAPL": [111.28],
    "MSFT": [336.32],
    "AMZN": [138.12],
    "BRK_B": [553.50],
    "IXIC": [0.0]
  }
}
```

### Contoh Response:  
```json
{
  "predictions": [
    1210132.75
  ]
}
```

---

## **Hasil Training dan Evaluasi**
### Hasil Tuning Hyperparameter:  
- **Best Hyperparameters**:  
  - **units_1**: 192  
  - **units_2**: 128  
- **Best Validation MAE**: 5045.67  

### Hasil Evaluasi Akhir:  
- **Mean Absolute Error (MAE)**: 238.13  
- **Mean Squared Error (MSE)**: 105652.79  

---

## **Kesimpulan**
Model berhasil dikembangkan dengan pipeline **TFX** yang lengkap, mulai dari pre-processing data, pelatihan model, tuning hyperparameter, evaluasi, hingga deployment.  
Performa model menunjukkan **MAE 238.13** dan **MSE 105652.79**, yang dapat ditingkatkan lebih lanjut melalui optimasi data dan arsitektur model.  
Model ini telah berhasil di-deploy menggunakan **Flask API** dan diuji menggunakan **Ngrok** untuk memastikan fungsionalitasnya dalam lingkungan produksi.

---
