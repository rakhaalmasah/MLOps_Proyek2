# **Submission 2: Prediksi Indeks Nasdaq**  
**Nama: Muhammad Rakha Almasah**  
**Username dicoding: muhrakhaal**  

| **Parameter**               | **Deskripsi**                                                                                 |
|-----------------------------|----------------------------------------------------------------------------------------------|
| **Dataset**                 | Dataset ini diperoleh dari proyek di kelas **Machine Learning Terapan (Predictive Analytics)**. Dataset mencakup harga saham harian dari perusahaan-perusahaan besar dan indeks Nasdaq (IXIC). [cleaned_merged_stock_data.csv](https://raw.githubusercontent.com/rakhaalmasah/MLOps_Proyek1/9475b7b9259adff80d5b639dc54ff9c8447db4be/cleaned_merged_stock_data.csv)  |
| **Masalah**                 | Permasalahan yang diangkat adalah prediksi harga indeks Nasdaq Composite (IXIC) berdasarkan harga saham harian perusahaan besar seperti Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), dan Berkshire Hathaway (BRK_B). Model bertujuan membantu investor memahami tren harga indeks untuk pengambilan keputusan yang lebih baik. |
| **Solusi Machine Learning** | Solusi yang dibuat adalah **model regresi menggunakan TensorFlow** untuk memprediksi harga indeks Nasdaq (IXIC) berdasarkan empat fitur input harga saham: AAPL, MSFT, AMZN, dan BRK_B. Model ini dilatih menggunakan data yang telah dipreproses untuk menghasilkan prediksi yang lebih akurat. |
| **Metode Pengolahan**       | Data dipreproses menggunakan **TensorFlow Transform (TFT)** dengan metode **scale_to_z_score** untuk menormalkan data. Seluruh fitur input (AAPL, MSFT, AMZN, BRK_B) dinormalisasi agar memiliki distribusi yang seimbang. Hal ini meningkatkan performa model saat training. |
| **Arsitektur Model**        | - **Input Layer**: 4 neuron untuk fitur input (AAPL, MSFT, AMZN, BRK_B). <br> - **Hidden Layer 1**: 128 neuron dengan **ReLU activation**. <br> - **Hidden Layer 2**: 64 neuron dengan **ReLU activation**. <br> - **Output Layer**: 1 neuron untuk memprediksi harga IXIC dengan **Linear Activation**. <br> - **Optimizer**: Adam dengan learning rate default. <br> - **Loss Function**: Mean Squared Error (MSE). |
| **Metrik Evaluasi**         | Model dievaluasi menggunakan dua metrik utama: <br> - **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat error antara prediksi model dan data aktual. <br> - **Mean Absolute Error (MAE)**: Mengukur rata-rata absolut dari error prediksi. |
| **Performa Model**          | Berdasarkan hasil training dan evaluasi: <br> - **Training Loss (MSE)**: 75,271.53 <br> - **Training MAE**: 214.51 <br> - **Validation Loss (MSE)**: 137,429.84 <br> - **Validation MAE**: 281.28 <br> Model menunjukkan performa yang stabil dengan nilai loss yang menurun secara konsisten pada training dan validation set. |
| **Opsi Deployment**         | Model di-deploy menggunakan **Flask** sebagai backend. Endpoint API dikembangkan untuk melayani permintaan prediksi. Layanan kemudian di-host di **Google Cloud Run** agar dapat diakses secara publik dengan autoscaling otomatis berdasarkan jumlah permintaan. |
| **Web App**                 | Endpoint dapat diakses melalui: <br> - [CC Model Serving Metrics](https://cc-model-serving-447282078912.asia-southeast2.run.app/metrics) <br> Endpoint: `/predict` **(method: POST)** <br> - **Monitoring**: Endpoint: `/metrics` **(method: GET)** untuk menampilkan data metrik yang digunakan oleh Prometheus. |
| **Monitoring**              | Monitoring performa model dilakukan menggunakan kombinasi **Google Cloud Run**, **Prometheus**, dan **Grafana**:  <br> - **Request Count**: Lonjakan permintaan berhasil direkam dengan status **2xx** untuk endpoint `/metrics` dan `/predict`. <br> - **Request Latency**: Waktu respons menunjukkan kestabilan meskipun ada lonjakan trafik. Grafik menunjukkan nilai latensi rata-rata **di bawah 5 detik**. <br> - **Container Instances**: Terdapat **1-2 container aktif** dengan **autoscaling** untuk menangani beban kerja yang meningkat. <br> - **Grafana Dashboard**: Menampilkan metrik **http_request_created** dan **http_request_duration_seconds** yang menunjukkan lonjakan konsisten pada endpoint produksi. <br> - **Prometheus**: Visualisasi grafik **http_request_total** menunjukkan peningkatan linear dari permintaan **GET** ke `/metrics` dan **POST** ke `/predict`. |

---

### **Penjelasan Penggunaan Endpoint `/predict`**  
Endpoint `/predict` digunakan untuk mendapatkan hasil prediksi dari model. Metode HTTP yang digunakan adalah **POST**, dan data input dikirim dalam format **JSON**. Berikut adalah contoh kode Python untuk mengakses endpoint ini:

```python
import requests

FLASK_SERVER_URL = "https://cc-model-serving-447282078912.asia-southeast2.run.app/predict"

data = {
    "features": {
        "AAPL": [142.65],
        "MSFT": [342.64],
        "AMZN": [131.12],
        "BRK_B": [453.45],
        "IXIC": [0.0]
    }
}

response = requests.post(FLASK_SERVER_URL, json=data)

if response.status_code == 200:
    print("Hasil Prediksi:", response.json()["predictions"])
else:
    print("Error:", response.text)
```

### **Penjelasan Endpoint `/metrics`**  
Endpoint `/metrics` digunakan untuk **monitoring** model dengan **Prometheus**. Data metrik seperti penggunaan CPU, memori, jumlah request, dan durasi request akan ditampilkan dalam format yang dapat diproses oleh Prometheus.

---
