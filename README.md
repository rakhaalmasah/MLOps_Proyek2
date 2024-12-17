# **Submission 2: Prediksi Indeks Nasdaq**  
**Nama: Muhammad Rakha Almasah**  
**Username dicoding: muhrakhaal**  

| **Parameter**               | **Deskripsi**                                                                                 |
|-----------------------------|----------------------------------------------------------------------------------------------|
| **Dataset**                 | Dataset ini diperoleh dari proyek di kelas Machine Learning Terapan (Predictive Analytics). Dataset mencakup harga saham harian dari perusahaan-perusahaan besar dan indeks Nasdaq (IXIC). [cleaned_merged_stock_data.csv](https://raw.githubusercontent.com/rakhaalmasah/MLOps_Proyek1/9475b7b9259adff80d5b639dc54ff9c8447db4be/cleaned_merged_stock_data.csv)  |
| **Masalah**                 | Permasalahan yang diangkat adalah prediksi harga indeks Nasdaq Composite (IXIC) berdasarkan harga saham harian perusahaan besar seperti Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), dan Berkshire Hathaway (BRK_B). Model bertujuan membantu investor untuk memahami tren harga indeks. |
| **Solusi Machine Learning** | Solusi yang dibuat adalah **model regresi menggunakan TensorFlow** untuk memprediksi harga IXIC berdasarkan fitur harga saham input. Model ini dilatih menggunakan data terpreproses untuk menghasilkan prediksi akurat. |
| **Metode Pengolahan**       | Data dipreproses menggunakan **TensorFlow Transform (TFT)**. Metode **scale_to_z_score** digunakan untuk menormalkan fitur AAPL, MSFT, AMZN, dan BRK_B. Proses ini membantu model memahami data yang terdistribusi dengan baik. |
| **Arsitektur Model**        | - **Input Layer**: 4 fitur input (AAPL, MSFT, AMZN, BRK_B) <br> - **Hidden Layers**: 2 Dense layers dengan **ReLU activation** dan **Dropout** untuk regularisasi. <br> - **Output Layer**: 1 unit untuk memprediksi IXIC dengan **Linear Activation**. <br> - **Optimizer**: Adam <br> - **Loss Function**: Mean Squared Error (MSE) |
| **Metrik Evaluasi**         | Model dievaluasi menggunakan dua metrik utama: <br> - **Mean Squared Error (MSE)** <br> - **Mean Absolute Error (MAE)** |
| **Performa Model**          | Berdasarkan hasil training, berikut performa model: <br> - **Training Loss**: 75,271.53 <br> - **Validation Loss (MSE)**: 109,337.27 <br> - **Validation MAE**: 243.39 <br> Model berhasil mencapai **lower bound** threshold dari MSE dan MAE pada tahap evaluasi. |
| **Opsi Deployment**         | Model dideploy menggunakan **Google Cloud Run**. Model dilayani melalui endpoint Flask yang menggunakan TensorFlow Serving. Ini memungkinkan integrasi model dengan aplikasi dan API yang lebih luas. |
| **Web App**                 | [CC Model Serving](https://cc-model-serving-447282078912.asia-southeast2.run.app) <br> Endpoint: `/predict` |
| **Monitoring**              | Proses monitoring dilakukan menggunakan **Google Cloud Run**, **Prometheus**, dan **Grafana**:  <br> - **Request Count**: Lonjakan permintaan **2xx** yang berhasil diakses pada endpoint `/metrics` dan `/predict`. <br> - **Request Latency**: Waktu respons rata-rata stabil meskipun terdapat lonjakan trafik. <br> - **Container Instances**: Jumlah container aktif **1-2 instances**, menunjukkan autoscaling berjalan sesuai trafik. <br> - **Grafana Dashboard**: Grafik menampilkan request metrics untuk endpoint produksi dengan tren meningkat secara linear. <br> - **Prometheus**: Mengumpulkan metrik `http_request_total` dengan pertumbuhan request GET dan POST yang konsisten. |

---

Jika ada revisi lebih lanjut atau tambahan penjelasan, silakan beri tahu saya!
