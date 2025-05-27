# Stock Price Prediction App

Aplikasi web prediksi harga saham Indonesia menggunakan Artificial Neural Network (ANN) dengan Backpropagation. Dibangun dengan Python, Flask, dan TensorFlow.

## Struktur Folder

```
TUGAS7/
├── app/               # Kode aplikasi Flask
│   ├── __init__.py
│   ├── routes.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── static/
│   └── templates/
├── data/              # Dataset & model
│   ├── data_saham.csv
│   └── model_saham.h5
├── notebooks/         # Notebook eksplorasi
├── app.py             # Entry point Flask
├── train_model.py     # Script training
├── requirements.txt   # Dependensi
├── Procfile           # Untuk deployment
└── README.md          # Dokumentasi
```

## Cara Menjalankan

1. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```
2. **Training Model**
   ```bash
   python train_model.py
   ```
3. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

## Deployment
- Siapkan file `Procfile` dan `requirements.txt`.
- Deploy ke Railway/Heroku sesuai instruksi platform.

---

Silakan edit dan lengkapi README ini sesuai kebutuhan proyekmu.
