import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import joblib
from aplikasi.preprocessing import preprocess_and_split, preprocess_stock_data
import logging
import sys
import time

# Setup logging ke file dan terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

start_time = time.time()
logging.info('=== TRAINING STARTED ===')

# 1. Load dataset
csv_path = 'data/DaftarSaham.csv'
logging.info(f'Membaca dataset dari {csv_path}')
df = pd.read_csv(csv_path)

# 2. Preprocessing & split (otomatis 70/15/15, target: LastPrice)
logging.info('Melakukan preprocessing dan split data (train/val/test) ...')
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_and_split(df, test_size=0.15, val_size=0.15, random_state=42)
logging.info(f'Jumlah data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')

# Simpan preprocessor untuk digunakan di Flask
joblib.dump(preprocessor, 'preprocessor.joblib')
logging.info('Preprocessor disimpan ke preprocessor.joblib')

# 3. Build ANN model
input_dim = X_train.shape[1]
logging.info(f'Membangun model ANN dengan input_dim={input_dim}')
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
logging.info('Model dikompilasi dengan optimizer Adam dan loss MSE')

# Callback custom untuk mencatat test MSE setiap epoch
class TestMSECallback(keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.test_mse_per_epoch = []
    def on_epoch_end(self, epoch, logs=None):
        test_mse = self.model.evaluate(self.X_test, self.y_test, verbose=0)[0]
        self.test_mse_per_epoch.append(test_mse)
        logging.info(f'Epoch {epoch+1}: loss={logs["loss"]:.4f} - val_loss={logs["val_loss"]:.4f} - test_loss={test_mse:.4f}')

# Inisialisasi callback
cb_test_mse = TestMSECallback(X_test, y_test)

# 4. Train model
logging.info('Mulai training model ...')
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=0,  # Supaya log lebih rapi, output dihandle oleh logging
    callbacks=[cb_test_mse]
)
logging.info('Training selesai!')

# Save model
model.save('model.h5')
logging.info('Model disimpan ke model.h5')

# 5. Evaluate
val_mse = model.evaluate(X_val, y_val, verbose=0)[0]
test_mse = model.evaluate(X_test, y_test, verbose=0)[0]
logging.info(f'Validation MSE akhir: {val_mse:.4f}')
logging.info(f'Test MSE akhir: {test_mse:.4f}')

print(f'Validation MSE: {val_mse:.4f}')
print(f'Test MSE: {test_mse:.4f}')

# 6. Plot loss + test MSE
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(cb_test_mse.test_mse_per_epoch, label='Test MSE', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training, Validation & Test Loss per Epoch')
plt.tight_layout()
plt.savefig('aplikasi/static/loss_plot_with_test.png')
logging.info('Plot loss training/validation/test disimpan ke aplikasi/static/loss_plot_with_test.png')
# plt.show()  # Dihilangkan agar script lanjut ke blok prediksi

# 7. Prediksi seluruh data
import traceback
logging.info('\n================= HASIL PREDIKSI SELURUH DATA =================')
try:
    print('BLOK PREDIKSI DIJALANKAN')
    logging.info('Melakukan prediksi pada seluruh data di DaftarSaham.csv ...')
    # Ambil fitur dan target dari data asli (tanpa split)
    X_all, y_all, preprocessor = preprocess_stock_data(df)
    # Ambil index baris valid hasil preprocessing
    if y_all is not None:
        valid_idx = y_all.index
    else:
        valid_idx = pd.RangeIndex(X_all.shape[0])
    y_all_true = y_all.values if y_all is not None else None
    all_preds = model.predict(X_all).flatten()

    # Simpan hasil prediksi ke CSV hanya untuk baris valid
    hasil_prediksi_df = pd.DataFrame({
        'Code': df.loc[valid_idx, 'Code'].values,
        'Name': df.loc[valid_idx, 'Name'].values,
        'Target_Asli': y_all_true,
        'Prediksi_Model': all_preds
    })
    hasil_prediksi_df.to_csv('hasil_prediksi_semua.csv', index=False)
    logging.info('Hasil prediksi seluruh data disimpan ke hasil_prediksi_semua.csv')

    # Tampilkan ringkasan beberapa baris ke log
    logging.info('Contoh hasil prediksi (5 data pertama):')
    for i, row in hasil_prediksi_df.head().iterrows():
        logging.info(f"{row['Code']} | Asli={row['Target_Asli']:.2f} | Prediksi={row['Prediksi_Model']:.2f}")
except Exception as e:
    logging.error(f'Gagal melakukan prediksi seluruh data: {e}')
    print('\n[ERROR] Gagal melakukan prediksi seluruh data:', e)
    print(traceback.format_exc())
    logging.error(traceback.format_exc())
logging.info('==============================================================\n')

end_time = time.time()
elapsed = end_time - start_time
logging.info(f'=== TRAINING FINISHED in {elapsed:.2f} seconds ===')
