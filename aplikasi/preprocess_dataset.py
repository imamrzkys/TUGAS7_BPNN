import pandas as pd
import os
import sys

# Pastikan folder root project ada di sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aplikasi.preprocessing import preprocess_stock_data

# Path ke dataset asli dan hasil
input_path = os.path.join('data', 'DaftarSaham.csv')
output_path = os.path.join('data', 'DaftarSaham_preprocessed.csv')

# Baca dataset
print(f"Membaca data dari {input_path} ...")
df = pd.read_csv(input_path)

# Preprocessing
print("Melakukan preprocessing data ...")
X_processed, y, preprocessor = preprocess_stock_data(df)

# Gabungkan fitur dan target jika target tersedia
# Ubah X_processed (numpy array) menjadi DataFrame dengan nama kolom yang sesuai
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

import pandas as pd

df_processed = pd.DataFrame(X_processed, columns=feature_names)

# Simpan hasil ke file baru
print(f"Menyimpan hasil ke {output_path} ...")
df_processed.to_csv(output_path, index=False)
print("Selesai! Data hasil preprocessing telah disimpan.")
