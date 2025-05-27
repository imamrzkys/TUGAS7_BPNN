import numpy as np
import joblib
from tensorflow import keras
from .preprocessing import preprocess_stock_data
import pandas as pd

# Load model and preprocessor once
def load_model_and_preprocessor():
    model = keras.models.load_model('data/model_saham.h5')
    preprocessor = joblib.load('data/preprocessor.joblib')
    return model, preprocessor

model, preprocessor = None, None

def predict_last_price(form_data):
    import pandas as pd
    code = form_data.get('Code', '').strip()
    name = form_data.get('Name', '').strip()
    # Baca hasil_prediksi_semua.csv
    try:
        df_pred = pd.read_csv('data/hasil_prediksi_semua.csv')
    except Exception:
        return {'error': 'File hasil_prediksi_semua.csv tidak ditemukan.'}
    # Cari baris yang cocok
    row = df_pred[(df_pred['Code'].astype(str).str.strip() == code) & (df_pred['Name'].astype(str).str.strip() == name)]
    if not row.empty:
        pred_val = row.iloc[0]['Prediksi_Model']
        return {'prediction': float(pred_val)}
    else:
        return {'error': f'Data dengan kode \'{code}\' dan nama \'{name}\' tidak ditemukan.'}

