from flask import Blueprint, render_template, request
from .model import predict_last_price
import pandas as pd
import os

main = Blueprint('main', __name__)

# Path to CSV
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'hasil_prediksi_semua.csv')

def load_prediction_data():
    df = pd.read_csv(CSV_PATH)
    # Drop rows with missing essential data
    df = df.dropna(subset=['Code', 'Name', 'Target_Asli'])
    code_options = df['Code'].astype(str).unique().tolist()
    name_options = df['Name'].astype(str).unique().tolist()
    target_options = sorted(df['Target_Asli'].unique().tolist())
    table_data = df.to_dict(orient='records')
    return code_options, name_options, target_options, table_data

@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    code_options, name_options, target_options, table_data = load_prediction_data()
    if request.method == 'POST':
        form_data = request.form.to_dict()
        result = predict_last_price(form_data)
        if isinstance(result, dict):
            if 'prediction' in result:
                prediction = result['prediction']
            elif 'error' in result:
                error = result['error']
        else:
            prediction = result
    import json
    code_to_name = {row['Code']: row['Name'] for row in table_data}
    name_to_code = {row['Name']: row['Code'] for row in table_data}
    codeToName_json = json.dumps(code_to_name, ensure_ascii=False)
    nameToCode_json = json.dumps(name_to_code, ensure_ascii=False)
    return render_template(
        'index.html',
        prediction=prediction,
        error=error,
        code_options=code_options,
        name_options=name_options,
        target_options=target_options,
        table_data=table_data,
        codeToName_json=codeToName_json,
        nameToCode_json=nameToCode_json
    )
