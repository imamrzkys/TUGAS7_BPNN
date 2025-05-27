import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def preprocess_stock_data(df, now=None):
    """
    Preprocess raw stock DataFrame for model training or inference.
    - Converts date columns
    - Adds DaysSinceListing
    - Converts time columns to numeric deltas
    - One-hot encodes categorical features
    - Normalizes numerical features
    - Cleans NaN/infinite values
    Returns: X_processed, y, preprocessor (ColumnTransformer)
    """
    if now is None:
        now = datetime.now()
    df = df.copy()
    # Convert ListingDate to datetime
    df['ListingDate'] = pd.to_datetime(df['ListingDate'], errors='coerce')
    df['DaysSinceListing'] = (now - df['ListingDate']).dt.days
    # Convert time columns to numerics
    time_columns = [
        ('MinutesFirstAdded', 'minutes'),
        ('MinutesLastUpdated', 'minutes'),
        ('HourlyFirstAdded', 'hours'),
        ('HourlyLastUpdated', 'hours'),
        ('DailyFirstAdded', 'days'),
        ('DailyLastUpdated', 'days')
    ]
    for col, unit in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].fillna(df['ListingDate'])
        if unit == 'minutes':
            df[col] = (df[col] - df['ListingDate']).dt.total_seconds() // 60
        elif unit == 'hours':
            df[col] = (df[col] - df['ListingDate']).dt.total_seconds() // 3600
        elif unit == 'days':
            df[col] = (df[col] - df['ListingDate']).dt.days
    # Features and target
    feature_cols = [
        'Shares', 'ListingBoard', 'Sector', 'MarketCap',
        'MinutesFirstAdded', 'MinutesLastUpdated',
        'HourlyFirstAdded', 'HourlyLastUpdated',
        'DailyFirstAdded', 'DailyLastUpdated', 'DaysSinceListing'
    ]
    X = df[feature_cols]
    y = df['LastPrice'] if 'LastPrice' in df else None
    # Clean NaN/infinite
    full = pd.concat([X, y], axis=1) if y is not None else X
    full = full.replace([np.inf, -np.inf], np.nan).dropna()
    X = full[feature_cols]
    if y is not None:
        y = full['LastPrice']
    # Categorical and numerical features
    categorical_features = ['ListingBoard', 'Sector']
    numerical_features = [col for col in feature_cols if col not in categorical_features]
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


def preprocess_and_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Preprocess dataset and split into train/val/test sets.
    - Tanggal ke numerik
    - One-hot encoding kategori
    - Normalisasi numerik
    - Split train/val/test
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    X_processed, y, preprocessor = preprocess_stock_data(df)
    # Split test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state)
    # Split validation from trainval
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
