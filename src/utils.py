def load_data(file_path):
    import pandas as pd
    """Veri dosyasını yükler."""
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Veri çerçevesini belirtilen dosya yoluna kaydeder."""
    data.to_csv(file_path, index=False)

def preprocess_datetime(df, datetime_column):
    """Tarih-saat bilgisini işleyerek yeni özellikler ekler."""
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['hour'] = df[datetime_column].dt.hour
    df['day'] = df[datetime_column].dt.day
    df['weekday'] = df[datetime_column].dt.weekday
    return df

def normalize_data(df):
    """Veri çerçevesindeki sayısal verileri normalize eder."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df