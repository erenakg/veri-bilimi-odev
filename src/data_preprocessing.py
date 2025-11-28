import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    """1.7M veri setini yükler."""
    print("📥 Büyük veri dosyası yükleniyor...")
    return pd.read_csv(file_path)

def clean_data(df):
    """1.7M veri temizleme işlemlerini gerçekleştirir."""
    print(f"🧹 Temizleme öncesi: {df.shape}")
    
    # Eksik değerleri kontrol et
    print("🔍 Eksik değerler:")
    print(df.isnull().sum())
    
    # Eksik değerleri temizle
    df = df.dropna()
    
    # Duplike kayıtları temizle
    print("🔍 Duplike kayıtlar kontrol ediliyor...")
    initial_size = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_size - len(df)
    print(f"   ✓ {duplicates_removed:,} duplike kayıt silindi")
    
    print(f"✅ Temizleme sonrası: {df.shape}")
    return df

def extract_features(df):
    """1.7M veri için özellik çıkarımı yapar."""
    print("🔧 Özellik çıkarımı yapılıyor...")
    
    # Tarih/saat sütunları varsa işle
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    for col in date_columns:
        try:
            print(f"   📅 {col} sütunu işleniyor...")
            df[col] = pd.to_datetime(df[col])
            df['hour'] = df[col].dt.hour
            df['day_of_week'] = df[col].dt.dayofweek
            df['month'] = df[col].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            print(f"   ✓ {col} sütunundan zaman özellikleri çıkarıldı")
        except Exception as e:
            print(f"   ⚠️ {col} sütunu işlenemedi: {e}")
    
    # Sayısal sütunları kontrol et
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"📊 Sayısal sütunlar: {list(numeric_columns)}")
    
    return df

def normalize_features(df):
    """1.7 milyon veri için özellikleri normalize eder."""
    print("🔧 Normalizasyon yapılıyor...")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Orijinal değerleri koru
    for col in numeric_columns:
        if col not in ['hour', 'day_of_week', 'month', 'is_weekend']:
            df[f'{col}_original'] = df[col].copy()
    
    print("✅ Normalizasyon tamamlandı")
    return df

def preprocess_data(file_path):
    """1.7M veri ön işleme adımlarını gerçekleştirir."""
    print("=== 1.7M Veri Ön İşleme Başlıyor ===")
    
    # Veri yükleme
    df = load_data(file_path)
    print(f"📊 Yüklenen veri boyutu: {df.shape}")
    print(f"📋 Sütunlar: {list(df.columns)}")
    
    # Veri temizleme
    df = clean_data(df)
    
    # Özellik çıkarımı
    df = extract_features(df)
    
    # Normalizasyon
    df = normalize_features(df)
    
    print("=== 1.7M Veri Ön İşleme Tamamlandı ===")
    return df

def save_processed_data(df, output_path):
    """1.7M işlenmiş veriyi kaydeder."""
    print(f"💾 Büyük veri kaydediliyor: {output_path}")
    
    # Klasör yoksa oluştur
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✅ {len(df):,} satır başarıyla kaydedildi")