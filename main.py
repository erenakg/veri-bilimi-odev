"""
İstanbul Trafik Davranış Kalıplarının Analizi - Ana Dosya
Bu dosya projenin tüm adımlarını sırasıyla çalıştırır.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

# Kendi modüllerimizi import edelim
import sys
sys.path.append('src')

from data_preprocessing import preprocess_data, save_processed_data
from clustering_analysis import determine_optimal_clusters, perform_kmeans_clustering
from visualization import plot_cluster_analysis, plot_pca_clusters, create_traffic_map, plot_cluster_characteristics

def create_directories():
    """Gerekli klasörleri oluşturur."""
    try:
        # Mevcut dosyaları sil, klasörleri oluştur
        if os.path.exists('results/graphs') and os.path.isfile('results/graphs'):
            os.remove('results/graphs')
        if os.path.exists('results/maps') and os.path.isfile('results/maps'):
            os.remove('results/maps')
            
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/graphs', exist_ok=True)
        os.makedirs('results/maps', exist_ok=True)
        print("✓ Klasörler kontrol edildi")
    except Exception as e:
        print(f"Klasör oluşturma hatası: {e}")

def main():
    """Ana analiz fonksiyonu"""
    print("=== İstanbul Trafik Davranış Kalıplarının Analizi ===\n")
    print("📊 1.7M veri ile tam analiz yapılıyor...")
    
    # Klasörleri oluştur
    create_directories()
    
    # 1. Veri Yükleme ve Ön İşleme
    print("1. Veri yükleniyor ve ön işleme yapılıyor...")
    
    try:
        # Tam veri dosyasını yükle
        raw_data = pd.read_csv('data/raw/traffic_density_202501.csv')
        print(f"Ham veri boyutu: {raw_data.shape}")
        print(f"Ham veri sütunları: {list(raw_data.columns)}")
        
        # Veri ön işleme (tam veri ile)
        processed_data = preprocess_data('data/raw/traffic_density_202501.csv')
        print(f"İşlenmiş veri boyutu: {processed_data.shape}")
        
        # İşlenmiş veriyi kaydet
        save_processed_data(processed_data, 'data/processed/temizlenmis_veri_tam.csv')
        print("✓ Veri ön işleme tamamlandı\n")
        
    except Exception as e:
        print(f"Veri işleme hatası: {e}")
        return
    
    # 2. Kümeleme için özellikleri hazırla
    print("2. Kümeleme için özellikler hazırlanıyor...")
    
    try:
        available_columns = processed_data.columns.tolist()
        print(f"Mevcut sütunlar: {available_columns}")
        
        # Kümeleme özelliklerini seç
        feature_columns = []
        
        # Ana trafik özellikleri
        if 'AVERAGE_SPEED' in available_columns:
            feature_columns.append('AVERAGE_SPEED')
        if 'NUMBER_OF_VEHICLES' in available_columns:
            feature_columns.append('NUMBER_OF_VEHICLES')
        if 'MINIMUM_SPEED' in available_columns:
            feature_columns.append('MINIMUM_SPEED')
        if 'MAXIMUM_SPEED' in available_columns:
            feature_columns.append('MAXIMUM_SPEED')
            
        # Zaman özellikleri
        if 'hour' in available_columns:
            feature_columns.append('hour')
        if 'day_of_week' in available_columns:
            feature_columns.append('day_of_week')
            
        print(f"Kümeleme için seçilen özellikler: {feature_columns}")
        
        if len(feature_columns) == 0:
            print("❌ Kümeleme için uygun özellik bulunamadı!")
            return
            
        X = processed_data[feature_columns].values
        
        # Veriyi normalize et
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"✓ Özellikler hazırlandı: {X_scaled.shape}")
        print(f"⚠️  Bu büyük veri ile işlem uzun sürebilir...\n")
        
    except Exception as e:
        print(f"Özellik hazırlama hatası: {e}")
        return
    
    # 3. Optimal küme sayısını belirle
    print("3. Optimal küme sayısı belirleniyor...")
    print("⏳ Büyük veri ile küme analizi yapılıyor, lütfen bekleyin...")
    
    try:
        # 1.7M veri için küme sayısını sınırla
        max_clusters = 6  # Büyük veri için makul limit
        
        print(f"🔍 Test edilecek küme sayısı: 2-{max_clusters}")
        
        inertia, silhouette_scores = determine_optimal_clusters(X_scaled, max_k=max_clusters)
        
        # Elbow ve Silhouette grafikleri
        plot_cluster_analysis(processed_data, inertia, silhouette_scores)
        
        # Optimal küme sayısını belirle
        if len(silhouette_scores) > 0:
            optimal_k = np.argmax(silhouette_scores) + 2
        else:
            optimal_k = 4  # Varsayılan
            
        print(f"✓ Optimal küme sayısı: {optimal_k}\n")
        
    except Exception as e:
        print(f"Küme sayısı belirleme hatası: {e}")
        optimal_k = 4
        print(f"Varsayılan küme sayısı kullanılıyor: {optimal_k}\n")
    
    # 4. K-Means kümeleme uygula
    print("4. K-Means kümeleme uygulanıyor...")
    print("⏳ 1.7M veri ile kümeleme yapılıyor, bu işlem 10-15 dakika sürebilir...")
    
    try:
        cluster_labels = perform_kmeans_clustering(X_scaled, optimal_k)
        processed_data['cluster'] = cluster_labels
        
        print(f"✓ {optimal_k} küme oluşturuldu")
        print(f"Küme dağılımı:")
        cluster_counts = processed_data['cluster'].value_counts().sort_index()
        print(cluster_counts)
        print()
        
    except Exception as e:
        print(f"Kümeleme hatası: {e}")
        return
    
    # 5. Sonuçları görselleştir
    print("5. Sonuçlar görselleştiriliyor...")
    print("⏳ Büyük veri görselleştirmesi yapılıyor...")
    
    try:
        # PCA ile kümeleri görselleştir (örnekleme ile)
        print("📊 PCA analizi için veri örneklemesi yapılıyor...")
        pca_sample_size = min(50000, len(processed_data))
        pca_sample_data = processed_data.sample(n=pca_sample_size, random_state=42)
        pca_sample_features = X_scaled[pca_sample_data.index]
        pca_sample_labels = cluster_labels[pca_sample_data.index]
        
        plot_pca_clusters(pca_sample_data, pca_sample_features, pca_sample_labels)
        
        # Küme özelliklerini analiz et (tam veri ile)
        cluster_stats = plot_cluster_characteristics(processed_data, feature_columns)
        if cluster_stats is not None:
            print("Küme İstatistikleri:")
            print(cluster_stats)
            print()
        
        # Harita oluştur (koordinat bilgisi varsa - örnekleme ile)
        if 'LATITUDE' in processed_data.columns and 'LONGITUDE' in processed_data.columns:
            print("🗺️  Harita için veri örneklemesi yapılıyor...")
            map_sample_size = min(10000, len(processed_data))  # Harita için 10K nokta
            map_data = processed_data.sample(n=map_sample_size, random_state=42)
            
            # Sütun isimlerini düzelt
            map_data = map_data.rename(columns={'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'})
            traffic_map = create_traffic_map(map_data)
        else:
            print("⚠️ Koordinat bilgisi bulunamadı, harita oluşturulamadı")
            
        print("✓ Görselleştirme tamamlandı\n")
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")
    
    # 6. Küme yorumları
    print("6. Küme Yorumları:")
    print("="*60)
    
    try:
        for cluster_id in range(optimal_k):
            cluster_data = processed_data[processed_data['cluster'] == cluster_id]
            
            print(f"\n🔵 KÜME {cluster_id}:")
            print(f"  📊 Veri Noktası Sayısı: {len(cluster_data):,}")
            print(f"  📈 Toplam Veri Oranı: %{(len(cluster_data)/len(processed_data)*100):.1f}")
            
            # Ana özellikler için ortalama
            for feature in feature_columns:
                if feature in cluster_data.columns:
                    avg_value = cluster_data[feature].mean()
                    if 'SPEED' in feature:
                        print(f"  🚗 Ortalama {feature}: {avg_value:.1f} km/h")
                    elif 'VEHICLES' in feature:
                        print(f"  🚙 Ortalama {feature}: {avg_value:.0f} araç")
                    elif feature == 'hour':
                        print(f"  🕐 Ortalama Saat: {avg_value:.1f}")
                    else:
                        print(f"  📋 Ortalama {feature}: {avg_value:.2f}")
            
            # Trafik kalıbını belirle
            if 'AVERAGE_SPEED' in cluster_data.columns:
                avg_speed = cluster_data['AVERAGE_SPEED'].mean()
                avg_vehicles = cluster_data['NUMBER_OF_VEHICLES'].mean() if 'NUMBER_OF_VEHICLES' in cluster_data.columns else 0
                
                if avg_speed < 30 and avg_vehicles > 150:
                    pattern = "🔴 Yoğun-Sıkışık Trafik"
                elif avg_speed > 50 and avg_vehicles < 100:
                    pattern = "🟢 Akıcı-Düşük Yoğunluk"
                elif 30 <= avg_speed <= 50:
                    pattern = "🟡 Orta Yoğunluk Trafik"
                else:
                    pattern = "⚪ Karma Trafik"
                    
                print(f"  🎯 Trafik Kalıbı: {pattern}")
                
    except Exception as e:
        print(f"Küme yorumlama hatası: {e}")
    
    print("\n" + "="*60)
    print("🎉 === 1.7M Veri ile Analiz Tamamlandı! ===")
    print("📁 Sonuçlar 'results' klasöründe kaydedildi.")
    print("\n📋 Oluşturulan dosyalar:")
    print("  📊 results/graphs/cluster_analysis.png")
    print("  🎨 results/graphs/pca_clusters.png") 
    print("  📈 results/graphs/cluster_characteristics.png")
    print("  🗺️ results/maps/traffic_clusters.html")
    print("  💾 data/processed/temizlenmis_veri_tam.csv")

if __name__ == "__main__":
    main()