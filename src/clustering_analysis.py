import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Özniteliklerin seçilmesi ve normalizasyonu
    features = data[['ortalama_hiz', 'arac_sayisi']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def determine_optimal_clusters(data, max_k=6):
    """1.7M veri için optimize edilmiş küme analizi."""
    print(f"🔍 Kümeleme analizi başlıyor...")
    print(f"   📊 Tam veri boyutu: {data.shape}")
    print(f"   🎯 Test edilecek küme sayısı: 2-{max_k}")
    print(f"   ⚠️  Büyük veri nedeniyle kümeleme için örnekleme yapılıyor...")
    
    # Büyük veri için kümeleme analizi örneklemesi
    sample_size = min(200000, len(data))  # 200K ile küme analizi
    sample_indices = np.random.choice(len(data), size=sample_size, replace=False)
    data_sample = data[sample_indices]
    print(f"   ✓ {len(data_sample):,} veri noktası ile küme analizi yapılacak")
    
    inertia = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        print(f"   🔄 K={k} test ediliyor...", end="", flush=True)
        try:
            # Büyük veri için hızlandırılmış ayarlar
            kmeans = KMeans(
                n_clusters=k, 
                random_state=42, 
                n_init=5,  # Hızlandırma için azaltıldı
                max_iter=200,  # Yeterli iterasyon
                tol=1e-3,  # Gevşetilmiş tolerans
                algorithm='lloyd'  # 'auto' yerine 'lloyd' veya 'elkan' kullanılabilir
            )
            kmeans.fit(data_sample)
            
            inertia_val = kmeans.inertia_
            silhouette_val = silhouette_score(data_sample, kmeans.labels_)
            
            inertia.append(inertia_val)
            silhouette_scores.append(silhouette_val)
            
            print(f" ✓ (Inertia: {inertia_val:.0f}, Silhouette: {silhouette_val:.3f})")
            
        except Exception as e:
            print(f" ❌ Hata: {e}")
            break
    
    print(f"✅ Kümeleme analizi tamamlandı!")
    return inertia, silhouette_scores

def plot_elbow_method(inertia):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(inertia) + 2), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()

def plot_silhouette_scores(silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()

def perform_kmeans_clustering(data, n_clusters):
    """1.7M veri için optimize edilmiş K-Means kümeleme."""
    print(f"🔄 K-Means kümeleme başlıyor (K={n_clusters})...")
    print(f"   📊 Tam veri boyutu: {data.shape}")
    print(f"   ⏳ Bu işlem 10-15 dakika sürebilir...")
    
    # 1.7M veri için optimizasyon ayarları
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=5,  # Hızlandırma için
        max_iter=300,
        tol=1e-3,
        algorithm='auto'  # En uygun algoritmayı seçer
    )
    
    # Tam veri ile kümeleme
    labels = kmeans.fit_predict(data)
    
    print(f"✅ 1.7M veri ile kümeleme tamamlandı!")
    return labels

# Örnek kullanım
if __name__ == "__main__":
    raw_data_path = '../data/processed/temizlenmis_veri.csv'
    data = load_data(raw_data_path)
    scaled_data = preprocess_data(data)
    
    inertia, silhouette_scores = determine_optimal_clusters(scaled_data)
    plot_elbow_method(inertia)
    plot_silhouette_scores(silhouette_scores)
    
    optimal_k = 3  # Örnek olarak belirlenen optimal küme sayısı
    labels = perform_kmeans_clustering(scaled_data, optimal_k)
    data['cluster'] = labels
    print(data.head())