import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import folium
import numpy as np
import os

plt.style.use('default')

def plot_cluster_characteristics(data, feature_columns):
    """Her kümenin özelliklerini görselleştirir."""
    try:
        n_features = len(feature_columns)
        if n_features == 0:
            print("⚠️ Görselleştirilecek özellik bulunamadı")
            return None
            
        # Grafik boyutunu ayarla
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Küme istatistikleri
        cluster_stats = data.groupby('cluster')[feature_columns].mean().round(3)
        
        # Her özellik için grafik çiz
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, feature in enumerate(feature_columns):
            if i < len(axes):
                bars = axes[i].bar(cluster_stats.index, cluster_stats[feature], 
                                 color=[colors[j % len(colors)] for j in cluster_stats.index])
                axes[i].set_title(f'Kümelere Göre {feature}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Küme')
                axes[i].set_ylabel(f'Ortalama {feature}')
                
                # Değerleri bar üzerine yaz
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:.2f}', ha='center', va='bottom')
        
        # Boş grafikleri gizle
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        os.makedirs('results/graphs', exist_ok=True)
        plt.savefig('results/graphs/cluster_characteristics.png', dpi=300, bbox_inches='tight')
        print("✓ Küme karakteristikleri grafiği kaydedildi")
        plt.show()
        
        return cluster_stats
    except Exception as e:
        print(f"Cluster characteristics plot hatası: {e}")
        return None

def plot_cluster_analysis(data, inertia, silhouette_scores):
    """Küme analizi sonuçlarını görselleştirir."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow Method
        k_range = range(2, len(inertia) + 2)
        ax1.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method - Optimal Küme Sayısı', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Küme Sayısı (k)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Silhouette Scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Silhouette Skorları', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Küme Sayısı (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Skoru', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('results/graphs', exist_ok=True)
        plt.savefig('results/graphs/cluster_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Küme analizi grafiği kaydedildi")
        plt.show()
    except Exception as e:
        print(f"Cluster analysis plot hatası: {e}")

def plot_pca_clusters(data, features, labels):
    """PCA ile kümeleri 2D'de görselleştirir."""
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i in range(len(np.unique(labels))):
            cluster_data = pca_result[labels == i]
            if len(cluster_data) > 0:
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                           c=colors[i % len(colors)], label=f'Küme {i}', 
                           alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        
        plt.title('PCA ile Küme Görselleştirmesi', fontsize=16, fontweight='bold')
        plt.xlabel(f'PCA Bileşen 1 (Varyans: {pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        plt.ylabel(f'PCA Bileşen 2 (Varyans: {pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        os.makedirs('results/graphs', exist_ok=True)
        plt.savefig('results/graphs/pca_clusters.png', dpi=300, bbox_inches='tight')
        print("✓ PCA küme grafiği kaydedildi")
        plt.show()
    except Exception as e:
        print(f"PCA plot hatası: {e}")

def create_traffic_map(data):
    """Harita üzerinde kümeleri görselleştirir."""
    try:
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            print("⚠️ Koordinat bilgisi bulunamadı (latitude, longitude)")
            return None
            
        # Merkez koordinat hesapla
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        
        # Harita oluştur
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Küme renkleri
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Her nokta için marker ekle
        for idx, row in data.iterrows():
            cluster = int(row['cluster'])
            color = colors[cluster % len(colors)]
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"Küme: {cluster}",
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Haritayı kaydet
        os.makedirs('results/maps', exist_ok=True)
        m.save('results/maps/traffic_clusters.html')
        print("✓ Harita kaydedildi: results/maps/traffic_clusters.html")
        
        return m
    except Exception as e:
        print(f"Harita oluşturma hatası: {e}")
        return None