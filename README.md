# İstanbul Trafik Davranış Kalıplarının Analizi

Bu proje, İstanbul'a ait saatlik trafik verilerini kullanarak trafik davranışlarını analiz etmeyi amaçlamaktadır. Proje, araç yoğunluğu, hız değerleri ve coğrafi konum bilgileri temel alınarak trafik davranışlarını otomatik olarak gruplandırmayı hedeflemektedir. K-Means algoritması ile trafik bölgeleri farklı kalıplara ayrılarak, sıkışık, akıcı ve karma trafik örüntüleri belirlenmektedir.

## Proje Yapısı

- **data/raw/istanbul_trafik_verisi.csv**: İstanbul'a ait saatlik trafik verilerini içeren ham veri dosyası.
- **data/processed/temizlenmis_veri.csv**: Veri ön işleme adımlarından geçirilmiş ve temizlenmiş trafik verilerini içeren dosya.
- **src/data_preprocessing.py**: Veri ön işleme işlemlerini gerçekleştiren fonksiyonları içerir. Zaman bilgisi çıkarımı, veri temizliği ve normalizasyon gibi işlemleri yapar.
- **src/clustering_analysis.py**: K-Means kümeleme algoritmasını uygulayan ve küme sayısını belirlemek için Elbow veya Silhouette yöntemlerini kullanan fonksiyonları içerir.
- **src/visualization.py**: Görselleştirme işlemlerini gerçekleştiren fonksiyonları içerir. PCA ile boyut indirgeme ve harita tabanlı küme gösterimi için gerekli grafik ve harita oluşturma işlemlerini yapar.
- **src/utils.py**: Projede kullanılan yardımcı fonksiyonları içerir. Veri yükleme, veri kaydetme gibi genel işlemleri gerçekleştirir.
- **notebooks/trafik_analiz_notebook.ipynb**: Proje sürecinin adım adım belgelenmesi ve analizlerin yapılması için kullanılan Jupyter Notebook dosyası.
- **results/graphs**: Proje sonuçlarına ait grafiklerin kaydedileceği klasör.
- **results/maps**: Harita tabanlı görselleştirmelerin kaydedileceği klasör.
- **requirements.txt**: Projede kullanılan Python kütüphanelerinin listelendiği dosya.

## Kullanım

1. Gerekli kütüphaneleri yüklemek için `requirements.txt` dosyasını kullanın.
2. Veri ön işleme adımlarını gerçekleştirmek için `data_preprocessing.py` dosyasını çalıştırın.
3. K-Means kümeleme analizi için `clustering_analysis.py` dosyasını kullanın.
4. Sonuçları görselleştirmek için `visualization.py` dosyasını çalıştırın.
5. Analiz sürecini takip etmek için `trafik_analiz_notebook.ipynb` dosyasını inceleyin.

## Sonuç

Bu proje, İstanbul'daki trafik davranışlarını anlamak ve yönetmek için önemli bilgiler sunmayı hedeflemektedir. Elde edilen bulgular, trafik yönetimi ve şehir planlaması açısından değerli çıkarımlar sağlayabilir.