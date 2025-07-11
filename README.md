# İstanbul Trafik Davranış Kalıplarının Analizi

Bu proje, İstanbul'a ait saatlik trafik verilerinin analizi ve kümeleme ile trafik kalıplarının belirlenmesi amacıyla geliştirilmiştir.

## Kurulum

1. Repository'yi klonlayın:
```bash
git clone <repository-url>
cd veri-bilimi-odev
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

1. Jupyter Notebook'u çalıştırın:
```bash
jupyter notebook notebooks/trafik_analiz_notebook.ipynb
```

2. Notebook'taki hücreleri sırasıyla çalıştırın.

## Proje Yapısı

- `data/raw/` - Ham veri dosyaları
- `data/processed/` - İşlenmiş veri dosyaları
- `notebooks/` - Jupyter Notebook dosyaları
- `src/` - Python modülleri
- `results/` - Analiz sonuçları ve görseller