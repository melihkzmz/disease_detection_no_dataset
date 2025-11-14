# 🫁 Akciğer Hastalıkları Dataset'leri

## ⭐ EN POPÜLER DATASET'LER

### 1. **Chest X-Ray Images (Pneumonia)** - ÖNERİLEN #1
**Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**İçerik:**
- 5,863 X-ray görüntüsü
- **2 Sınıf:**
  - Normal (Sağlıklı)
  - Pneumonia (Zatürre)
- Pediatrik hastalar (1-5 yaş)
- Train/Val/Test ayrımı hazır
- **Boyut:** ~2 GB

**Artıları:**
- ✓ Çok popüler (5000+ notebook)
- ✓ İyi organize edilmiş
- ✓ Yüksek kalite görüntüler
- ✓ Başlangıç için mükemmel

---

### 2. **COVID-19 Radiography Database** - ÖNERİLEN #2
**Link:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

**İçerik:**
- 21,165 X-ray görüntüsü
- **4 Sınıf:**
  - COVID-19
  - Normal (Sağlıklı)
  - Lung Opacity (Akciğer Opasitesi)
  - Viral Pneumonia (Viral Zatürre)
- **Boyut:** ~1.2 GB

**Artıları:**
- ✓ 4 farklı sınıf
- ✓ COVID-19 tespiti
- ✓ Güncel dataset
- ✓ Dengeli dağılım

---

### 3. **NIH Chest X-rays** - EN KAPSAMLI
**Link:** https://www.kaggle.com/datasets/nih-chest-xrays/data

**İçerik:**
- 112,120 X-ray görüntüsü
- **14 Farklı Hastalık:**
  - Atelectasis (Atelektazi)
  - Cardiomegaly (Kardiyomegali)
  - Effusion (Efüzyon)
  - Infiltration (İnfiltrasyon)
  - Mass (Kitle)
  - Nodule (Nodül)
  - Pneumonia (Zatürre)
  - Pneumothorax (Pnömotoraks)
  - Consolidation (Konsolidasyon)
  - Edema (Ödem)
  - Emphysema (Amfizem)
  - Fibrosis (Fibrozis)
  - Pleural Thickening
  - Hernia
- **Boyut:** ~45 GB ⚠️ (ÇOK BÜYÜK!)

**Artıları:**
- ✓ En kapsamlı dataset
- ✓ 14 farklı hastalık
- ✓ Profesyonel kalite

**Eksileri:**
- ✗ Çok büyük (45GB)
- ✗ İndirme ve işleme uzun sürer
- ✗ Güçlü donanım gerekir

---

### 4. **TBX11K - Tuberculosis Dataset**
**Link:** https://www.kaggle.com/datasets/usmanshams/tbx-11

**İçerik:**
- 11,200 X-ray görüntüsü
- **2 Sınıf:**
  - Normal
  - Tuberculosis (Tüberküloz/Verem)
- **Boyut:** ~3 GB

**Artıları:**
- ✓ Tüberküloz tespiti
- ✓ Yüksek kalite
- ✓ Dengeli dağılım

---

### 5. **Lung Segmentation Dataset**
**Link:** https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

**İçerik:**
- 800 X-ray görüntüsü + Segmentation masks
- Akciğer segmentasyonu için
- **Boyut:** ~200 MB

**Artıları:**
- ✓ Segmentation için etiketlenmiş
- ✓ Küçük boyut
- ✓ Detaylı annotation

---

## 🎯 ÖNERİM: HANGİSİNİ SEÇMELİ?

### Başlangıç İçin (Kolay):
**→ Chest X-Ray Pneumonia Dataset** (2 sınıf)
- Basit, hızlı eğitim
- İyi sonuçlar
- Öğrenmeye ideal

### Orta Seviye (Önerilen):
**→ COVID-19 Radiography** (4 sınıf)
- COVID-19 + diğer hastalıklar
- Dengeli ve yönetilebilir
- Modern ve güncel

### İleri Seviye (Profesyonel):
**→ NIH Chest X-rays** (14 hastalık)
- En kapsamlı
- Gerçek dünya senaryosu
- Çoklu etiket classification

---

## 🔧 PROJE ENTEGRASYONư

### Seçenek 1: Ayrı Bir Model (Önerilen)
```
Cilt Hastalıkları Model (7 sınıf) → Mevcut API
Akciğer Hastalıkları Model (4 sınıf) → Yeni API
```

**Artıları:**
- ✓ İki model bağımsız çalışır
- ✓ Her biri kendi alanında uzman
- ✓ Farklı görüntü tipleri (cilt vs X-ray)

### Seçenek 2: Birleşik Sistem
```
Ana API → Görüntü tipi tespit → İlgili modeli çağır
```

**Artıları:**
- ✓ Tek API endpoint
- ✓ Otomatik yönlendirme
- ✓ Kullanıcı dostu

---

## 📦 HIZLI BAŞLANGIÇ

### 1. Dataset İndir (COVID-19 Radiography)

**Manuel İndirme:**
1. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. "Download" butonuna tıkla
3. ZIP'i çıkar

**Klasör Yapısı:**
```
datasets/
  ├── HAM10000/           # Mevcut - Cilt hastalıkları
  └── COVID19-Radiography/  # Yeni - Akciğer hastalıkları
      ├── COVID/
      ├── Normal/
      ├── Lung_Opacity/
      └── Viral Pneumonia/
```

### 2. Model Eğitimi

Aynı `train_combined_model.py` scriptini uyarlayabiliriz:
- MobileNetV2 → Chest X-ray için
- 4 sınıf: COVID, Normal, Opacity, Pneumonia
- Data augmentation

### 3. API Oluştur

```python
# lung_disease_api.py
# 4 sınıflı akciğer hastalığı API
```

### 4. Web Arayüzü

```html
<!-- index_lung.html -->
<!-- Akciğer X-ray yükleme ve analiz -->
```

---

## 💡 ÖNERİLER

### Görüntü Boyutu:
- Cilt hastalıkları: 224x224 (renkli, RGB)
- Akciğer X-ray: 224x224 (gri tonlama, genellikle)

### Model Mimarisi:
- Her iki proje için MobileNetV2 kullanılabilir
- Transfer learning ile hızlı eğitim

### Deployment:
- İki ayrı Flask API:
  - `http://localhost:5000` → Cilt hastalıkları
  - `http://localhost:5001` → Akciğer hastalıkları

---

## 🚀 SONRAKİ ADIMLAR

1. **Hangi dataset'i istiyorsunuz?**
   - Pneumonia (2 sınıf) - Basit
   - COVID-19 (4 sınıf) - **ÖNERİLEN**
   - NIH (14 hastalık) - İleri seviye

2. **Dataset'i indirin** (manuel veya Kaggle API)

3. **Eğitim scriptini hazırlayalım**

4. **Yeni API ve web arayüzü oluşturalım**

---

**Hangisini tercih ediyorsunuz?**
- A) COVID-19 Radiography (4 sınıf) ⭐ Önerilen
- B) Pneumonia (2 sınıf) - Basit başlangıç
- C) NIH Dataset (14 hastalık) - Kapsamlı

**Veya başka bir dataset mi istiyorsunuz?**

