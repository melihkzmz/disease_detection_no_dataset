# 🫁 Kapsamlı Akciğer Hastalıkları Dataset Listesi

**Tarih:** 28 Ekim 2025  
**İstenen Hastalıklar:** COVID-19, Pnömoni, Tüberküloz, Akciğer Kanseri, Pnömotoraks

---

## ⭐ EN İYİ KAPSAMLI DATASET'LER

### 1. **COVID-QU-Ex Dataset** ⭐⭐⭐ EN KAPSAMLI

**Link:** https://www.kaggle.com/datasets/anasmohammedtahir/covidqu

**İçerik:**
- **33,920 X-ray görüntüsü**
- **4 Sınıf:**
  1. COVID-19 (3,616 görüntü)
  2. Non-COVID Pneumonia (Zatürre) (6,012 görüntü)
  3. Normal (Sağlıklı) (10,192 görüntü)
  4. Lung Opacity (Akciğer Opasitesi)
- **Boyut:** ~4.5 GB
- **Kalite:** Yüksek çözünürlük, tıbbi olarak doğrulanmış

**Artıları:**
- ✓ Dengeli veri dağılımı
- ✓ COVID-19 + Pnömoni ayırımı
- ✓ Profesyonel kalite
- ✓ Tıbbi doğrulama yapılmış

---

### 2. **TBX11K - Tuberculosis Dataset** ⭐⭐⭐ TÜBERKÜLOZ

**Link:** https://www.kaggle.com/datasets/usmanshams/tbx-11

**İçerik:**
- **11,200 X-ray görüntüsü**
- **2 Sınıf:**
  1. Tuberculosis (Tüberküloz/Verem)
  2. Normal (Sağlıklı)
- **Boyut:** ~3 GB
- **Özellik:** Bounding box annotations (segmentation için)

**Artıları:**
- ✓ Tüberküloz için en iyi dataset
- ✓ Lokalizasyon bilgisi var
- ✓ Dengeli dağılım
- ✓ Yüksek kalite

---

### 3. **Chest X-Ray - Pneumothorax** ⭐⭐ PNÖMOTORAKS

**Link:** https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task

**İçerik:**
- **12,047 X-ray görüntüsü**
- **2 Sınıf:**
  1. Pneumothorax (Pnömotoraks)
  2. Normal
- **Boyut:** ~5 GB

**Artıları:**
- ✓ Pnömotoraks için özel
- ✓ Çok sayıda örnek
- ✓ Segmentation masks mevcut

---

### 4. **LIDC-IDRI Lung Cancer Dataset** ⭐⭐⭐ AKCİĞER KANSERİ

**Link:** https://www.kaggle.com/datasets/danieldorenbaum/lidc-idri-tcga-manifest

**İçerik:**
- **1,018 hasta CT scan**
- Akciğer kanseri nodül tespiti
- **Boyut:** ~100+ GB ⚠️ (ÇOK BÜYÜK!)
- Annotation'lar mevcut

**Alternatif - Daha Küçük:**
**Lung Cancer CT Scan**
- Link: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- **1,000 CT görüntüsü**
- **3 Sınıf:** Normal, Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma
- **Boyut:** ~1.2 GB

---

### 5. **NIH Chest X-rays** ⭐⭐⭐ 14 HASTALIK

**Link:** https://www.kaggle.com/datasets/nih-chest-xrays/data

**İçerik:**
- **112,120 X-ray görüntüsü**
- **14 Hastalık sınıfı:**
  1. Atelectasis
  2. Cardiomegaly
  3. Effusion
  4. Infiltration
  5. Mass
  6. Nodule
  7. **Pneumonia** (Pnömoni) ✓
  8. **Pneumothorax** (Pnömotoraks) ✓
  9. Consolidation
  10. Edema
  11. Emphysema
  12. Fibrosis
  13. Pleural Thickening
  14. Hernia
- **Boyut:** ~45 GB ⚠️

**Artıları:**
- ✓ En kapsamlı dataset
- ✓ Pnömoni ve Pnömotoraks var
- ✓ Multi-label (bir görüntüde birden fazla hastalık olabilir)

**Eksileri:**
- ✗ Çok büyük
- ✗ COVID-19 yok
- ✗ Tüberküloz yok

---

## 🎯 ÖNERİLEN KOMBINASYONLAR

### Seçenek 1: KAPSAMLI 5 HASTALIK ⭐ ÖNERİLEN

**Dataset'ler:**
1. **COVID-QU-Ex** → COVID-19 + Pnömoni + Normal
2. **TBX11K** → Tüberküloz
3. **Pneumothorax Dataset** → Pnömotoraks
4. **Lung Cancer CT** → Akciğer Kanseri

**Toplam Sınıflar:**
1. COVID-19
2. Pneumonia (Bacterial/Viral)
3. Tuberculosis
4. Pneumothorax
5. Lung Cancer
6. Normal (Sağlıklı)

**Toplam Boyut:** ~15 GB
**Görüntü Sayısı:** 55,000+

---

### Seçenek 2: HIZLI BAŞLANGIÇ - 4 HASTALIK

**Sadece COVID-QU-Ex + TBX11K:**

**Sınıflar:**
1. COVID-19
2. Pneumonia (Non-COVID)
3. Tuberculosis
4. Normal

**Toplam Boyut:** ~7.5 GB
**Görüntü Sayısı:** 45,000+

---

### Seçenek 3: NIH KAPSAMLI - 14 HASTALIK

**Sadece NIH Dataset:**

**Avantajları:**
- 14 farklı hastalık
- Pnömoni ✓
- Pnömotoraks ✓
- Profesyonel kalite

**Dezavantajları:**
- COVID-19 YOK ✗
- Tüberküloz YOK ✗
- 45 GB (çok büyük) ✗

---

## 📊 DETAYLI KARŞILAŞTIRMA

| Dataset | COVID-19 | Pnömoni | Tüberküloz | Pnömotoraks | Kanser | Boyut | Görüntü |
|---------|----------|---------|------------|-------------|--------|-------|---------|
| COVID-QU-Ex | ✓ | ✓ | ✗ | Kısmen | ✗ | 4.5 GB | 33K |
| TBX11K | ✗ | ✗ | ✓ | ✗ | ✗ | 3 GB | 11K |
| Pneumothorax | ✗ | ✗ | ✗ | ✓ | ✗ | 5 GB | 12K |
| Lung Cancer CT | ✗ | ✗ | ✗ | ✗ | ✓ | 1.2 GB | 1K |
| NIH Chest X-rays | ✗ | ✓ | ✗ | ✓ | Kısmen | 45 GB | 112K |

---

## 🔧 TEKNIK DETAYLAR

### Görüntü Formatları:
- **X-ray (Röntgen):** COVID, Pnömoni, Tüberküloz, Pnömotoraks
- **CT Scan:** Akciğer Kanseri (daha iyi tespit)

### Görüntü Boyutları:
- COVID-QU-Ex: 256x256, 512x512, 1024x1024 (çeşitli)
- TBX11K: 512x512 ortalama
- Pneumothorax: 1024x1024
- Lung Cancer CT: 512x512

### Veri Augmentation Önerileri:
- Rotation: ±15 derece
- Zoom: %10
- Horizontal Flip: Hayır (tıbbi görüntülerde yön önemli)
- Brightness: %10 değişim
- Contrast: %10 değişim

---

## 🚀 HIZLI BAŞLANGIÇ REHBERİ

### 1. Basit Başlangıç (3 Hastalık)

**İndirilecekler:**
- COVID-QU-Ex Dataset

**Model:**
- 4 Sınıf: COVID-19, Pneumonia, Lung Opacity, Normal
- Boyut: 4.5 GB
- Eğitim süresi: ~2-3 saat (GPU ile)

---

### 2. Orta Seviye (5 Hastalık) ⭐ ÖNERİLEN

**İndirilecekler:**
1. COVID-QU-Ex → COVID + Pneumonia
2. TBX11K → Tuberculosis
3. Pneumothorax → Pneumothorax

**Model:**
- 6 Sınıf: COVID-19, Pneumonia, Tuberculosis, Pneumothorax, Lung Opacity, Normal
- Toplam Boyut: ~12.5 GB
- Eğitim süresi: ~4-6 saat (GPU ile)

---

### 3. Profesyonel (6 Hastalık + Kanser)

**İndirilecekler:**
1. COVID-QU-Ex
2. TBX11K
3. Pneumothorax
4. Lung Cancer CT

**Model:**
- 7 Sınıf: COVID-19, Pneumonia, Tuberculosis, Pneumothorax, Lung Cancer, Lung Opacity, Normal
- Toplam Boyut: ~13.7 GB
- Eğitim süresi: ~6-8 saat (GPU ile)

---

## 💡 DATASET İNDİRME LİNKLERİ

### Manuel İndirme (Önerilen):

**1. COVID-QU-Ex:**
```
https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
```

**2. TBX11K (Tuberculosis):**
```
https://www.kaggle.com/datasets/usmanshams/tbx-11
```

**3. Pneumothorax:**
```
https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task
```

**4. Lung Cancer CT:**
```
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
```

**5. NIH Chest X-rays (Opsiyonel - Çok büyük):**
```
https://www.kaggle.com/datasets/nih-chest-xrays/data
```

---

## 📁 KLASÖR YAPISI

İndirildikten sonra şu şekilde organize edin:

```
disease_detection/
├── Skin-Disease-Classifier/     # Mevcut cilt hastalıkları
│   └── (7 sınıf HAM10000)
│
└── Lung-Disease-Classifier/     # YENİ - Akciğer hastalıkları
    ├── datasets/
    │   ├── COVID-QU-Ex/
    │   │   ├── COVID/
    │   │   ├── Non-COVID/
    │   │   ├── Normal/
    │   │   └── Lung_Opacity/
    │   ├── TBX11K/
    │   │   ├── Tuberculosis/
    │   │   └── Normal/
    │   ├── Pneumothorax/
    │   │   ├── Pneumothorax/
    │   │   └── Normal/
    │   └── Lung_Cancer/
    │       ├── Adenocarcinoma/
    │       ├── Large_Cell_Carcinoma/
    │       ├── Squamous_Cell_Carcinoma/
    │       └── Normal/
    │
    ├── train_lung_model.py        # Eğitim scripti
    ├── lung_api.py                 # Flask API
    └── index.html                  # Web arayüzü
```

---

## 🎯 MODEL MİMARİSİ ÖNERİSİ

### Transfer Learning:
- **ResNet50** veya **DenseNet121** (Akciğer görüntüleri için daha iyi)
- **MobileNetV2** (Daha hızlı, daha hafif)
- **EfficientNet-B0** (Dengeli performans)

### Önerilen Ayarlar:
- Input Size: 224x224 (X-ray için yeterli)
- Batch Size: 32
- Epochs: 30 (early stopping ile)
- Learning Rate: 0.001 (başlangıç)
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## ⚕️ ÖNEMLİ NOTLAR

### Tıbbi Uyarı:
⚠️ Bu modeller sadece **eğitim/araştırma** amaçlıdır.  
⚠️ **Gerçek tıbbi teşhis için ASLA kullanmayın!**  
⚠️ Mutlaka bir **doktor/radyolog** görüşü alınmalıdır.

### Veri Gizliliği:
- Tüm dataset'ler anonim hasta verileri içerir
- HIPAA/GDPR uyumlu
- Araştırma amaçlı kullanım için açık

---

## 📞 Sonraki Adımlar

Hangi yaklaşımı seçmek istersiniz?

**A)** Basit Başlangıç - COVID-QU-Ex (3-4 hastalık)  
**B)** Dengeli - COVID + Tüberküloz + Pnömotoraks (5-6 hastalık) ⭐ ÖNERİLEN  
**C)** Tam Kapsamlı - Tüm hastalıklar + Kanser (6-7 hastalık)  
**D)** NIH ile 14 hastalık (45 GB)

---

**Hangi dataset kombinasyonunu indirip başlayalım?**

