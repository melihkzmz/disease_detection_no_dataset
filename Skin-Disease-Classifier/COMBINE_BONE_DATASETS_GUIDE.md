# Farklı Veri Setlerini Birleştirme Rehberi

## ✅ Evet, Kesinlikle Yapılabilir!

Farklı kaynaklardan gelen veri setlerini birleştirip tek bir model eğitebilirsin. Bu yaklaşım:
- ✅ **Çok yaygın** ve pratik
- ✅ **Daha fazla veri** = daha iyi model
- ✅ **Sınıf dengesi** sağlamak daha kolay
- ✅ **Gerçek dünya senaryosuna** daha yakın

---

## 📋 Birleştirme Stratejisi

### Örnek Senaryo:
```
1. Mendeley'den → Tumor dataset (500 görüntü)
2. Kaggle'dan → Fracture dataset (800 görüntü)
3. Mendeley'den → Infection dataset (600 görüntü)
4. Kaggle'dan → Normal dataset (1000 görüntü)
5. MURA'dan → Normal/Abnormal (istenirse)

BİRLEŞTİR → Tek bir model eğit
```

---

## ⚠️ Dikkat Edilmesi Gerekenler

### 1. **Format Standardizasyonu** (ÖNEMLİ!)
Farklı veri setleri farklı formatlarda olabilir:
- DICOM → PNG/JPG'ye dönüştür
- Farklı çözünürlükler → Aynı boyuta getir (örn: 256x256, 512x512)
- Farklı renk modları → RGB'ye normalize et

### 2. **Sınıf İsimlendirmesi**
Tutarlı sınıf isimleri kullan:
```python
CLASS_MAPPING = {
    # Dataset 1'den
    'bone_tumor': 'Tumor',
    'tumor': 'Tumor',
    'osteosarcoma': 'Tumor',
    
    # Dataset 2'den
    'fracture': 'Fracture',
    'broken_bone': 'Fracture',
    'bone_break': 'Fracture',
    
    # Dataset 3'ten
    'infection': 'Infection',
    'osteomyelitis': 'Infection',
    'bone_infection': 'Infection',
    
    # Normal
    'normal': 'Normal',
    'healthy': 'Normal',
    'no_disease': 'Normal'
}
```

### 3. **Sınıf Dengesi**
Her sınıftan yeterli örnek olduğundan emin ol:
```python
# İdeal: Her sınıftan en az 500-1000 görüntü
# Minimum: Her sınıftan en az 200-300 görüntü
```

### 4. **Train/Val/Test Split**
Birleştirmeden SONRA split yap:
```python
# YANLIŞ: Her dataset'i ayrı ayrı split yap
# DOĞRU: Tümünü birleştir, sonra split yap
```

### 5. **Preprocessing Tutarlılığı**
Tüm görüntülere aynı preprocessing uygula:
- Resize (aynı boyut)
- Normalization (aynı aralık)
- Augmentation (aynı teknikler)

---

## 🔧 Uygulama Adımları

### Adım 1: Veri Setlerini Topla
```
datasets/
  bone/
    tumor/
      mendeley_tumor_dataset/
        img1.png
        img2.png
        ...
    fracture/
      kaggle_fracture_dataset/
        img1.jpg
        img2.jpg
        ...
    infection/
      mendeley_infection_dataset/
        img1.dcm
        img2.dcm
        ...
    normal/
      mura_normal/
        img1.png
        ...
```

### Adım 2: Format Standardizasyonu
- DICOM → PNG dönüştür
- Tüm görüntüleri aynı boyuta getir
- RGB formatına çevir

### Adım 3: Birleştirme ve Organizasyon
```python
# organize_combined_bone_data.py
# - Tüm dataset'leri oku
# - Sınıf isimlerini standardize et
# - Birleştir
# - Train/Val/Test split yap
```

### Adım 4: Model Eğitimi
```python
# train_combined_bone_disease.py
# - Birleştirilmiş dataset'i kullan
# - Multi-class classification
# - Transfer learning
```

---

## 📊 Örnek Veri Seti Yapısı

### Senaryo: 5 Sınıf Modeli

#### Veri Kaynakları:
1. **Tumor:**
   - Mendeley: "Bone Tumor X-Ray Dataset" (500 görüntü)
   - Kaggle: "Osteosarcoma Detection" (300 görüntü)
   - **Toplam: 800 görüntü**

2. **Fracture:**
   - Kaggle: "Bone Fracture Classification" (1000 görüntü)
   - Mendeley: "Fracture Types Dataset" (400 görüntü)
   - **Toplam: 1400 görüntü**

3. **Infection:**
   - Mendeley: "Osteomyelitis X-Ray" (600 görüntü)
   - Kaggle: "Bone Infection Dataset" (200 görüntü)
   - **Toplam: 800 görüntü**

4. **Osteoporosis:**
   - Mendeley: "Osteoporosis Detection" (700 görüntü)
   - **Toplam: 700 görüntü**

5. **Normal:**
   - MURA: Normal subset (1500 görüntü)
   - Kaggle: "Normal Bone X-Ray" (500 görüntü)
   - **Toplam: 2000 görüntü**

**TÜM VERİ SETİ: ~5700 görüntü**

### Split:
- **Train:** 4560 görüntü (80%)
- **Validation:** 570 görüntü (10%)
- **Test:** 570 görüntü (10%)

---

## 🎯 Avantajlar

1. **Daha Büyük Veri Seti**
   - Tek kaynak: 1000-2000 görüntü
   - Birleştirilmiş: 5000-10000 görüntü

2. **Daha İyi Sınıf Dengesi**
   - Her sınıftan yeterli örnek
   - Eksik sınıfları tamamlayabilirsin

3. **Daha Fazla Çeşitlilik**
   - Farklı kaynaklardan gelen görüntüler
   - Daha genel bir model

4. **Esneklik**
   - Yeni sınıf eklemek kolay
   - Eksik veriyi tamamlamak kolay

---

## ⚠️ Potansiyel Sorunlar ve Çözümler

### Sorun 1: Format Farklılıkları
**Çözüm:** Ön işleme scripti yaz
```python
def standardize_image(image_path):
    # DICOM okuyup PNG'ye çevir
    # Resize yap
    # RGB'ye çevir
    # Normalize et
    return standardized_image
```

### Sorun 2: Farklı Etiketleme Sistemleri
**Çözüm:** Mapping dictionary kullan
```python
CLASS_MAPPING = {
    'tumor': 'Tumor',
    'osteosarcoma': 'Tumor',
    'bone_cancer': 'Tumor',
    # ...
}
```

### Sorun 3: Sınıf Dengesizliği
**Çözüm:** 
- Oversampling (az örnekli sınıfları çoğalt)
- Class weights kullan
- Minimum threshold belirle (örn: 200 görüntü)

### Sorun 4: Farklı Çözünürlükler
**Çözüm:** Tüm görüntüleri aynı boyuta getir
```python
target_size = (256, 256)  # veya (512, 512)
```

### Sorun 5: Veri Kalitesi Farklılıkları
**Çözüm:** 
- Kalite kontrolü ekle
- Düşük kaliteli görüntüleri filtrele
- Minimum çözünürlük threshold'u

---

## 🚀 Hazır Script Yapısı

### 1. `download_bone_datasets.py`
- Mendeley'den indir
- Kaggle'dan indir
- Dizinlere yerleştir

### 2. `standardize_bone_formats.py`
- DICOM → PNG
- Resize
- Format dönüşümü

### 3. `organize_combined_bone_data.py`
- Sınıf mapping
- Birleştirme
- Train/Val/Test split
- Final organizasyon

### 4. `train_combined_bone_disease.py`
- Birleştirilmiş dataset ile eğitim
- Multi-class classification
- Transfer learning

---

## ✅ Özet

**Evet, kesinlikle yapılabilir ve önerilir!**

**Yapılacaklar:**
1. ✅ Her hastalık için ayrı dataset topla
2. ✅ Formatları standardize et
3. ✅ Sınıf isimlerini birleştir
4. ✅ Birleştir ve split yap
5. ✅ Model eğit

**Avantajlar:**
- Daha büyük veri seti
- Daha iyi sınıf dengesi
- Daha genel model

**Dikkat Edilmesi Gerekenler:**
- Format standardizasyonu
- Sınıf isimlendirme tutarlılığı
- Preprocessing tutarlılığı
- Sınıf dengesi

---

## 🎯 Sonraki Adımlar

Hangi veri setlerini toplamak istersin? Listeyi belirlersen:
1. İndirme scriptlerini hazırlarım
2. Birleştirme ve organizasyon scriptini yazarım
3. Eğitim scriptini hazırlarım

**Hazırım! 🚀**

