# Kemik Hastalığı Tespiti - Veri Setleri

## 🦴 X-Ray Kemik Hastalığı Veri Setleri

### 1. **MURA (Musculoskeletal Radiographs) Dataset** ⭐ ÖNERİLEN
- **Kaynak:** Stanford ML Group
- **İçerik:** ~40,000 kemik X-ray görüntüsü (el, parmak, el bileği, ön kol, omuz, humerus, dirsek)
- **Hastalıklar:** Kırıklar, anormallikler, normal görüntüler
- **Link:** https://stanfordmlgroup.github.io/competitions/mura/
- **Format:** DICOM ve PNG
- **Avantajlar:** 
  - Büyük ve dengeli veri seti
  - Standart bir benchmark
  - Çok sınıflı sınıflandırma için uygun

---

### 2. **NIH Chest X-ray Dataset** (Göğüs X-Ray ama kemik anormallikleri içerir)
- **Kaynak:** NIH Clinical Center
- **İçerik:** ~112,000 göğüs X-ray görüntüsü
- **İlgili Etiketler:** Kemik yoğunluğu, kemik anormallikleri
- **Link:** https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
- **Format:** PNG
- **Avantajlar:** 
  - Çok büyük veri seti
  - Çoklu hastalık etiketleri

---

### 3. **Bone Age Assessment Dataset**
- **Kaynak:** RSNA (Radiological Society of North America)
- **İçerik:** El bileği X-ray'leri, kemik yaşı tahmini
- **Link:** https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge
- **Format:** DICOM
- **Avantajlar:** 
  - Spesifik bir görev için optimize edilmiş
  - Tıbbi standartlara uygun

---

### 4. **Osteoporosis Dataset (Mendeley)**
- **Kaynak:** Mendeley Data
- **İçerik:** Osteoporoz tespiti için X-ray görüntüleri
- **Arama:** https://data.mendeley.com/ (arama: "osteoporosis", "bone disease", "bone x-ray")
- **Format:** Genelde PNG/JPG
- **Avantajlar:** 
  - Mendeley'de birçok küçük veri seti var
  - İndirme kolaylığı

---

### 5. **Fracture Detection Datasets (Kaggle)**
- **Kaynak:** Kaggle
- **İçerik:** Kemik kırığı tespiti için veri setleri
- **Linkler:**
  - https://www.kaggle.com/datasets?search=bone+fracture
  - https://www.kaggle.com/datasets?search=x-ray+fracture
- **Format:** Çeşitli
- **Avantajlar:** 
  - Çok sayıda küçük veri seti
  - Hızlı indirme

---

### 6. **PadChest Dataset** (Kemik anormallikleri dahil)
- **Kaynak:** Hospital San Juan (İspanya)
- **İçerik:** ~160,000 göğüs X-ray görüntüsü
- **İlgili Etiketler:** Kemik patolojileri
- **Link:** https://bimcv.cipf.es/bimcv-projects/padchest/
- **Format:** DICOM ve PNG
- **Avantajlar:** 
  - Çok detaylı etiketleme
  - Açık erişim

---

### 7. **Bone Tumor Dataset**
- **Kaynak:** Çeşitli akademik kaynaklar
- **İçerik:** Kemik tümörü tespiti
- **Arama:** 
  - Mendeley Data: "bone tumor x-ray"
  - Kaggle: "bone tumor" veya "osteosarcoma"
- **Format:** Çeşitli

---

## 🎯 Öneriler

### En İyi Seçenekler:

1. **MURA Dataset** (En kapsamlı)
   - Büyük ve iyi organize edilmiş
   - Standart benchmark
   - Çok sayıda sınıf (el, parmak, dirsek, vs.)

2. **Mendeley + Kaggle Kombinasyonu**
   - Birden fazla küçük veri setini birleştir
   - Daha fazla çeşitlilik
   - İndirme kolaylığı

3. **NIH Chest X-ray** (Göğüs kemikleri için)
   - Çok büyük veri seti
   - Göğüs kemik hastalıkları için uygun

---

## 📋 İndirme ve Kurulum Adımları

### MURA Dataset İçin:
```bash
# 1. Stanford ML Group sitesinden indir
# https://stanfordmlgroup.github.io/competitions/mura/

# 2. Veri seti genelde zip formatında
# 3. datasets/bone klasörüne çıkart
```

### Mendeley Dataset İçin:
```bash
# 1. Mendeley Data sitesinden seçilen veri setini indir
# 2. datasets/bone/mendeley klasörüne yerleştir
```

### Kaggle Dataset İçin:
```bash
# 1. Kaggle CLI kur (gerekirse)
pip install kaggle

# 2. Kaggle API credentials ayarla
# ~/.kaggle/kaggle.json dosyasına token ekle

# 3. Veri setini indir
kaggle datasets download -d [dataset-name] -p datasets/bone/
```

---

## 🔍 Veri Seti Arama İpuçları

### Mendeley Data'da Arama:
- "bone x-ray"
- "osteoporosis detection"
- "bone fracture classification"
- "bone disease x-ray"
- "musculoskeletal x-ray"

### Kaggle'da Arama:
- "bone fracture"
- "x-ray bone"
- "osteoporosis"
- "bone disease"
- "orthopedic x-ray"

---

## 📊 Veri Seti Seçim Kriterleri

✅ **Önerilen:**
- En az 5,000 görüntü (sınıf başına 500+)
- Dengeli sınıf dağılımı
- DICOM veya PNG formatı
- Açık erişim
- İyi etiketlenmiş

⚠️ **Dikkat Edilmesi Gerekenler:**
- Çok küçük veri setleri (<1000 görüntü)
- Dengesiz sınıf dağılımı
- Düşük çözünürlük
- Eksik etiketler

---

## 🚀 Sonraki Adımlar

1. **Veri Seti Seçimi:** MURA veya Mendeley önerilir
2. **İndirme:** Seçilen veri setini indir
3. **Organizasyon:** `organize_bone_data.py` scripti oluştur
4. **Eğitim:** `train_bone_disease.py` scripti hazırla
5. **Model:** Transfer learning ile EfficientNet veya MobileNet kullan

---

## 📝 Notlar

- **X-Ray görüntüleri:** DICOM formatından PNG'ye dönüştürme gerekebilir
- **Etiketleme:** Bazı veri setleri otomatik etiketlenmiş, bazıları manuel
- **Boyutlandırma:** X-Ray görüntüleri genelde büyük olur (1024x1024+)
- **Preprocessing:** Contrast enhancement gibi teknikler gerekebilir

---

**Hangi veri setini seçmek istersin? MURA en kapsamlı olanı ama daha küçük başlamak istersen Mendeley veya Kaggle'dan birkaç veri setini birleştirebiliriz.**

