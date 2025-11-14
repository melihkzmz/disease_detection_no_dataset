# Kemik Veri Setleri - Detaylı Analiz Raporu

## 📊 Genel Bakış

**İncelenen Veri Setleri:**
1. **Bone Fractures Detection** - Kırık tespiti
2. **Tumor & Normal** - Tümör ve normal görüntüler

---

## 1. BONE FRACTURES DETECTION

### 📁 Dosya Yapısı:
```
Bone Fractures Detection/
├── data.yaml          (Sınıf bilgileri)
├── README.roboflow.txt
├── train/
│   ├── images/       (1347 görüntü)
│   └── labels/       (1347 YOLO label)
├── valid/
│   ├── images/       (128 görüntü)
│   └── labels/       (128 YOLO label)
└── test/
    ├── images/       (64 görüntü)
    └── labels/       (64 YOLO label)
```

### 📋 Sınıflar (10 Sınıf):
1. **Comminuted** - Parçalı kırık
2. **Greenstick** - Yeşil ağaç kırığı
3. **Healthy** - Sağlıklı/Normal
4. **Linear** - Çizgisel kırık
5. **Oblique Displaced** - Eğik yer değiştirmiş
6. **Oblique** - Eğik kırık
7. **Segmental** - Bölümsel kırık
8. **Spiral** - Spiral kırık
9. **Transverse Displaced** - Enine yer değiştirmiş
10. **Transverse** - Enine kırık

### 📊 Veri Dağılımı:
- **Train:** 1,347 görüntü
- **Validation:** 128 görüntü
- **Test:** 64 görüntü
- **Toplam:** 1,539 görüntü

### 🔧 Format:
- **Görüntü formatı:** JPG
- **Label formatı:** YOLO (object detection)
- **Boyutlandırma:** 640x640 (stretch)

### ⚠️ Notlar:
- **Object Detection** formatında (sınıflandırma değil!)
- Train/Val/Test split hazır
- YOLO formatından image classification'a dönüştürme gerekebilir

---

## 2. TUMOR & NORMAL

### 📁 Dosya Yapısı:
```
Tumor & Normal/
├── dataset.xlsx           (540.47 KB - Detaylı bilgi)
├── ~$dataset(total).xlsx  (Geçici Excel dosyası)
├── images/                (3,746 görüntü - JPEG)
└── Annotations/           (1,867 JSON annotation)
```

### 📊 Veri İstatistikleri:
- **Toplam görüntü:** 3,746 adet
- **Annotation sayısı:** 1,867 JSON dosyası
- **Format:** JPEG, JPG
- **Annotation format:** LabelMe (JSON)

### 🏷️ Bulunan Label'lar (İlk 100 dosya analizi):
- **osteosarcoma:** 178 örnek
- **other mt:** 24 örnek
- **other bt:** (bulundu)
- **simple bone cyst:** (bulundu)
- **multiple osteochondromas:** (bulundu)
- **other mt, other bt:** (farklı alt kategoriler)

### 📋 JSON Annotation Yapısı:
```json
{
  "version": "5.4.1",
  "shapes": [
    {
      "label": "osteosarcoma",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "rectangle" veya "polygon"
    }
  ],
  "imagePath": "IMG000001.jpeg",
  "imageHeight": 1200,
  "imageWidth": 768
}
```

### 📄 Excel Dosyası (dataset.xlsx):
- **Boyut:** 540.47 KB
- **İçerik:** Muhtemelen görüntü-metadata eşleşmesi
- **Not:** Pandas/openpyxl ile okunması gerekiyor

---

## 🔍 Detaylı Label Analizi

### JSON Annotation'larda Bulunan Label Tipleri:
1. **osteosarcoma** - Osteosarkom (kemik kanseri)
2. **other mt** - Diğer malign tümörler
3. **other bt** - Diğer benign tümörler
4. **simple bone cyst** - Basit kemik kisti
5. **multiple osteochondromas** - Çoklu osteokondromalar
6. **normal** - Normal görüntüler (muhtemelen Excel'de)

---

## 📋 Excel Dosyası İçeriği (Tahmini)

Excel dosyası muhtemelen şunları içeriyor:
- Görüntü isimleri
- Sınıf etiketleri
- Metadata bilgileri
- Train/Val/Test split bilgisi (muhtemelen)

**⚠️ Pandas ile okunması gerekiyor!**

---

## 🎯 Birleştirme Stratejisi

### Senaryo 1: Tüm Sınıfları Birleştir

**Bone Fractures Dataset:**
- 10 kırık tipi → `Fracture` (genel kategori) veya her birini ayrı tut

**Tumor & Normal Dataset:**
- `osteosarcoma` → `Tumor`
- `other mt` → `Tumor` (malign)
- `other bt` → `Tumor` (benign)
- `simple bone cyst` → `Cyst`
- `multiple osteochondromas` → `Tumor`
- `normal` → `Normal`

### Senaryo 2: Basitleştirilmiş Sınıflar

1. **Normal**
2. **Fracture** (tüm kırık tipleri birleşik)
3. **Tumor** (tüm tümör tipleri birleşik)
4. **Cyst** (kist)

### Senaryo 3: Detaylı Sınıflar (Önerilen)

1. **Normal**
2. **Fracture** (tüm tipler)
3. **Osteosarcoma**
4. **Other_Tumor**
5. **Bone_Cyst**
6. **Osteochondroma**

---

## 🚀 Sonraki Adımlar

### 1. Excel Dosyasını Oku
```python
import pandas as pd
df = pd.read_excel('datasets/bone/Tumor & Normal/dataset.xlsx')
print(df.columns)
print(df.head())
print(df['label'].value_counts())
```

### 2. JSON Annotation'ları Parse Et
- Her görüntü için dominant label'ı belirle
- Image classification için label'ı belirle
- Görüntü-label eşleşmesini oluştur

### 3. YOLO Formatını Dönüştür (İsteğe Bağlı)
- Bone Fractures dataset'i object detection'dan classification'a çevir
- Veya object detection modeli eğit (farklı yaklaşım)

### 4. Veri Setlerini Birleştir
- Tumor & Normal: Classification formatına çevir
- Bone Fractures: Classification formatına çevir (veya ayrı tut)
- Train/Val/Test split yap

### 5. Organizasyon Scripti Yaz
- `organize_combined_bone_data.py`

---

## 📝 Önemli Notlar

### Bone Fractures Dataset:
- ✅ Train/Val/Test split hazır
- ⚠️ YOLO formatı (object detection)
- ⚠️ Classification için dönüştürme gerekli (veya object detection modeli eğit)

### Tumor & Normal Dataset:
- ✅ Çok fazla görüntü (3,746)
- ✅ LabelMe formatı (JSON)
- ⚠️ Excel dosyası okunmalı (metadata)
- ⚠️ Train/Val/Test split yok (oluşturulmalı)
- ⚠️ Annotation'lar object detection için, classification için dominant label belirlenmeli

---

## 🔧 Önerilen Yaklaşım

1. **Excel dosyasını oku** → Görüntü-label eşleşmesini anla
2. **JSON annotation'ları parse et** → Her görüntü için label belirle
3. **Bone Fractures'i classification'a çevir** (veya ayrı model eğit)
4. **Tüm veri setlerini birleştir**
5. **Train/Val/Test split yap** (80/10/10)
6. **Model eğit**

---

## ✅ Hazır Scriptler

Analiz scripti çalıştırıldı: `analyze_bone_datasets.py`

**Sonraki script:** Excel okuma ve organizasyon scriptleri hazırlanacak.

