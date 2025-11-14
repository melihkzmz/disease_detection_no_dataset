# 🫁 Akciğer Hastalıkları Tespit Sistemi

## 📊 Proje Özeti

Bu sistem, akciğer X-Ray görüntülerinden COVID-19, Pnömoni ve Normal akciğer durumlarını tespit eden bir yapay zeka modelidir.

### Model Performansı
- **Test Accuracy:** 85.14%
- **Model Mimarisi:** MobileNetV2 (Transfer Learning)
- **Eğitim Süresi:** 2 saat 23 dakika
- **Sınıf Sayısı:** 3

---

## 🎯 Sınıflar

1. **COVID-19** - Koronavirüs enfeksiyonu
2. **Non-COVID (Pnömoni)** - Diğer pnömoni türleri
3. **Normal** - Sağlıklı akciğer

---

## 📁 Dosya Yapısı

```
Skin-Disease-Classifier/
│
├── models/
│   ├── lung_disease_model.keras          # Eğitilmiş model
│   └── training_history_lung.png         # Eğitim grafikleri
│
├── datasets/
│   ├── Lung Segmentation Data/           # COVID-QU-Ex Dataset
│   │   ├── Train/                        # 9,052 eğitim görüntüsü
│   │   ├── Val/                          # 5,417 validasyon görüntüsü
│   │   └── Test/                         # 6,573 test görüntüsü
│   │
│   └── Infection Segmentation Data/      # Alternatif dataset
│
├── train_lung_disease.py                 # Model eğitim scripti
├── lung_disease_api.py                   # Flask API servisi
└── LUNG_DISEASE_README.md                # Bu dosya
```

---

## 🚀 Kullanım

### 1. Flask API'yi Başlatma

```bash
cd Skin-Disease-Classifier
python lung_disease_api.py
```

API şu adreste çalışacak: `http://localhost:5000`

### 2. Web Arayüzü

Tarayıcınızda açın:
```
http://localhost:5000/web
```

**Özellikler:**
- ✅ Drag & Drop ile görüntü yükleme
- ✅ Anlık tahmin sonuçları
- ✅ Tüm sınıflar için güven skorları
- ✅ Modern ve kullanıcı dostu arayüz

### 3. API Endpoint'leri

#### GET `/` - API Durumu
```bash
curl http://localhost:5000/
```

**Yanıt:**
```json
{
  "status": "OK",
  "message": "Akciger Hastaliklari Tespit API",
  "version": "1.0",
  "model": "MobileNetV2",
  "accuracy": "85.14%",
  "classes": ["COVID-19", "Non-COVID (Pnomoni)", "Normal"]
}
```

#### POST `/predict` - Tahmin
```bash
curl -X POST -F "image=@xray.jpg" http://localhost:5000/predict
```

**Yanıt:**
```json
{
  "success": true,
  "prediction": "COVID-19",
  "confidence": "92.34%",
  "all_predictions": [
    {"class": "COVID-19", "confidence": 0.9234, "percentage": "92.34%"},
    {"class": "Non-COVID (Pnomoni)", "confidence": 0.0612, "percentage": "6.12%"},
    {"class": "Normal", "confidence": 0.0154, "percentage": "1.54%"}
  ]
}
```

---

## 🔧 Model Eğitimi

Modeli yeniden eğitmek için:

```bash
cd Skin-Disease-Classifier
python train_lung_disease.py
```

### Eğitim Parametreleri

- **Input Size:** 224x224 RGB
- **Batch Size:** 32
- **Epochs:** 30 (Early stopping ile)
- **Learning Rate:** 0.0005
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

### Data Augmentation

Eğitim sırasında kullanılan augmentasyonlar:
- Rotasyon (±20°)
- Yatay/Dikey kaydırma (±20%)
- Yatay çevirme
- Zoom (±20%)

---

## 📈 Eğitim Sonuçları

### Model Metrikleri

| Metric | Değer |
|--------|-------|
| Test Accuracy | 85.14% |
| Test Loss | 0.4264 |
| Validation Accuracy | 82.65% |
| Training Time | 2h 23m |

### Dataset İstatistikleri

| Split | COVID-19 | Non-COVID | Normal | Toplam |
|-------|----------|-----------|--------|--------|
| Train | 4,005 | 1,495 | 3,552 | 9,052 |
| Val | 1,903 | 1,802 | 1,712 | 5,417 |
| Test | 2,180 | 2,253 | 2,140 | 6,573 |
| **TOPLAM** | **8,088** | **5,550** | **7,404** | **21,042** |

---

## 🛠️ Teknik Detaylar

### Gereksinimler

```
tensorflow>=2.20.0
keras>=3.12.0
flask>=3.0.0
pillow>=10.0.0
numpy>=1.26.0
```

### Model Mimarisi

```
MobileNetV2 (Base Model)
    ↓
Global Average Pooling 2D
    ↓
Dropout (0.5)
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (3, Softmax)
```

### Transfer Learning Stratejisi

1. **İlk Aşama:** MobileNetV2 katmanları donduruldu
2. **Fine-tuning:** Sadece üst katmanlar eğitildi
3. **Early Stopping:** Validation accuracy 10 epoch gelişmeyince durdu
4. **Learning Rate Reduction:** Val loss platoya ulaşınca LR yarıya indi

---

## 📊 Dataset Bilgisi

### COVID-QU-Ex Dataset

**Kaynak:** Kaggle - [Lung Segmentation Data](https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset)

**Özellikler:**
- ✅ Yüksek kaliteli X-Ray görüntüleri
- ✅ Dengeli sınıf dağılımı
- ✅ Professional annotasyonlar
- ✅ PNG format (224x224 recommended)

**Kullanım:**
Dataset, akciğer segmentasyon maskeleri ile birlikte gelir, ancak bu projede sadece `images` klasörleri kullanılmıştır.

---

## 🎯 Gelecek Geliştirmeler

- [ ] Daha fazla hastalık sınıfı ekleme (Tüberküloz, Akciğer Kanseri, Pnömotoraks)
- [ ] Grad-CAM ile görselleştirme
- [ ] Model ensemble (birden fazla model kombinasyonu)
- [ ] TensorFlow.js'e dönüştürme (tarayıcıda çalıştırma)
- [ ] Mobile app entegrasyonu
- [ ] DICOM format desteği

---

## ⚠️ Önemli Notlar

### Medikal Kullanım Uyarısı

⚠️ **Bu sistem bir eğitim/araştırma projesidir ve klinik tanı amaçlı kullanılmamalıdır.**

- Sonuçlar yalnızca bilgilendirme amaçlıdır
- Profesyonel tıbbi danışmanlık yerine geçmez
- Tanı için mutlaka uzman bir doktora başvurun

### Kullanım Sınırlamaları

- Model sadece akciğer X-Ray görüntüleri için eğitilmiştir
- Farklı cihazlardan alınan görüntülerde performans değişebilir
- Düşük kaliteli veya çok farklı açılardan çekilmiş görüntülerde hata payı artar

---

## 📞 Destek & İletişim

Sorularınız veya önerileriniz için:
- GitHub Issues açın
- Pull Request gönderin
- Dokumentasyonu geliştirin

---

## 📝 Lisans

Bu proje eğitim amaçlıdır ve açık kaynak olarak sunulmaktadır.

---

## 🙏 Teşekkürler

- **COVID-QU-Ex Dataset** sağlayıcılarına
- **TensorFlow/Keras** ekibine
- **MobileNetV2** mimarisi geliştiricilerine
- Tüm açık kaynak topluluğuna

---

**Son Güncelleme:** 29 Ekim 2025  
**Versiyon:** 1.0  
**Model Accuracy:** 85.14%

