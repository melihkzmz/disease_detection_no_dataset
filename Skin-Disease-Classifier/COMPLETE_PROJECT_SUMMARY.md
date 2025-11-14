# 🏥 Tıbbi Görüntü Analizi - Çoklu Hastalık Tespit Sistemi

**Yapay Zeka Tabanlı Entegre Sağlık Platformu**

---

## 📊 Proje Genel Bakış

Bu proje, **3 farklı tıbbi görüntü analiz sistemi** içerir:

| # | Sistem | Hastalık Sayısı | Accuracy | Dataset |
|---|--------|----------------|----------|---------|
| 1️⃣ | **Cilt Hastalıkları** | 7 | ~85% | HAM10000 |
| 2️⃣ | **Akciğer Hastalıkları** | 3 | 85.14% | Lung Segmentation |
| 3️⃣ | **Göz Hastalıkları** | 8 | 38.27% (Top-3: 82.69%) | ODIR-5K |

**Toplam:** 18 farklı hastalık tespit edilebilir!

---

## 1️⃣ Cilt Hastalıkları Tespit Sistemi

### 🎯 Özellikler
- **Teknoloji:** TensorFlow.js (In-browser ML)
- **Model:** MobileNetV2
- **Dataset:** HAM10000 (10,000+ dermoskopik görüntü)
- **Port:** Statik web (Python HTTP server: 8000)

### 📋 Tespit Edilen Hastalıklar
1. Actinic Keratoses (Aktiniktik keratoz)
2. Basal Cell Carcinoma (Bazal hücreli karsinom)
3. Benign Keratosis (İyi huylu keratoz)
4. Dermatofibroma (Dermatofibrom)
5. Melanoma (Melanom)
6. Melanocytic Nevi (Melanositik nevus)
7. Vascular Lesions (Vasküler lezyonlar)

### 🚀 Nasıl Çalıştırılır?
```bash
cd Skin-Disease-Classifier
python -m http.server 8000
# Tarayıcı: http://localhost:8000
```

### 📁 Dosyalar
- `index.html` - Ana web sayfası
- `jscript/` - TensorFlow.js model ve prediction kodları
- `final_model_kaggle_version1/` - TensorFlow.js modeli

---

## 2️⃣ Akciğer Hastalıkları Tespit Sistemi

### 🎯 Özellikler
- **Teknoloji:** Flask API + MobileNetV2
- **Accuracy:** 85.14%
- **Dataset:** Lung Segmentation Data (X-ray görüntüleri)
- **Port:** 5000

### 📋 Tespit Edilen Hastalıklar
1. **COVID-19** - Koronavirüs enfeksiyonu
2. **Non-COVID (Pnomoni)** - Diğer pnömoni türleri
3. **Normal** - Sağlıklı akciğer

### 🚀 Nasıl Çalıştırılır?
```bash
python lung_disease_api.py
# Web Arayüzü: http://localhost:5000/web
# API: http://localhost:5000/predict
```

### 📁 Dosyalar
- `train_lung_disease.py` - Model eğitim scripti
- `lung_disease_api.py` - Flask API
- `test_lung_api.py` - API test scripti
- `models/lung_disease_model.keras` - Eğitilmiş model

### 📊 Performans
- **Test Accuracy:** 85.14%
- **Dataset:** 6,392 X-ray görüntüsü
- **Eğitim Süresi:** ~30 dakika

---

## 3️⃣ Göz Hastalıkları Tespit Sistemi

### 🎯 Özellikler
- **Teknoloji:** Flask API + MobileNetV2
- **Accuracy:** 38.27% (Top-3: 82.69%)
- **Dataset:** ODIR-5K (Fundus görüntüleri)
- **Port:** 5002

### 📋 Tespit Edilen Hastalıklar
1. **AMD** - Makula dejenerasyonu
2. **Cataract** - Katarakt
3. **Diabetes** - Diabetik retinopati
4. **Glaucoma** - Glokom
5. **Hypertension** - Hipertansif retinopati
6. **Myopia** - Miyopi
7. **Normal** - Sağlıklı göz
8. **Other** - Diğer göz hastalıkları

### 🚀 Nasıl Çalıştırılır?
```bash
python eye_disease_api.py
# Web Arayüzü: http://localhost:5002/web
# API: http://localhost:5002/predict
```

### 📁 Dosyalar
- `organize_eye_data.py` - Veri organizasyon scripti
- `train_eye_disease.py` - Model eğitim scripti
- `eye_disease_api.py` - Flask API
- `test_eye_api.py` - API test scripti
- `models/eye_disease_model.keras` - Eğitilmiş model

### 📊 Performans
- **Test Accuracy:** 38.27%
- **Top-3 Accuracy:** 82.69%
- **Dataset:** 6,392 fundus görüntüsü
- **Eğitim Süresi:** ~51 dakika

---

## 🏗️ Proje Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│                  FRONTEND (Web Browsers)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Cilt     │  │ Akciğer  │  │ Göz      │             │
│  │ (HTML/JS)│  │ (HTML)   │  │ (HTML)   │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
└───────┼─────────────┼─────────────┼────────────────────┘
        │             │             │
        │             │             │
┌───────▼─────────────▼─────────────▼────────────────────┐
│                    BACKEND LAYER                        │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │TensorFlow│  │  Flask   │  │  Flask   │             │
│  │   .js    │  │   API    │  │   API    │             │
│  │ (Client) │  │ :5000    │  │ :5002    │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
└───────┼─────────────┼─────────────┼────────────────────┘
        │             │             │
        │             │             │
┌───────▼─────────────▼─────────────▼────────────────────┐
│                    MODEL LAYER                          │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │MobileNet │  │MobileNet │  │MobileNet │             │
│  │   V2     │  │   V2     │  │   V2     │             │
│  │(TF.js)   │  │(Keras)   │  │(Keras)   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Gereksinimler

### Python Paketleri
```bash
pip install tensorflow pillow flask numpy matplotlib scikit-learn
```

### Dosya Boyutları
```
models/
├── lung_disease_model.keras     (~9 MB)
├── eye_disease_model.keras      (~9 MB)
└── final_model_kaggle_version1/ (~9 MB)

datasets/
├── HAM10000/                    (~1.5 GB)
├── Lung Segmentation Data/      (~2 GB)
└── Eye_Organized/               (~1.2 GB)
```

---

## 🚀 Hızlı Başlangıç

### Tüm Sistemleri Çalıştırma

#### Terminal 1: Cilt Hastalıkları
```bash
cd Skin-Disease-Classifier
python -m http.server 8000
```

#### Terminal 2: Akciğer Hastalıkları
```bash
python lung_disease_api.py
```

#### Terminal 3: Göz Hastalıkları
```bash
python eye_disease_api.py
```

### Erişim URL'leri
- **Cilt:** http://localhost:8000
- **Akciğer:** http://localhost:5000/web
- **Göz:** http://localhost:5002/web

---

## 📊 Karşılaştırmalı Analiz

### Model Performansı
| Sistem | Accuracy | Top-3 | Dataset Size | Training Time |
|--------|----------|-------|--------------|---------------|
| Cilt | ~85% | - | 10,000+ | Pre-trained |
| Akciğer | 85.14% | - | 6,392 | ~30 min |
| Göz | 38.27% | 82.69% | 6,392 | ~51 min |

### Teknoloji Stack
| Sistem | Framework | Deployment | Model Format |
|--------|-----------|------------|--------------|
| Cilt | TensorFlow.js | Client-side | TF.js |
| Akciğer | Flask + TF | Server-side | Keras |
| Göz | Flask + TF | Server-side | Keras |

### Görüntü Türleri
| Sistem | Görüntü Tipi | Çözünürlük |
|--------|--------------|-----------|
| Cilt | Dermoskopik | Değişken |
| Akciğer | X-Ray (Göğüs) | 224x224 |
| Göz | Fundus (Retina) | 224x224 |

---

## 🔌 API Kullanımı

### Cilt Hastalıkları (Client-side)
```javascript
// TensorFlow.js ile tarayıcıda çalışır
const model = await tf.loadGraphModel('final_model_kaggle_version1/model.json');
const prediction = model.predict(imageData);
```

### Akciğer Hastalıkları (REST API)
```bash
curl -X POST -F "image=@xray.jpg" http://localhost:5000/predict
```

### Göz Hastalıkları (REST API)
```bash
curl -X POST -F "image=@fundus.jpg" http://localhost:5002/predict
```

---

## 📈 Eğitim Grafikleri

Tüm modeller için eğitim grafikleri kaydedildi:
- `models/training_history_lung.png` (Akciğer)
- `models/training_history_eye.png` (Göz)

Her grafik şunları içerir:
1. Training vs Validation Accuracy
2. Training vs Validation Loss
3. Top-K Accuracy (varsa)

---

## 🔧 Model Eğitim Parametreleri

### Ortak Özellikler
- **Base Model:** MobileNetV2 (ImageNet pretrained)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Data Augmentation:** ✅

### Farklılıklar
| Parametre | Akciğer | Göz |
|-----------|---------|-----|
| Batch Size | 32 | 32 |
| Learning Rate | 0.001 | 0.001 |
| Epochs (max) | 50 | 50 |
| Dense Layers | 256 | 512→256 |
| Dropout | 0.5, 0.4, 0.3 | 0.5, 0.4, 0.3 |

---

## ⚠️ Önemli Notlar

### Klinik Kullanım
🚨 **DİKKAT:** Bu modeller sadece araştırma ve eğitim amaçlıdır!

- ✅ Ön tanı desteği olarak kullanılabilir
- ✅ Tarama programlarında yardımcı olabilir
- ❌ Tek başına teşhis aracı OLAMAZ
- ❌ Klinik kararlarda mutlaka uzman doktor onayı gerekir
- ⚠️ FDA/CE onayı yoktur

### Veri Gizliliği
- Hasta verileri saklanmaz
- Sadece lokal prediction
- GDPR/HIPAA uyumlu değildir (production için uyarlanmalı)

### Model Limitasyonları
- **Cilt:** HAM10000 ile sınırlı hastalıklar
- **Akciğer:** COVID-19 erken dönem tespitinde yetersiz olabilir
- **Göz:** Düşük accuracy (%38) - uzman onayı şart

---

## 🚀 Gelecek İyileştirmeler

### Teknik İyileştirmeler
- [ ] Fine-tuning ile accuracy artırma
- [ ] Ensemble models
- [ ] Grad-CAM görselleştirme
- [ ] Model compression (TFLite)
- [ ] Batch prediction desteği

### Deployment
- [ ] Docker containerization
- [ ] Kubernetes orchestration
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Load balancer ekle
- [ ] CI/CD pipeline

### Yeni Özellikler
- [ ] Multi-model ensemble API
- [ ] Report generation (PDF)
- [ ] Patient history tracking
- [ ] Mobile app (React Native)
- [ ] Real-time video analysis

---

## 📚 Kullanılan Teknolojiler

### Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5
- TensorFlow.js

### Backend
- Python 3.10+
- Flask (REST API)
- TensorFlow/Keras 2.x

### ML/DL
- MobileNetV2 (Transfer Learning)
- ImageNet pretrained weights
- Data Augmentation
- Class Weighting

### Tools
- Jupyter Notebooks
- Matplotlib (visualization)
- Pillow (image processing)
- NumPy, Pandas

---

## 📖 Referanslar

### Datasets
1. **HAM10000:** Human Against Machine with 10000 training images
2. **Lung Segmentation:** COVID-19 + Pneumonia X-Ray
3. **ODIR-5K:** Ocular Disease Intelligent Recognition

### Papers
- MobileNetV2: [arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- Transfer Learning: [cs231n.github.io/transfer-learning](https://cs231n.github.io/transfer-learning/)

### Frameworks
- TensorFlow: [tensorflow.org](https://tensorflow.org)
- Flask: [flask.palletsprojects.com](https://flask.palletsprojects.com)

---

## 👥 Kullanım Senaryoları

### 1️⃣ Sağlık Kurumları
- İlk tanı desteği
- Tarama programları
- Yük azaltma (önceliklendirme)

### 2️⃣ Telemedisin
- Uzaktan danışmanlık
- Kırsal bölgelerde sağlık hizmeti
- Home healthcare

### 3️⃣ Araştırma
- Tıbbi görüntü analizi
- Deep learning studies
- Dataset curation

### 4️⃣ Eğitim
- Tıp öğrencileri için
- Radyoloji eğitimi
- AI in healthcare courses

---

## 📊 İstatistikler

### Toplam Proje
- **Toplam Hastalık:** 18 farklı hastalık
- **Toplam Model:** 3 ayrı model
- **Toplam Dataset:** ~20,000+ görüntü
- **Toplam Kod Satırı:** ~3,000+ satır Python/JS
- **Geliştirme Süresi:** ~1 hafta

### Model Boyutları
- **Toplam Model Boyutu:** ~27 MB
- **Dataset Boyutu:** ~4.7 GB
- **Dependency Size:** ~2 GB (TensorFlow)

---

## 🎯 Sonuç

Bu proje, **3 farklı tıbbi görüntü analiz sistemini** bir araya getiren kapsamlı bir **AI-powered healthcare platform** prototipidir.

### ✨ Güçlü Yönler
✅ Çoklu hastalık tespiti (18 hastalık)  
✅ Farklı görüntü türleri (dermoskopik, X-ray, fundus)  
✅ Hybrid deployment (client + server)  
✅ Modern web arayüzleri  
✅ RESTful API'ler  
✅ Transfer learning ile hızlı eğitim  

### ⚠️ Limitasyonlar
- Production-ready değil
- Klinik onay yok
- Bazı modellerde düşük accuracy
- Scalability sorunları olabilir

### 🚀 Potansiyel
Bu proje, gerçek dünya healthcare uygulamaları için güçlü bir **proof-of-concept**'tir ve uygun iyileştirmelerle production ortamına taşınabilir.

---

**🎉 Proje Tamamlandı!**

**Cilt:** http://localhost:8000  
**Akciğer:** http://localhost:5000/web  
**Göz:** http://localhost:5002/web  

---

*Sağlıklı günler dileriz! 🏥💚*

