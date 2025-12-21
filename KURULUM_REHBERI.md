# 🚀 Proje Kurulum Rehberi

Bu rehber, projeyi ilk defa bilgisayarınızda çalıştırmak için gereken tüm adımları içerir.

## 📋 Gereksinimler

### 1. Python Kurulumu
- **Python 3.8 veya üzeri** gereklidir
- Python'un kurulu olup olmadığını kontrol etmek için terminalde şu komutu çalıştırın:
  ```bash
  python --version
  ```
- Eğer Python yoksa: [Python İndirme Sayfası](https://www.python.org/downloads/)

### 2. Gerekli Python Paketleri

Proje klasörüne gidin ve gerekli paketleri yükleyin:

```bash
cd Skin-Disease-Classifier
pip install -r requirements.txt
```

**Not:** Eğer `pip` komutu çalışmazsa, `pip3` veya `python -m pip` deneyin.

### 3. Model Dosyası Kontrolü

Kemik hastalıkları analizi için model dosyasının mevcut olması gereklidir. API şu sırayla model dosyasını arar:

1. `models/bone_disease_model_4class_densenet121_macro_f1_savedmodel/` (SavedModel formatı - önerilen)
2. `models/bone_disease_model_4class_densenet121_macro_f1.keras` (Keras formatı)

Model dosyası yoksa, API çalışmayacaktır. Model dosyaları GitHub'da mevcut olmalıdır.

## 🏃 Projeyi Çalıştırma

### Adım 1: Backend API'yi Başlatın

Terminal/PowerShell'de proje klasörüne gidin:

```bash
cd Skin-Disease-Classifier
```

Ardından backend API'yi başlatın:

```bash
python bone_disease_api.py
```

**Başarılı başlatma çıktısı:**
```
======================================================================
KEMIK HASTALIKLARI TESPIT API
======================================================================

[YUKLENIYOR] Model: models/bone_disease_model_4class_densenet121_macro_f1_savedmodel
[BASARILI] Model yuklendi!
[SERVER] Calisiyor: http://localhost:5002
```

**Önemli:** API'nin çalıştığını görmek için terminal penceresini açık tutun!

### Adım 2: Frontend Web Sunucusunu Başlatın

**Yeni bir terminal/PowerShell penceresi açın** ve şu komutu çalıştırın:

**Windows için:**
```bash
cd Skin-Disease-Classifier
start_server.bat
```

**Veya manuel olarak:**
```bash
cd Skin-Disease-Classifier
python -m http.server 8000
```

### Adım 3: Web Arayüzünü Açın

Tarayıcınızda şu adresi açın:
```
http://localhost:8000/analyze.html
```

## ✅ Kontrol Listesi

Projeyi çalıştırmadan önce:

- [ ] Python 3.8+ kurulu
- [ ] `pip install -r requirements.txt` komutu başarıyla çalıştı
- [ ] Model dosyası mevcut (`models/` klasöründe)
- [ ] Backend API çalışıyor (`http://localhost:5002`)
- [ ] Frontend sunucusu çalışıyor (`http://localhost:8000`)

## 🔧 Sorun Giderme

### Problem: "Model yüklenemedi" hatası

**Çözüm:**
- Model dosyasının `models/` klasöründe olduğundan emin olun
- Dosya yollarını kontrol edin
- Model dosyasının bozuk olmadığından emin olun

### Problem: "ModuleNotFoundError" hatası

**Çözüm:**
```bash
pip install -r requirements.txt
```

Eksik paketleri tek tek yükleyin:
```bash
pip install Flask flask-cors tensorflow opencv-python Pillow numpy
```

### Problem: "Port 5002 already in use" hatası

**Çözüm:**
- Başka bir program 5002 portunu kullanıyor olabilir
- `bone_disease_api.py` dosyasındaki port numarasını değiştirebilirsiniz (satır 630)
- Veya o portu kullanan programı kapatın

### Problem: Frontend API'ye bağlanamıyor

**Çözüm:**
- Backend API'nin çalıştığından emin olun (`http://localhost:5002`)
- Tarayıcı konsolunda (F12) hata mesajlarını kontrol edin
- CORS hatası alıyorsanız, `flask-cors` paketinin yüklü olduğundan emin olun

### Problem: "OpenCV (cv2) bulunamadı" uyarısı

**Çözüm:**
```bash
pip install opencv-python
```

Bu uyarı kritik değildir, ancak CLAHE özelliği devre dışı kalır.

## 📝 Kullanım

1. Web arayüzünde **"Hastalık Türü"** olarak **"Kemik Hastalıkları"** seçin
2. Bir görüntü dosyası yükleyin (JPG, PNG, vb.)
3. **"Analiz Et"** butonuna tıklayın
4. Sonuçlar ekranda görünecektir

## 🎯 API Endpoint'leri

Backend API şu endpoint'leri sağlar:

- `GET http://localhost:5002/` - API durumu
- `POST http://localhost:5002/predict` - Görüntü analizi
- `GET http://localhost:5002/classes` - Tüm sınıfları listele

## 📦 Gerekli Paketler Listesi

- `tensorflow` - Makine öğrenmesi modeli
- `Flask` - Web API framework
- `flask-cors` - CORS desteği
- `opencv-python` - Görüntü işleme (CLAHE için)
- `Pillow` - Görüntü işleme
- `numpy` - Sayısal hesaplamalar
- `scikit-learn` - Makine öğrenmesi yardımcıları
- `pandas` - Veri işleme
- `matplotlib` - Görselleştirme

## 💡 İpuçları

- Backend API'yi ve frontend sunucusunu **ayrı terminal pencerelerinde** çalıştırın
- Model yükleme ilk başta biraz zaman alabilir (özellikle büyük modeller için)
- Grad-CAM görselleştirmesi için OpenCV önerilir
- Windows'ta UTF-8 karakter desteği için Python 3.7+ gereklidir

## 🆘 Yardım

Sorun yaşıyorsanız:
1. Terminal çıktılarını kontrol edin
2. Tarayıcı konsolunu açın (F12)
3. Model dosyasının varlığını kontrol edin
4. Tüm paketlerin yüklü olduğundan emin olun

---

**Not:** Bu proje araştırma ve eğitim amaçlıdır. Klinik tanı için kullanılmamalıdır.

