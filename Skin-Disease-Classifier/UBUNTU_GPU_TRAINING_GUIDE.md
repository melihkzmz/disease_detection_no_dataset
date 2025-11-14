# Ubuntu GPU Eğitim Rehberi - Kemik Hastalığı Tespiti

Bu rehber, Ubuntu'da kendi GPU'nuzu kullanarak kemik hastalığı tespiti modelini eğitmeniz için adım adım talimatlar içerir.

---

## 📋 Önkoşullar

### 1. NVIDIA GPU
- NVIDIA GPU (CUDA destekli) gerekli
- Minimum 4GB GPU memory önerilir (512x512 görüntüler için)

### 2. NVIDIA Driver
```bash
# Driver versiyonunu kontrol et
nvidia-smi

# Eğer driver yoksa:
sudo apt update
sudo apt install nvidia-driver-535  # veya en son sürüm
sudo reboot
```

### 3. CUDA Toolkit (11.8+ veya 12.x)
```bash
# CUDA versiyonunu kontrol et
nvcc --version

# CUDA yoksa indir ve kur:
# https://developer.nvidia.com/cuda-downloads
```

### 4. cuDNN (CUDA Deep Neural Network Library)
- CUDA ile birlikte kurulabilir veya ayrı kurulabilir
- TensorFlow için gerekli

### 5. Python 3.8+
```bash
python3 --version
```

---

## 🚀 Hızlı Başlangıç

### Yöntem 1: Hazır Script ile (Önerilen)

```bash
# Script'e çalıştırma izni ver
chmod +x start_training_ubuntu_gpu.sh

# Eğitimi başlat
./start_training_ubuntu_gpu.sh
```

Script otomatik olarak:
- ✅ GPU ve CUDA kontrolü yapar
- ✅ Python paketlerini kontrol eder
- ✅ TensorFlow GPU desteğini doğrular
- ✅ Dataset yapısını kontrol eder
- ✅ Eğitimi başlatır ve loglar

---

### Yöntem 2: Manuel Başlatma

#### Adım 1: Gerekli Paketleri Kur

```bash
# Proje dizinine git
cd Skin-Disease-Classifier

# Python paketlerini kur
pip3 install -r requirements.txt

# Ek paketler (eksik olabilir)
pip3 install numpy matplotlib scikit-learn seaborn pandas openpyxl tensorflow[and-cuda]
```

#### Adım 2: CUDA Environment Variables Ayarla

```bash
# CUDA path'ini ayarla (kendi kurulumunuza göre)
export CUDA_HOME=/usr/local/cuda  # veya /usr/local/cuda-11.8, /usr/local/cuda-12.0, vb.
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Bu ayarları kalıcı yapmak için `~/.bashrc` dosyasına ekleyin:
```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Adım 3: GPU'yu Kontrol Et

```bash
# NVIDIA GPU kontrolü
nvidia-smi

# TensorFlow GPU desteği kontrolü
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

#### Adım 4: Dataset'i Kontrol Et

```bash
# Dataset dizinini kontrol et
ls -la datasets/bone/Bone_4Class_Final/
# train/, val/, test/ klasörleri olmalı
```

#### Adım 5: Eğitimi Başlat

```bash
# Direkt başlat
python3 train_bone_4class_optimized.py

# Log ile birlikte başlat
python3 train_bone_4class_optimized.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔧 Sorun Giderme

### Problem 1: "No GPU detected" hatası

**Çözüm:**
```bash
# 1. Driver kontrolü
nvidia-smi

# 2. CUDA kontrolü
nvcc --version

# 3. TensorFlow GPU build kontrolü
python3 -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# 4. CUDA/cuDNN versiyonları uyumlu mu kontrol et
# TensorFlow 2.x için genellikle CUDA 11.8 veya 12.x gerekir
```

**Yeniden Kurulum:**
```bash
# TensorFlow GPU desteği ile yeniden kur
pip3 uninstall tensorflow
pip3 install tensorflow[and-cuda]
# veya
pip3 install tensorflow-gpu  # Eski versiyonlar için
```

---

### Problem 2: "CUDA out of memory" hatası

**Çözüm:**
1. **Batch size'ı küçült:**
   ```python
   # train_bone_4class_optimized.py dosyasında
   BATCH_SIZE = 8  # 16 yerine 8
   ```

2. **Image size'ı küçült:**
   ```python
   IMG_SIZE = (256, 256)  # 512 yerine 256
   ```

3. **GPU memory growth ayarla (zaten script'te var):**
   ```python
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

---

### Problem 3: "Dataset not found" hatası

**Çözüm:**
```bash
# Dataset organizasyon scriptini çalıştır
python3 organize_bone_4class_final.py

# Dataset dizinini kontrol et
ls -la datasets/bone/Bone_4Class_Final/
```

---

### Problem 4: "ModuleNotFoundError"

**Çözüm:**
```bash
# Eksik paketleri kur
pip3 install numpy matplotlib scikit-learn seaborn pandas openpyxl

# veya requirements.txt'ten kur
pip3 install -r requirements.txt
```

---

### Problem 5: CUDA versiyon uyumsuzluğu

**Kontrol:**
```bash
# CUDA versiyonu
nvcc --version

# TensorFlow'un beklediği CUDA versiyonu
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

**TensorFlow 2.x CUDA Gereksinimleri:**
- TensorFlow 2.13+: CUDA 11.8 veya 12.x
- TensorFlow 2.10-2.12: CUDA 11.8
- TensorFlow 2.9-: CUDA 11.2

**Çözüm:**
- CUDA versiyonunu TensorFlow ile uyumlu hale getir
- Veya TensorFlow versiyonunu CUDA ile uyumlu hale getir

---

## 📊 Eğitim İzleme

### GPU Kullanımını İzle

Yeni bir terminal açın:
```bash
# Sürekli GPU izleme
watch -n 1 nvidia-smi

# veya
while true; do clear; nvidia-smi; sleep 1; done
```

### Eğitim Loglarını İzle

```bash
# Log dosyasını takip et
tail -f training_logs/bone_4class_training_*.log
```

---

## ⚙️ İleri Seviye Ayarlar

### Mixed Precision Training (Hız Artışı)

`train_bone_4class_optimized.py` dosyasında şu satırı aktifleştir:
```python
# Satır 43-45'i uncomment et
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("[GPU] Mixed precision enabled")
```

**Fayda:** ~%50 daha hızlı eğitim, aynı accuracy

---

### Multi-GPU Training (Birden Fazla GPU)

```python
# train_bone_4class_optimized.py başına ekle
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    # Model tanımı buraya
```

---

### Conda Environment Kullanımı

```bash
# Conda environment oluştur
conda create -n bone_disease python=3.10
conda activate bone_disease

# CUDA toolkit kur (conda üzerinden)
conda install -c conda-forge cudatoolkit=11.8 cudnn

# TensorFlow kur
pip install tensorflow

# Diğer paketler
pip install -r requirements.txt
```

---

## 📁 Dosya Yapısı

```
Skin-Disease-Classifier/
├── train_bone_4class_optimized.py  # Ana eğitim scripti
├── start_training_ubuntu_gpu.sh    # Otomatik başlatma scripti
├── UBUNTU_GPU_TRAINING_GUIDE.md    # Bu rehber
├── datasets/
│   └── bone/
│       └── Bone_4Class_Final/
│           ├── train/
│           ├── val/
│           └── test/
└── models/
    ├── bone_4class_initial.keras
    ├── bone_4class_finetuned.keras
    └── bone_disease_model_4class.keras  # Final model
```

---

## ⏱️ Beklenen Eğitim Süresi

- **512x512 görüntü boyutu, Batch size 16:**
  - Phase 1 (Initial): ~2-4 saat (100 epochs)
  - Phase 2 (Fine-tuning): ~1-2 saat (50 epochs)
  - **Toplam: ~3-6 saat** (GPU'ya bağlı)

- **256x256 görüntü boyutu, Batch size 32:**
  - Phase 1: ~1-2 saat
  - Phase 2: ~30-60 dakika
  - **Toplam: ~2-3 saat**

---

## ✅ Başarı Kriterleri

Eğitim başarılı sayılır eğer:
- ✅ Model tüm 4 sınıfı tahmin edebiliyor
- ✅ Test accuracy > %70
- ✅ Confusion matrix dengeli
- ✅ Overfitting yok (val accuracy ≈ train accuracy)

---

## 📞 Yardım

Sorun yaşarsanız:
1. Log dosyasını kontrol edin: `training_logs/`
2. GPU durumunu kontrol edin: `nvidia-smi`
3. TensorFlow GPU desteğini test edin:
   ```python
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

---

**İyi eğitimler! 🚀**

