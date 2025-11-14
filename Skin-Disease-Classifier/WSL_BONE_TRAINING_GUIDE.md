# WSL GPU Eğitim Rehberi - Kemik Hastalığı Tespiti

Bu rehber, **Windows Subsystem for Linux (WSL2)** üzerinde kendi GPU'nuzu kullanarak kemik hastalığı tespiti modelini eğitmeniz için adım adım talimatlar içerir.

---

## 📋 Önkoşullar

### 1. Windows'ta NVIDIA Driver
- Windows'ta NVIDIA driver kurulu olmalı
- WSL2 için GPU desteği etkin olmalı
- Windows 11 veya Windows 10 (May 2020 Update+) önerilir

**Kontrol:**
```powershell
# Windows PowerShell'de
nvidia-smi
```

### 2. WSL2 Kurulumu
```bash
# WSL versiyonunu kontrol et
wsl --version

# WSL2 yoksa kur
wsl --install
```

### 3. CUDA Toolkit (WSL için)
WSL'de CUDA toolkit kurulu olmalı. İki yöntem:

**Yöntem 1: Conda ile (Önerilen)**
```bash
# Conda environment oluştur
conda create -n tf_gpu python=3.10
conda activate tf_gpu

# CUDA toolkit ve cuDNN kur
conda install -c conda-forge cudatoolkit=11.8 cudnn

# TensorFlow kur
pip install tensorflow
```

**Yöntem 2: NVIDIA CUDA Toolkit (WSL için)**
```bash
# NVIDIA'nın resmi CUDA toolkit'ini indir ve kur
# https://developer.nvidia.com/cuda-downloads
# "Linux" > "x86_64" > "WSL-Ubuntu" > "deb (local)"
```

### 4. Conda Environment (Zaten Var)
Mevcut `tf_gpu` environment'ınızı kullanabilirsiniz.

---

## 🚀 Hızlı Başlangıç

### Yöntem 1: Hazır Script ile (Önerilen)

```bash
# WSL Ubuntu'da (bash)

# Script'e çalıştırma izni ver
chmod +x start_training_bone_wsl.sh

# Eğitimi başlat
./start_training_bone_wsl.sh
```

Script otomatik olarak:
- ✅ Conda environment'ı aktive eder (`tf_gpu`)
- ✅ GPU'yu kontrol eder
- ✅ Dataset yapısını doğrular
- ✅ Python paketlerini kontrol eder
- ✅ Eğitimi başlatır ve loglar

---

### Yöntem 2: Manuel Başlatma

```bash
# 1. WSL'de bash aç

# 2. Conda environment'ı aktive et
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# 3. GPU library path ayarla
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 4. Proje dizinine git
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier

# 5. GPU kontrolü
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 6. Eğitimi başlat
python3 train_bone_4class_optimized.py
```

---

## 🔧 WSL Özel Ayarlar

### Windows Dosya Yolu
WSL'de Windows dosyalarına `/mnt/c/` üzerinden erişilir:
```bash
# Windows: C:\Users\melih\dev\disease_detection\...
# WSL:     /mnt/c/Users/melih/dev/disease_detection/...
```

### GPU Erişimi
WSL'de GPU **Windows driver üzerinden** erişilir:
- Windows'ta NVIDIA driver kurulu olmalı
- WSL2 otomatik olarak GPU'yu paylaşır
- `nvidia-smi` komutu WSL'de çalışır

**Kontrol:**
```bash
# WSL'de GPU kontrolü
nvidia-smi
```

---

## ⚙️ Script Detayları

### `start_training_bone_wsl.sh` Scripti

Script şu adımları gerçekleştirir:

1. **Conda Environment Aktifleştirme**
   ```bash
   conda activate tf_gpu
   ```

2. **GPU Library Path Ayarlama**
   ```bash
   export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **GPU Kontrolü**
   - TensorFlow'un GPU'yu görüp görmediğini kontrol eder
   - Hata varsa açıklama yapar

4. **Dataset Kontrolü**
   - Dataset dizinini kontrol eder
   - Train/Val/Test klasörlerini doğrular
   - Görüntü sayılarını gösterir

5. **Python Paket Kontrolü**
   - TensorFlow versiyonunu gösterir
   - Gerekli paketleri kontrol eder

6. **Eğitimi Başlatma**
   - Log dosyası oluşturur
   - Eğitimi başlatır ve loglar

---

## 🔍 Sorun Giderme

### Problem 1: "GPU not detected" hatası

**Çözüm 1: Windows driver kontrolü**
```powershell
# Windows PowerShell'de
nvidia-smi
```

**Çözüm 2: WSL CUDA desteği**
```bash
# WSL'de CUDA kontrolü
nvidia-smi

# TensorFlow GPU kontrolü
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Çözüm 3: LD_LIBRARY_PATH**
```bash
# Script'teki path'i kontrol et
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Conda CUDA path'ini kontrol et
ls -la ~/miniconda3/envs/tf_gpu/lib/libcudart*
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

---

### Problem 3: "Dataset not found" hatası

**Çözüm:**
```bash
# Windows'ta dataset organizasyon scriptini çalıştır
# veya WSL'de Python script'i çalıştır
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
python3 organize_bone_4class_final.py
```

---

### Problem 4: "ModuleNotFoundError"

**Çözüm:**
```bash
# Conda environment'ı aktifleştir
conda activate tf_gpu

# Eksik paketleri kur
pip install numpy matplotlib scikit-learn seaborn pandas openpyxl
```

---

### Problem 5: Yavaş dosya erişimi

WSL'de Windows dosyalarına erişim (`/mnt/c/`) yavaş olabilir.

**Çözüm:**
1. **Dataset'i WSL dosya sistemine kopyala:**
   ```bash
   # WSL'de hızlı erişim için
   mkdir -p ~/datasets/bone
   cp -r /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier/datasets/bone/Bone_4Class_Final ~/datasets/bone/
   
   # Script'teki DATASET_PATH'i değiştir
   ```

2. **Veya Windows'ta organize et, WSL'de sadece eğit:**
   - Dataset Windows'ta organize edilmiş olabilir
   - WSL'de sadece okuma yapılır (eğitim sırasında)

---

## 📊 Eğitim İzleme

### GPU Kullanımını İzle

**WSL'de:**
```bash
# Sürekli GPU izleme
watch -n 1 nvidia-smi
```

**Windows'ta:**
```powershell
# GPU izleme
nvidia-smi -l 1
```

### Eğitim Loglarını İzle

```bash
# WSL'de log takibi
tail -f training_logs/bone_4class_training_*.log
```

---

## 📁 Dosya Yapısı

```
Windows: C:\Users\melih\dev\disease_detection\Skin-Disease-Classifier\
WSL:     /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier/

├── train_bone_4class_optimized.py     # Ana eğitim scripti
├── start_training_bone_wsl.sh          # WSL başlatma scripti
├── WSL_BONE_TRAINING_GUIDE.md         # Bu rehber
├── datasets/
│   └── bone/
│       └── Bone_4Class_Final/
│           ├── train/
│           ├── val/
│           └── test/
└── models/
    ├── bone_4class_initial.keras
    ├── bone_4class_finetuned.keras
    └── bone_disease_model_4class.keras
```

---

## ⏱️ Beklenen Eğitim Süresi

**WSL2 + RTX GPU:**
- **512x512, Batch 16:**
  - Phase 1: ~2-4 saat
  - Phase 2: ~1-2 saat
  - **Toplam: ~3-6 saat**

- **256x256, Batch 32:**
  - Phase 1: ~1-2 saat
  - Phase 2: ~30-60 dakika
  - **Toplam: ~2-3 saat**

**Not:** WSL'de Windows dosyalarına erişim yavaş olabilir, bu süreyi etkileyebilir.

---

## ✅ Başarı Kriterleri

Eğitim başarılı sayılır eğer:
- ✅ GPU kullanılıyor (nvidia-smi'de %100 kullanım görülür)
- ✅ Model tüm 4 sınıfı tahmin edebiliyor
- ✅ Test accuracy > %70
- ✅ Confusion matrix dengeli
- ✅ Overfitting yok

---

## 🎯 Hızlı Komutlar

```bash
# WSL'de eğitimi başlat
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
chmod +x start_training_bone_wsl.sh
./start_training_bone_wsl.sh

# GPU kontrolü
nvidia-smi

# TensorFlow GPU test
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Conda environment aktive et
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
```

---

## 📝 Önemli Notlar

1. **Windows dosya yolu:** `/mnt/c/` üzerinden erişilir
2. **GPU:** Windows driver üzerinden otomatik paylaşılır
3. **Performance:** WSL'de Windows dosyalarına erişim yavaş olabilir
4. **LD_LIBRARY_PATH:** Conda CUDA library path'i gerekli

---

**WSL'de iyi eğitimler! 🚀**

