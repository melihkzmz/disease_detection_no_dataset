# WSL2'de GPU ile Eğitim Başlatma Rehberi

## 🚀 Hızlı Başlangıç (Önerilen)

### Yöntem 1: Hazır Script ile (Kolay)

WSL2 Ubuntu terminalinde şu komutu çalıştır:

```bash
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
bash start_training_improved_wsl.sh
```

Bu script otomatik olarak:
- ✅ GPU library path'ini ayarlar
- ✅ Conda environment'ı aktive eder
- ✅ GPU kontrolü yapar
- ✅ Eğitimi başlatır
- ✅ Log dosyası oluşturur

---

## 🛠️ Yöntem 2: Manuel Komutlar

WSL2 Ubuntu terminalinde adım adım:

### Adım 1: GPU Library Path Ayarla
```bash
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Adım 2: Conda Environment Aktive Et
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
```

### Adım 3: GPU Kontrolü
```bash
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU: {len(gpus)}')"
```

**Beklenen çıktı:** `GPU: 1`

### Adım 4: Proje Dizinine Git
```bash
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
```

### Adım 5: Eğitimi Başlat
```bash
python3 train_mendeley_eye_5class_improved.py 2>&1 | tee training_improved.log
```

---

## 📋 Tek Satır Komut (Tüm Adımlar)

```bash
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && source ~/miniconda3/etc/profile.d/conda.sh && conda activate tf_gpu && cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier && python3 train_mendeley_eye_5class_improved.py 2>&1 | tee training_improved_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔍 Eğitim İlerlemesini İzleme

### Gerçek Zamanlı Log İzleme
Başka bir terminal aç ve:
```bash
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
tail -f training_improved_*.log
```

### GPU Kullanımını İzleme
```bash
watch -n 1 nvidia-smi
```

---

## ⚠️ Önemli Notlar

1. **LD_LIBRARY_PATH**: Her yeni terminalde ayarlanması gerekir (script otomatik yapar)

2. **Terminali Kapatma**: Eğitim sırasında WSL2 terminalini kapatmayın!

3. **Eğitim Süresi**: 
   - Phase 1: ~5-6 saat (100 epochs)
   - Phase 2: ~2-3 saat (50 epochs)
   - **Toplam: ~8-10 saat**

4. **Log Dosyası**: `training_improved_YYYYMMDD_HHMMSS.log` dosyasında tüm çıktı kaydedilir

5. **Durdurma**: Ctrl+C ile durdurabilirsiniz (ama önerilmez - model kaybolabilir)

---

## 🐛 Sorun Giderme

### GPU: 0 Görünüyorsa
```bash
# LD_LIBRARY_PATH kontrol et
echo $LD_LIBRARY_PATH

# Tekrar ayarla
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Conda Command Not Found
```bash
# Conda'yı initialize et
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### ModuleNotFoundError
```bash
# Conda environment içinde olduğundan emin ol
conda activate tf_gpu
pip list | grep tensorflow
```

---

## 📊 Eğitim Sonrası

Eğitim bittiğinde:
- ✅ Model: `models/eye_disease_model_5class_improved.keras`
- ✅ Training plot: `models/training_history_mendeley_eye_5class_improved.png`
- ✅ Log: `training_improved_*.log`

**Başarı kontrolü:**
- Accuracy > %50 olmalı
- Tüm 5 sınıf confusion matrix'te görünmeli
- Top-3 accuracy > %80 olmalı

---

## 🎯 Başlatma Komutu (Özet)

**En kolay yol:**
```bash
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
bash start_training_improved_wsl.sh
```

**Hazır!** 🚀


