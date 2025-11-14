# 5 Sınıflı Göz Hastalığı Eğitimi - Başlatma Adımları

## 🚀 WSL2 Ubuntu'da GPU ile Eğitim Başlatma

### Adım 1: WSL2 Ubuntu Terminal'ini Aç
- Windows Terminal veya WSL2 terminal aç
- `wsl` komutu ile Ubuntu'ya bağlan

### Adım 2: Conda Environment'ı Aktive Et
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu
```

### Adım 3: Proje Dizinine Git
```bash
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier
```

### Adım 4: GPU Kontrolü (Opsiyonel)
```bash
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]"
```

Çıktı: `GPU: 1` ve `NVIDIA GeForce RTX 5070` görünmeli.

### Adım 5: Eğitimi Başlat

**Seçenek 1: Script ile (Önerilen)**
```bash
bash start_training_5class_wsl.sh
```

**Seçenek 2: Manuel**
```bash
python3 train_mendeley_eye_5class.py 2>&1 | tee training_5class_live.log
```

---

## 📊 Eğitimi İzleme

### Terminal'de Canlı İzleme
```bash
tail -f training_5class_live.log
```

### Windows'tan İzleme
- PowerShell'de:
```powershell
Get-Content training_5class_live.log -Wait -Tail 50
```

---

## ⏱️ Beklenen Süre

- **Phase 1 (Initial Training)**: ~3-4 saat (60 epochs)
- **Phase 2 (Fine-tuning)**: ~2-3 saat (40 epochs)
- **Toplam**: ~5-7 saat (GPU hızına bağlı)

Her epoch yaklaşık: **2-4 dakika** (RTX 5070 ile)

---

## 📁 Çıktı Dosyaları

Eğitim tamamlandığında şu dosyalar oluşur:

1. `models/eye_disease_model_5class.keras` - Final model
2. `models/mendeley_eye_5class_initial.keras` - Phase 1 best model
3. `models/mendeley_eye_5class_finetuned.keras` - Phase 2 best model
4. `models/training_history_mendeley_eye_5class.png` - Training plots
5. `training_5class_live.log` - Training log

---

## ⚠️ Önemli Notlar

1. **Terminal'i KAPATMA**: Eğitim sürerken WSL2 terminal'ini kapatma
2. **Bilgisayarı Uyku Moduna Alma**: Eğitim kesilir
3. **GPU Belleği**: Eğitim sırasında GPU %80-100 kullanılır (normal)
4. **Durdurma**: `Ctrl+C` ile durdurabilirsin (best model kaydedilir)

---

## 🔧 Sorun Giderme

### GPU Bulunamıyorsa
```bash
# CUDA path kontrolü
echo $LD_LIBRARY_PATH

# CUDA kurulumunu kontrol et
nvcc --version
```

### Memory Hatası
- `BATCH_SIZE`'ı 32'den 16'ya düşür
- Script'te `BATCH_SIZE = 16` olarak değiştir

### Conda Environment Hatası
```bash
conda env list  # Tüm environment'ları listele
conda activate tf_gpu  # Tekrar aktif et
```

---

## ✅ Başarı Kontrolü

Eğitim başladığında göreceksin:

```
[GPU] 1 GPU(s) available
  - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

[DATA] Generators created:
  Training samples: 12754
  Validation samples: 722
  Test samples: 728

[MODEL] Building EfficientNetB3 model...
```

Bu mesajları görürsen **başarıyla başladı!** ✅


