# 🚀 Google Colab Hızlı Başlangıç - Eye Disease Detection

## 📋 Ön Hazırlık (Lokal Bilgisayar)

### 1. ZIP Oluştur
```powershell
cd C:\Users\melih\dev\disease_detection\Skin-Disease-Classifier
Compress-Archive -Path datasets\Eye_Mendeley -DestinationPath Eye_Mendeley.zip -Force
```

**Çıktı:** `Eye_Mendeley.zip` (2-3 GB, 18,363 görüntü)

### 2. Google Drive'a Yükle (Manuel)
1. https://drive.google.com/ aç
2. **Yeni → Dosya yükle**
3. `Eye_Mendeley.zip` seç
4. Bekle (10-20 dakika)

---

## 🌐 Google Colab Kurulum

### ADIM 1: Yeni Notebook Oluştur
1. https://colab.research.google.com/ aç
2. **Dosya → Yeni not defteri**
3. **Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU (T4)** seç

### ADIM 2: Aşağıdaki Kodu Kopyala-Yapıştır

---

## 📝 COLAB KODU (Tüm hücreler)

### Hücre 1: GPU Kontrolü
```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
```

**Beklenen:** `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

### Hücre 2: Drive Mount & Dataset Extract
```python
from google.colab import drive
drive.mount('/content/drive')

# ZIP'i kopyala (Drive'dan Colab'a)
!cp /content/drive/MyDrive/Eye_Mendeley.zip /content/

# Çıkart
!unzip -q Eye_Mendeley.zip

# Kontrol
!echo "Dataset yapısı:"
!ls -lh Eye_Mendeley/
!echo ""
!echo "Train klasörü:"
!ls Eye_Mendeley/train/
```

---

### Hücre 3: Import & Config
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Config
TRAIN_DIR = '/content/Eye_Mendeley/train'
VAL_DIR = '/content/Eye_Mendeley/val'
TEST_DIR = '/content/Eye_Mendeley/test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 30
LEARNING_RATE = 0.0001
FINE_TUNE_LR = 0.00005

CLASS_NAMES = [
    'Diabetic_Retinopathy', 'Disc_Edema', 'Glaucoma',
    'Macular_Scar', 'Myopia', 'Normal',
    'Pterygium', 'Retinal_Detachment', 'Retinitis_Pigmentosa'
]
NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes: {NUM_CLASSES}")
print(f"Learning Rates: {LEARNING_RATE} / {FINE_TUNE_LR}")
```

---

### Hücre 4: Data Generators
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False
)

print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

# Class weights
class_counts = np.bincount(train_gen.classes)
total = len(train_gen.classes)
class_weights = total / (NUM_CLASSES * class_counts)
class_weights_dict = dict(zip(range(NUM_CLASSES), class_weights))

print("\nClass weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_weights_dict[i]:.2f} (n={class_counts[i]})")
```

---

### Hücre 5: Build Model
```python
base_model = EfficientNetB3(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print(f"Total params: {model.count_params():,}")
```

---

### Hücre 6: Phase 1 Training
```python
print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (Base Frozen)")
print("="*70)

checkpoint = ModelCheckpoint(
    'eye_initial.keras', monitor='val_accuracy',
    save_best_only=True, mode='max', verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss', patience=30,
    restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=5, min_lr=1e-7, verbose=1
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Load best
model = keras.models.load_model('eye_initial.keras')
loss1, acc1, top3_1 = model.evaluate(test_gen, verbose=0)
print(f"\nPhase 1: Acc={acc1*100:.2f}% | Top3={top3_1*100:.2f}%")
```

---

### Hücre 7: Phase 2 Fine-tuning
```python
print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Partial Unfreeze)")
print("="*70)

base_model.trainable = True
for layer in base_model.layers[:250]:
    layer.trainable = False

print(f"Unfrozen: {sum([l.trainable for l in base_model.layers])} layers")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

checkpoint2 = ModelCheckpoint(
    'eye_final.keras', monitor='val_accuracy',
    save_best_only=True, mode='max', verbose=1
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint2, early_stop, reduce_lr],
    verbose=1
)

# Load best
model = keras.models.load_model('eye_final.keras')
```

---

### Hücre 8: Final Evaluation
```python
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, acc, top3 = model.evaluate(test_gen, verbose=1)

print(f"\n🎯 RESULTS:")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  Top-3: {top3*100:.2f}%")

# Per-class
y_pred = model.predict(test_gen, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

print("\n" + classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Accuracy: {acc*100:.2f}%')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
```

---

### Hücre 9: Save & Download
```python
# Save to Drive
model.save('/content/drive/MyDrive/eye_disease_model.keras')
!cp confusion_matrix.png /content/drive/MyDrive/

print("✅ Model Google Drive'a kaydedildi!")
print("  Dosya: MyDrive/eye_disease_model.keras")

# Download direkt
from google.colab import files
files.download('eye_final.keras')
files.download('confusion_matrix.png')

print("\n✅ EĞİTİM TAMAMLANDI!")
print(f"  Final Accuracy: {acc*100:.2f}%")
```

---

## ⏱️ Beklenen Süre

| Adım | Süre |
|------|------|
| Dataset upload (Drive) | 10-20 dk (manuel) |
| Dataset extract (Colab) | 2-3 dk |
| Phase 1 (50 epochs) | ~60 dk |
| Phase 2 (30 epochs) | ~40 dk |
| **TOPLAM** | **~2 saat** 🚀 |

---

## 📊 Beklenen Sonuçlar

```
Test Accuracy:  60-70% (önceki: 28%)
Top-3 Accuracy: 85-90% (önceki: 37%)
```

---

## ✅ Checklist

**Hazırlık:**
- [ ] Eye_Mendeley.zip oluşturuldu
- [ ] Google Drive'a yüklendi
- [ ] Colab notebook oluşturuldu
- [ ] GPU (T4) seçildi

**Eğitim:**
- [ ] Drive mount edildi
- [ ] Dataset extract edildi
- [ ] Tüm hücreler çalıştırıldı
- [ ] Phase 1 tamamlandı
- [ ] Phase 2 tamamlandı
- [ ] Model indirildi

---

## 🎯 Sonraki Adımlar

Model eğitimi bittikten sonra:

```bash
# Lokal'e indir: eye_disease_model.keras
# Kopyala:
cp ~/Downloads/eye_disease_model.keras C:/Users/melih/dev/disease_detection/Skin-Disease-Classifier/models/

# API başlat:
cd C:\Users\melih\dev\disease_detection\Skin-Disease-Classifier
python eye_disease_api.py

# Test et:
python test_eye_api.py
```

---

**Başarılar! 🚀**

