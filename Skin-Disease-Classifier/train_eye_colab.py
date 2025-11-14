"""
GOOGLE COLAB İÇİN MENDELEY GÖZ HASTALIĞI EĞİTİMİ
Bu dosyayı Colab'a kopyala-yapıştır

ADIMLAR:
1. Google Colab aç: https://colab.research.google.com/
2. Runtime -> Change runtime type -> GPU (T4)
3. Bu dosyanın içeriğini yeni bir hücreye yapıştır
4. Çalıştır!

NOT: Dataset'i Google Drive'a yüklemen gerekecek veya direkt Colab'a upload edeceksin
"""

# ========== KURULUM ==========
print("📦 Paketler kuruluyor...")
!pip install -q tensorflow pillow matplotlib seaborn scikit-learn

# ========== DATASET UPLOAD ==========
print("\n📂 Dataset yükleniyor...")
print("Seçenek 1: Google Drive'dan")
from google.colab import drive
drive.mount('/content/drive')

# Dataset Google Drive'daysa:
# !cp /content/drive/MyDrive/Eye_Mendeley.zip /content/
# !unzip -q Eye_Mendeley.zip

# Seçenek 2: Direkt upload (YAVAŞ! - 2-3 GB)
# from google.colab import files
# uploaded = files.upload()  # Eye_Mendeley.zip seç
# !unzip -q Eye_Mendeley.zip

print("⚠️ MANUEL ADIM:")
print("1. Eye_Mendeley klasörünü Google Drive'a yükle")
print("2. Yukarıdaki yorum satırlarını kaldır")
print("3. Veya direkt upload et (yavaş)")

# ========== IMPORT ==========
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print(f"\n🎮 TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

# ========== CONFIG ==========
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

print(f"\n⚙️ Configuration:")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Epochs: {INITIAL_EPOCHS} + {FINE_TUNE_EPOCHS}")

# ========== DATA GENERATORS ==========
print("\n📊 Data generators oluşturuluyor...")

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

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False
)

print(f"  Train: {train_generator.samples}")
print(f"  Val: {val_generator.samples}")
print(f"  Test: {test_generator.samples}")

# ========== CLASS WEIGHTS ==========
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)
class_weights = total_samples / (NUM_CLASSES * class_counts)
class_weights_dict = dict(zip(range(NUM_CLASSES), class_weights))

print("\n⚖️ Class Weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_weights_dict[i]:.2f}")

# ========== MODEL ==========
print("\n🏗️ Model oluşturuluyor...")

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

print(f"  Total params: {model.count_params():,}")

# ========== CALLBACKS ==========
checkpoint_initial = ModelCheckpoint(
    'eye_initial.keras', monitor='val_accuracy',
    save_best_only=True, mode='max', verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=30,
    restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=5, min_lr=1e-7, verbose=1
)

# ========== PHASE 1: INITIAL TRAINING ==========
print("\n" + "="*70)
print("🚀 PHASE 1: INITIAL TRAINING (Base Frozen)")
print("="*70)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_initial, early_stopping, reduce_lr],
    verbose=1
)

model = keras.models.load_model('eye_initial.keras')

loss, acc, top3 = model.evaluate(test_generator, verbose=0)
print(f"\n📊 Phase 1 Results:")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  Top-3: {top3*100:.2f}%")

# ========== PHASE 2: FINE-TUNING ==========
print("\n" + "="*70)
print("🔥 PHASE 2: FINE-TUNING (Partial Unfreeze)")
print("="*70)

base_model.trainable = True
for layer in base_model.layers[:250]:
    layer.trainable = False

print(f"  Unfrozen: {len([l for l in base_model.layers if l.trainable])} layers")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

checkpoint_finetune = ModelCheckpoint(
    'eye_finetuned.keras', monitor='val_accuracy',
    save_best_only=True, mode='max', verbose=1
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_finetune, early_stopping, reduce_lr],
    verbose=1
)

model = keras.models.load_model('eye_finetuned.keras')

# ========== FINAL EVALUATION ==========
print("\n" + "="*70)
print("🎯 FINAL EVALUATION")
print("="*70)

loss, acc, top3 = model.evaluate(test_generator, verbose=1)

print(f"\n✅ FINAL RESULTS:")
print(f"  Test Accuracy: {acc*100:.2f}%")
print(f"  Top-3 Accuracy: {top3*100:.2f}%")

# Per-class
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Accuracy: {acc*100:.2f}%')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ========== SAVE & DOWNLOAD ==========
model.save('eye_disease_model.keras')
print("\n💾 Model kaydedildi: eye_disease_model.keras")

# Download
from google.colab import files
files.download('eye_disease_model.keras')
files.download('confusion_matrix.png')

print("\n✅ EĞİTİM TAMAMLANDI!")
print(f"  Final Accuracy: {acc*100:.2f}%")
print(f"  Model indirildi!")

