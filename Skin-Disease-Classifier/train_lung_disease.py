#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKCIGER HASTALIKLARI MODEL EGITIMI
===================================
Dataset: COVID-QU-Ex (Lung Segmentation Data)
Siniflar: COVID-19, Non-COVID (Pnomoni), Normal
Model: MobileNetV2 (Transfer Learning)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime

# UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print(" AKCIGER HASTALIKLARI SINIFLANDIRMA - MODEL EGITIMI")
print("=" * 70)

# ============================================================================
# 1. VERİ YOLLARI VE PARAMETRELERİ
# ============================================================================

BASE_DIR = "datasets/Lung Segmentation Data/Lung Segmentation Data"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR = os.path.join(BASE_DIR, "Val")
TEST_DIR = os.path.join(BASE_DIR, "Test")

# Model parametreleri
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

# Sınıf isimleri
CLASS_NAMES = ['COVID-19', 'Non-COVID', 'Normal']
NUM_CLASSES = len(CLASS_NAMES)

print("\n[VERI] Dizinler:")
print(f"  Train: {TRAIN_DIR}")
print(f"  Val:   {VAL_DIR}")
print(f"  Test:  {TEST_DIR}")

# ============================================================================
# 2. VERİ SETİNİ HAZIRLA
# ============================================================================

print("\n[VERI] Veri seti hazirlaniyor...")

# Sadece 'images' alt klasörlerini kullanacağız
def create_image_generators():
    """
    Her sınıfın 'images' klasörünü otomatik olarak bulup kullanır
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Özel fonksiyon: Her sınıf için 'images' klasörünü kullan
    def get_images_dir(base_dir):
        """Her sınıf klasörünün içindeki 'images' klasörünü kullan"""
        import shutil
        import tempfile
        
        # Geçici bir klasör oluştur
        temp_dir = tempfile.mkdtemp()
        
        for class_name in CLASS_NAMES:
            class_images_dir = os.path.join(base_dir, class_name, 'images')
            temp_class_dir = os.path.join(temp_dir, class_name)
            
            if os.path.exists(class_images_dir):
                # Sembolik link veya kopya oluştur
                if os.name != 'nt':  # Unix/Linux/Mac
                    os.symlink(class_images_dir, temp_class_dir)
                else:  # Windows
                    shutil.copytree(class_images_dir, temp_class_dir)
        
        return temp_dir
    
    # ImageDataGenerator için doğru yapıyı oluştur
    train_dir_processed = get_images_dir(TRAIN_DIR)
    val_dir_processed = get_images_dir(VAL_DIR)
    test_dir_processed = get_images_dir(TEST_DIR)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir_processed,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir_processed,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir_processed,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

try:
    train_gen, val_gen, test_gen = create_image_generators()
    
    print("\n[OK] Veri seti basariyla yuklendi!")
    print(f"\n[STATS] Veri Istatistikleri:")
    print(f"  Training samples:   {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples:       {test_gen.samples}")
    print(f"  Siniflar: {list(train_gen.class_indices.keys())}")
    
except Exception as e:
    print(f"\n[HATA] Veri yuklenirken sorun olustu: {e}")
    print("\n[INFO] Alternatif yontem deneniyor...")
    
    # Alternatif: Manuel veri yükleme
    def load_data_manually(base_dir):
        images = []
        labels = []
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_images_dir = os.path.join(base_dir, class_name, 'images')
            
            if not os.path.exists(class_images_dir):
                print(f"HATA - Klasör bulunamadı: {class_images_dir}")
                continue
            
            image_files = [f for f in os.listdir(class_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"  {class_name}: {len(image_files)} görüntü")
            
            for img_file in image_files[:100]:  # İlk 100 görüntü (test için)
                img_path = os.path.join(class_images_dir, img_file)
                try:
                    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
                    images.append(img_array)
                    labels.append(idx)
                except Exception as e:
                    continue
        
        return np.array(images), np.array(labels)
    
    print("\n📂 Manuel veri yükleme başladı...")
    X_train, y_train = load_data_manually(TRAIN_DIR)
    X_val, y_val = load_data_manually(VAL_DIR)
    X_test, y_test = load_data_manually(TEST_DIR)
    
    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print(f"\n✓ Manuel yükleme tamamlandı!")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

# ============================================================================
# 3. MODEL OLUŞTUR
# ============================================================================

print("\n Model oluşturuluyor (MobileNetV2 Transfer Learning)...")

# Base model (önceden eğitilmiş)
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# İlk katmanları dondur (transfer learning)
base_model.trainable = False

# Yeni model oluştur
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Model derleme
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model oluşturuldu!")
print("\n Model Özeti:")
model.summary()

# ============================================================================
# 4. CALLBACK'LER
# ============================================================================

# Model kaydetme yolu
MODEL_SAVE_PATH = "models/lung_disease_model.keras"
os.makedirs("models", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# 5. MODEL EĞİTİMİ
# ============================================================================

print("\n" + "=" * 70)
print(" EĞİTİM BAŞLIYOR")
print("=" * 70)

start_time = datetime.now()

try:
    # Generator ile eğitim
    if 'train_gen' in locals():
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Manuel veri ile eğitim
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print(" EĞİTİM TAMAMLANDI")
    print("=" * 70)
    print(f"⏱ Toplam süre: {training_time}")
    
except Exception as e:
    print(f"\n EĞİTİM HATASI: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 6. MODEL DEĞERLENDİRME
# ============================================================================

print("\n Model değerlendiriliyor...")

try:
    if 'test_gen' in locals():
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    else:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print("\n" + "=" * 70)
    print(" TEST SONUÇLARI")
    print("=" * 70)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 70)
    
except Exception as e:
    print(f" Değerlendirme hatası: {e}")

# ============================================================================
# 7. EĞİTİM GRAFİKLERİ
# ============================================================================

print("\n Eğitim grafikleri oluşturuluyor...")

try:
    plt.figure(figsize=(14, 5))
    
    # Accuracy grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history_lung.png', dpi=150)
    print("✓ Grafikler kaydedildi: models/training_history_lung.png")
    
except Exception as e:
    print(f" Grafik oluşturulamadı: {e}")

# ============================================================================
# 8. ÖZET RAPOR
# ============================================================================

print("\n" + "=" * 70)
print(" SON RAPOR")
print("=" * 70)
print(f"✓ Model kaydedildi: {MODEL_SAVE_PATH}")
print(f"✓ Sınıf sayısı: {NUM_CLASSES}")
print(f"✓ Sınıflar: {CLASS_NAMES}")
print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Eğitim süresi: {training_time}")
print("\n Sonraki adım: Flask API oluştur ve web arayüzünü güncelle")
print("=" * 70)

