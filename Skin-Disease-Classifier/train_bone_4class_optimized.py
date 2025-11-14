#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bone Disease Detection - 4 CLASS OPTIMIZED TRAINING
Optimized for X-ray images with best practices
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2  # B3 yerine B2 (daha hızlı, benzer performans)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("\n" + "="*70)
print("BONE DISEASE DETECTION - 4 CLASS OPTIMIZED TRAINING")
print("Optimized for X-ray images")
print("="*70)

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"\n[GPU] {len(physical_devices)} GPU(s) available")
    for device in physical_devices:
        print(f"  - {device}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[GPU] Memory growth enabled")
        
        # Mixed precision training (optional - %50 hız artışı)
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        # print("[GPU] Mixed precision enabled")
    except:
        pass
else:
    print("\n[CPU] No GPU found, training on CPU")

# Hyperparameters - OPTIMIZED FOR BONE X-RAY
TRAIN_DIR = 'datasets/bone/Bone_4Class_Final/train'
VAL_DIR = 'datasets/bone/Bone_4Class_Final/val'
TEST_DIR = 'datasets/bone/Bone_4Class_Final/test'

# Image size - X-ray için daha büyük (detaylar için)
IMG_SIZE = (512, 512)  # 256 yerine 512 (detaylar için daha iyi)
BATCH_SIZE = 16  # 512x512 için batch size küçült (GPU memory için)
INITIAL_EPOCHS = 100
FINE_TUNE_EPOCHS = 50
LEARNING_RATE = 0.001  # 4 sınıf için biraz daha yüksek
FINE_TUNE_LR = 0.0001  # Fine-tune için
LABEL_SMOOTHING = 0.1

# 4 Classes
CLASS_NAMES = [
    'Normal',
    'Fracture',
    'Benign_Tumor',
    'Malignant_Tumor'
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Label Smoothing: {LABEL_SMOOTHING}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")

# Label smoothing loss
def categorical_crossentropy_smooth(smoothing=0.1):
    """
    Categorical crossentropy with label smoothing
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        num_classes = tf.cast(tf.shape(y_true)[1], tf.float32)
        y_true_smooth = y_true * (1.0 - smoothing) + smoothing / num_classes
        loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(loss)
    return loss

# X-RAY SPECIFIC DATA AUGMENTATION (anatomik açıdan uygun)
print("\n[DATA] Creating X-ray optimized data generators...")

# X-ray için özel augmentation:
# - Dikey/horizontal flip YOK (anatomik açıdan yanlış)
# - Rotation sınırlı (10-15 derece)
# - Brightness/contrast çok az
# - Shear ve zoom sınırlı
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # X-ray için sınırlı (anatomiyi bozmamak için)
    width_shift_range=0.1,  # Azaltıldı
    height_shift_range=0.1,  # Azaltıldı
    shear_range=0.05,  # Çok az
    zoom_range=0.1,  # Azaltıldı
    horizontal_flip=False,  # X-ray için YOK (anatomik açıdan yanlış)
    vertical_flip=False,  # X-ray için YOK (anatomik açıdan yanlış)
    fill_mode='constant',  # X-ray için 'constant' (siyah kenar)
    cval=0.0,  # Siyah doldurma
    brightness_range=[0.9, 1.1],  # Çok az brightness değişikliği
    channel_shift_range=5  # Azaltıldı
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True,
    seed=42,
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    color_mode='rgb'
)

print(f"\n[DATA] Generators created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# Calculate class weights
print("\n[DATA] Calculating class weights...")
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)

# Inverse frequency weighting
class_weights = total_samples / (NUM_CLASSES * class_counts)
class_weights_dict = dict(zip(range(NUM_CLASSES), class_weights))

print("\n[INFO] Class weights (inverse frequency):")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {class_weights_dict[i]:.3f} (samples: {class_counts[i]})")

# Build model - EfficientNetB2 (B3'ten daha hızlı, benzer performans)
print("\n[MODEL] Building EfficientNetB2 model...")

base_model = EfficientNetB2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# Optimized architecture
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile with label smoothing loss
smooth_loss = categorical_crossentropy_smooth(LABEL_SMOOTHING)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss=smooth_loss,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')  # 4 sınıf için top-2 yeterli
    ]
)

print("\n[MODEL] Model architecture:")
model.summary()
print(f"\n[MODEL] Total parameters: {model.count_params():,}")

# Callbacks
os.makedirs('models', exist_ok=True)

checkpoint_initial = ModelCheckpoint(
    'models/bone_4class_initial.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=50,  # Yüksek patience (model öğrenene kadar bekle)
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1,
    cooldown=5
)

# Cosine decay learning rate schedule (opsiyonel - daha iyi sonuçlar için)
def cosine_decay(epoch, total_epochs=INITIAL_EPOCHS):
    """Cosine decay learning rate"""
    import math
    lr = LEARNING_RATE
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    return lr * cosine_decay

# Cosine decay callback (ReduceLROnPlateau yerine kullanılabilir)
# cosine_lr = LearningRateScheduler(cosine_decay, verbose=1)

# Phase 1: Initial Training
print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (Base Model Frozen)")
print("Using: Label Smoothing + Class Weights + X-ray Augmentation")
print("="*70)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_initial, early_stopping, reduce_lr],
    verbose=1
)

# Load best model from phase 1
print("\n[INFO] Loading best model from Phase 1...")
model = keras.models.load_model(
    'models/bone_4class_initial.keras',
    custom_objects={'loss': smooth_loss}
)

# Check intermediate results
print("\n[EVAL] Phase 1 Results:")
loss, accuracy, top_2_acc = model.evaluate(test_generator, verbose=0)
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-2 Accuracy: {top_2_acc*100:.2f}%")

# Check if model is learning all classes
test_generator.reset()
y_pred_phase1 = model.predict(test_generator, verbose=0)
y_pred_classes_phase1 = np.argmax(y_pred_phase1, axis=1)
y_true = test_generator.classes

unique_preds = np.unique(y_pred_classes_phase1)
print(f"\n[INFO] Phase 1: Model predicts {len(unique_preds)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds}")
if len(unique_preds) == NUM_CLASSES:
    print("  EXCELLENT: Model is learning all classes!")
else:
    print(f"  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds)}")

# Phase 2: Fine-tuning
print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Unfreezing Top Layers)")
print("="*70)

base_model.trainable = True

# Unfreeze last 100 layers
freeze_until = len(base_model.layers) - 100
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

print(f"\n[MODEL] Unfrozen layers: {len([l for l in base_model.layers if l.trainable])}")
print(f"[MODEL] Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=smooth_loss,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
)

checkpoint_finetune = ModelCheckpoint(
    'models/bone_4class_finetuned.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping_finetune = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.001
)

# Fine-tune
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_finetune, early_stopping_finetune, reduce_lr],
    verbose=1
)

# Load best fine-tuned model
print("\n[INFO] Loading best fine-tuned model...")
model = keras.models.load_model(
    'models/bone_4class_finetuned.keras',
    custom_objects={'loss': smooth_loss}
)

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy, top_2_acc = model.evaluate(test_generator, verbose=1)

print(f"\n[RESULTS] Final Test Metrics:")
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-2 Accuracy: {top_2_acc*100:.2f}%")

# Per-class accuracy
print("\n[RESULTS] Per-class Accuracy:")
test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\n" + classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(y_true, y_pred_classes)
print("\n[RESULTS] Confusion Matrix:")
print(cm)

# Check prediction distribution
unique_preds_final = np.unique(y_pred_classes)
print(f"\n[INFO] Final: Model predicts {len(unique_preds_final)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds_final}")
if len(unique_preds_final) == NUM_CLASSES:
    print("  EXCELLENT: Model predicts all classes!")
else:
    print(f"  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds_final)}")

# Save final model
final_model_path = 'models/bone_disease_model_4class.keras'
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# Plot training history
print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_initial = range(1, len(history_initial.history['accuracy']) + 1)
epochs_finetune = range(len(history_initial.history['accuracy']) + 1, 
                        len(history_initial.history['accuracy']) + len(history_finetune.history['accuracy']) + 1)

# Accuracy
axes[0, 0].plot(epochs_initial, history_initial.history['accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[0, 0].plot(epochs_initial, history_initial.history['val_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[0, 0].plot(epochs_finetune, history_finetune.history['accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[0, 0].plot(epochs_finetune, history_finetune.history['val_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(epochs_initial, history_initial.history['loss'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[0, 1].plot(epochs_initial, history_initial.history['val_loss'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[0, 1].plot(epochs_finetune, history_finetune.history['loss'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[0, 1].plot(epochs_finetune, history_finetune.history['val_loss'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_title('Model Loss (Label Smoothing)', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-2 Accuracy
axes[1, 0].plot(epochs_initial, history_initial.history['top_2_accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_initial, history_initial.history['val_top_2_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_finetune, history_finetune.history['top_2_accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[1, 0].plot(epochs_finetune, history_finetune.history['val_top_2_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Top-2 Accuracy', fontsize=12)
axes[1, 0].set_title('Model Top-2 Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1],
            cbar_kws={'label': 'Count'})
axes[1, 1].set_xlabel('Predicted', fontsize=12)
axes[1, 1].set_ylabel('True', fontsize=12)
axes[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
plt.setp(axes[1, 1].get_yticklabels(), rotation=0)

plt.tight_layout()
plot_path = 'models/training_history_bone_4class.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"[SUCCESS] Training plots saved to: {plot_path}")

# Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n[FILES]")
print(f"  Final Model: {final_model_path}")
print(f"  Training Plot: {plot_path}")
print(f"\n[METRICS]")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-2 Accuracy: {top_2_acc*100:.2f}%")
print(f"  Total Classes: {NUM_CLASSES}")
print(f"  Total Parameters: {model.count_params():,}")
print(f"\n[CLASSES]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {i+1}. {class_name}")
print(f"\n[OPTIMIZATIONS]")
print(f"  ✓ X-ray specific augmentation (anatomik açıdan uygun)")
print(f"  ✓ Larger image size (512x512 for details)")
print(f"  ✓ EfficientNetB2 (B3'ten daha hızlı)")
print(f"  ✓ Label Smoothing Loss")
print(f"  ✓ Class Weights (inverse frequency)")
print(f"  ✓ High Early Stopping Patience (50 epochs)")
print(f"  ✓ Two-phase training (transfer + fine-tuning)")
if len(unique_preds_final) == NUM_CLASSES:
    print(f"\n  ✓ Model predicts all classes - Excellent!")
else:
    print(f"\n  ⚠ Model does not predict all classes")
print("="*70 + "\n")

