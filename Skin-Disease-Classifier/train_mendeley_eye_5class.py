#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mendeley Eye Disease Detection - 5 CLASS TRAINING
Using EfficientNetB3 + Focal Loss + Better Data Augmentation
5 Classes: Macular_Scar, Myopia, Diabetic_Retinopathy, Glaucoma, Normal
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("\n" + "="*70)
print("MENDELEY EYE DISEASE DETECTION - 5 CLASS TRAINING")
print("Model: EfficientNetB3 + Focal Loss + Enhanced Augmentation")
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
    except:
        pass
else:
    print("\n[CPU] No GPU found, training on CPU")

# Hyperparameters
TRAIN_DIR = 'datasets/Eye_Mendeley/train'
VAL_DIR = 'datasets/Eye_Mendeley/val'
TEST_DIR = 'datasets/Eye_Mendeley/test'

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
INITIAL_EPOCHS = 60
FINE_TUNE_EPOCHS = 40
LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.0001

# 5 Selected Classes (Medium + High only)
CLASS_NAMES = [
    'Diabetic_Retinopathy',  # High
    'Glaucoma',              # High
    'Macular_Scar',          # Medium
    'Myopia',                # Medium
    'Normal'                 # High
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")
print(f"  Using: Focal Loss + Enhanced Augmentation")

# Focal Loss for imbalanced data
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss: Addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = alpha * tf.pow((1.0 - p_t), gamma)
        focal_loss_value = focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        return tf.reduce_mean(focal_loss_value)
    
    return focal_loss_fixed

# Enhanced data augmentation
print("\n[DATA] Creating enhanced data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    brightness_range=[0.7, 1.3],
    channel_shift_range=20
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
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\n[DATA] Generators created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# Calculate class weights (sqrt-adjusted for better balance)
print("\n[DATA] Calculating class weights...")
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)

# Use sqrt to make weights less extreme (better for balanced data)
class_weights = np.sqrt(total_samples / (NUM_CLASSES * class_counts))
class_weights_dict = dict(zip(range(NUM_CLASSES), class_weights))

print("\n[INFO] Class weights (sqrt-adjusted):")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {class_weights_dict[i]:.2f} (samples: {class_counts[i]})")

# Build model with EfficientNetB3
print("\n[MODEL] Building EfficientNetB3 model...")

base_model = EfficientNetB3(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# Enhanced model architecture
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.6),
    layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile with Focal Loss
focal = focal_loss(gamma=2.0, alpha=0.25)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

print("\n[MODEL] Model architecture:")
model.summary()

print(f"\n[MODEL] Total parameters: {model.count_params():,}")

# Callbacks
os.makedirs('models', exist_ok=True)

checkpoint_initial = ModelCheckpoint(
    'models/mendeley_eye_5class_initial.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=7,
    min_lr=1e-8,
    verbose=1
)

# Phase 1: Initial Training
print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (Base Model Frozen)")
print("Using: Focal Loss + Class Weights")
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
    'models/mendeley_eye_5class_initial.keras',
    custom_objects={'focal_loss_fixed': focal}
)

# Evaluate after Phase 1
print("\n[EVAL] Phase 1 Results:")
loss, accuracy, top_3_acc = model.evaluate(test_generator, verbose=0)
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-3 Accuracy: {top_3_acc*100:.2f}%")

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
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=focal,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

checkpoint_finetune = ModelCheckpoint(
    'models/mendeley_eye_5class_finetuned.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Fine-tune
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_finetune, early_stopping, reduce_lr],
    verbose=1
)

# Load best fine-tuned model
print("\n[INFO] Loading best fine-tuned model...")
model = keras.models.load_model(
    'models/mendeley_eye_5class_finetuned.keras',
    custom_objects={'focal_loss_fixed': focal}
)

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy, top_3_acc = model.evaluate(test_generator, verbose=1)

print(f"\n[RESULTS] Final Test Metrics:")
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-3 Accuracy: {top_3_acc*100:.2f}%")

# Per-class accuracy
print("\n[RESULTS] Per-class Accuracy:")
test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix

print("\n" + classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(y_true, y_pred_classes)
print("\n[RESULTS] Confusion Matrix:")
print(cm)

# Save final model
final_model_path = 'models/eye_disease_model_5class.keras'
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# Plot training history
print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_initial = range(1, len(history_initial.history['accuracy']) + 1)
epochs_finetune = range(len(history_initial.history['accuracy']) + 1, 
                        len(history_initial.history['accuracy']) + len(history_finetune.history['accuracy']) + 1)

# Accuracy
axes[0, 0].plot(epochs_initial, history_initial.history['accuracy'], 'b-', label='Train (Phase 1)')
axes[0, 0].plot(epochs_initial, history_initial.history['val_accuracy'], 'b--', label='Val (Phase 1)')
axes[0, 0].plot(epochs_finetune, history_finetune.history['accuracy'], 'r-', label='Train (Phase 2)')
axes[0, 0].plot(epochs_finetune, history_finetune.history['val_accuracy'], 'r--', label='Val (Phase 2)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(epochs_initial, history_initial.history['loss'], 'b-', label='Train (Phase 1)')
axes[0, 1].plot(epochs_initial, history_initial.history['val_loss'], 'b--', label='Val (Phase 1)')
axes[0, 1].plot(epochs_finetune, history_finetune.history['loss'], 'r-', label='Train (Phase 2)')
axes[0, 1].plot(epochs_finetune, history_finetune.history['val_loss'], 'r--', label='Val (Phase 2)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Model Loss (Focal Loss)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Top-3 Accuracy
axes[1, 0].plot(epochs_initial, history_initial.history['top_3_accuracy'], 'b-', label='Train (Phase 1)')
axes[1, 0].plot(epochs_initial, history_initial.history['val_top_3_accuracy'], 'b--', label='Val (Phase 1)')
axes[1, 0].plot(epochs_finetune, history_finetune.history['top_3_accuracy'], 'r-', label='Train (Phase 2)')
axes[1, 0].plot(epochs_finetune, history_finetune.history['val_top_3_accuracy'], 'r--', label='Val (Phase 2)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Top-3 Accuracy')
axes[1, 0].set_title('Model Top-3 Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Confusion Matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')
axes[1, 1].set_title('Confusion Matrix')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plot_path = 'models/training_history_mendeley_eye_5class.png'
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
print(f"  Top-3 Accuracy: {top_3_acc*100:.2f}%")
print(f"  Total Classes: {NUM_CLASSES}")
print(f"  Total Parameters: {model.count_params():,}")
print(f"\n[CLASSES]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {i+1}. {class_name}")
print(f"\n[IMPROVEMENTS]")
print(f"  - 5 balanced classes (removed 4 imbalanced classes)")
print(f"  - Focal Loss for better learning")
print(f"  - Enhanced data augmentation")
print(f"  - Larger image size (256x256)")
print(f"  - Better model architecture")
print(f"\n[NEXT STEP] Update Flask API to use 5-class model")
print("="*70 + "\n")


