#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mendeley Eye Disease Detection - IMPROVED Training Script
Using EfficientNetB3 + Focal Loss + Better Data Augmentation
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
print("MENDELEY EYE DISEASE DETECTION - IMPROVED TRAINING")
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

IMG_SIZE = (256, 256)  # Increased from 224 to 256 for better features
BATCH_SIZE = 32
INITIAL_EPOCHS = 60  # Increased
FINE_TUNE_EPOCHS = 40  # Increased
LEARNING_RATE = 0.001  # Increased back (was too low)
FINE_TUNE_LR = 0.0001  # Increased

# Remove Pterygium - too few samples (only 86 total)
CLASS_NAMES = [
    'Diabetic_Retinopathy',
    'Disc_Edema',
    'Glaucoma',
    'Macular_Scar',
    'Myopia',
    'Normal',
    'Retinal_Detachment',
    'Retinitis_Pigmentosa'
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Number of Classes: {NUM_CLASSES} (Pterygium removed)")
print(f"  Using: Focal Loss + Enhanced Augmentation")

# Focal Loss for imbalanced data
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss: Addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to avoid numerical issues
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate p_t (probability of true class)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Calculate focal weight
        focal_weight = alpha * tf.pow((1.0 - p_t), gamma)
        
        # Calculate focal loss
        focal_loss_value = focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        
        return tf.reduce_mean(focal_loss_value)
    
    return focal_loss_fixed

# Enhanced data augmentation
print("\n[DATA] Creating enhanced data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased from 20
    width_shift_range=0.25,  # Increased from 0.2
    height_shift_range=0.25,  # Increased from 0.2
    shear_range=0.2,  # Increased from 0.15
    zoom_range=0.3,  # Increased from 0.2
    horizontal_flip=True,
    vertical_flip=True,  # Changed from False
    fill_mode='reflect',  # Changed from 'nearest'
    brightness_range=[0.7, 1.3],  # Increased range
    channel_shift_range=20,  # New: color augmentation
    featurewise_center=False,
    featurewise_std_normalization=False
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

# Calculate class weights (still using for balance)
print("\n[DATA] Calculating class weights...")
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)

# Use sqrt to make weights less extreme
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
    layers.Dropout(0.6),  # Increased from 0.5
    layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  # Increased from 512
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Increased from 0.3
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  # Increased from 256
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # Increased from 0.2
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
    'models/mendeley_eye_initial_v2.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Changed from val_loss
    patience=20,  # Reduced from 30 (faster stopping if no improvement)
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,  # More aggressive reduction
    patience=7,  # Increased from 5
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
    'models/mendeley_eye_initial_v2.keras',
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

# Unfreeze last 100 layers (more aggressive fine-tuning)
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
    'models/mendeley_eye_finetuned_v2.keras',
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
    'models/mendeley_eye_finetuned_v2.keras',
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
final_model_path = 'models/eye_disease_model_v2.keras'
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
plot_path = 'models/training_history_mendeley_eye_v2.png'
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
print(f"  Total Classes: {NUM_CLASSES} (Pterygium removed)")
print(f"  Total Parameters: {model.count_params():,}")
print(f"\n[IMPROVEMENTS]")
print(f"  - Focal Loss for imbalanced data")
print(f"  - Enhanced data augmentation")
print(f"  - Larger image size (256x256)")
print(f"  - Removed Pterygium (too few samples)")
print(f"  - Better class weights (sqrt-adjusted)")
print(f"\n[NEXT STEP] Run Flask API:")
print(f"  python eye_disease_api.py")
print("="*70 + "\n")


