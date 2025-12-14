#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skin Disease Detection - 5 CLASS MACRO F1 TRAINING (df and vasc excluded)
Model: EfficientNetB3 (Medical Imaging Optimized)
Optimized for: Class Imbalance + Medical Safety + Professional Standards
Uses: Class-Balanced Focal Loss + Macro F1 Metric
Color Mode: RGB (Dermatoscopic images optimized)
Classes: akiec, bcc, bkl, mel, nv (df and vasc excluded - insufficient data)
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("\n" + "="*70)
print("🧬 SKIN DISEASE DETECTION - 5 CLASS MACRO F1 TRAINING")
print("="*70)
print("✅ Model: EfficientNetB3 (Medical Imaging Optimized)")
print("✅ Optimized for: Class Imbalance + Medical Safety")
print("✅ Uses: Class-Balanced Focal Loss + Macro F1 Metric")
print("✅ Color Mode: RGB (Dermatoscopic images)")
print("✅ Classes: akiec, bcc, bkl, mel, nv (df and vasc excluded)")
print("✅ Professional Standard for Medical AI")
print("="*70)

# Mixed precision disabled for Windows compatibility
print("[MEMORY] Mixed Precision Training DISABLED for Windows compatibility")

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
USE_GPU = False

if len(physical_devices) > 0:
    print(f"\n[GPU] {len(physical_devices)} GPU(s) available")
    for device in physical_devices:
        print(f"  - {device}")
    
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[GPU] Memory growth enabled")
        
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0])
            _ = tf.reduce_sum(test_tensor)
        USE_GPU = True
        print("[GPU] ✅ GPU test passed, using GPU for training")
    except Exception as e:
        print(f"[GPU] ⚠️  GPU configuration failed: {e}")
        print("[GPU] 🔄 Falling back to CPU mode")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        USE_GPU = False
        tf.config.set_visible_devices([], 'GPU')
else:
    print("\n[CPU] No GPU found, training on CPU")
    USE_GPU = False

# Hyperparameters
TRAIN_DIR = 'datasets/HAM10000/base_dir/train_dir'
VAL_DIR = 'datasets/HAM10000/base_dir/val_dir'
TEST_DIR = 'datasets/HAM10000/base_dir/test_dir'

IMG_SIZE = (300, 300)  # EfficientNetB3 input size (deri görüntüleri için optimize)
BATCH_SIZE = 16  # EfficientNetB3 için uygun batch size
INITIAL_EPOCHS = 100
FINE_TUNE_EPOCHS = 50
LEARNING_RATE = 0.0001  # Macro F1 loss için optimize edilmiş
FINE_TUNE_LR = 0.00001
COLOR_MODE = 'rgb'  # RGB dermatoscopic images

# 5 Classes (df and vasc excluded - insufficient data)
CLASS_NAMES = [
    'akiec',   # Actinic Keratoses
    'bcc',     # Basal Cell Carcinoma
    'bkl',     # Benign Keratosis
    'mel',     # Melanoma
    'nv'       # Melanocytic Nevi
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Color Mode: {COLOR_MODE}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")
print(f"  Excluded: df (Dermatofibroma), vasc (Vascular Lesions - insufficient data)")

# Create models directory
os.makedirs('models', exist_ok=True)

# ============================================================================
# CLASS-BALANCED FOCAL LOSS FUNCTION
# ============================================================================

def class_balanced_focal_loss(y_true, y_pred, alpha=None, gamma=2.0):
    """
    Class-Balanced Focal Loss for imbalanced datasets.
    
    Combines Focal Loss (focuses on hard examples) with class balancing.
    
    Formula:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t: probability of true class
    - α_t: class-specific weighting factor (balanced)
    - γ: focusing parameter (default 2.0)
    
    Args:
        y_true: One-hot encoded true labels
        y_pred: Predicted probabilities
        alpha: Class-specific weights (list or None for auto-balancing)
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Class-balanced focal loss
    """
    # Clip predictions to prevent numerical issues
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # If alpha not provided, use balanced weights based on class frequency
    if alpha is None:
        # Auto-balance: give more weight to minority classes
        # This will be set dynamically based on class distribution
        alpha = [1.0] * NUM_CLASSES
    
    # Calculate focal loss for each sample
    # Get true class probabilities
    p_t = tf.reduce_sum(y_true * y_pred, axis=1, keepdims=True)  # (batch_size, 1)
    
    # Calculate (1 - p_t)^γ
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    
    # Calculate cross-entropy: -log(p_t)
    ce = -tf.math.log(p_t + 1e-7)
    
    # Apply class-specific alpha weights
    # For each sample, get the alpha corresponding to its true class
    alpha_t = tf.reduce_sum(y_true * tf.constant(alpha, dtype=tf.float32), axis=1, keepdims=True)
    
    # Calculate focal loss: -α_t * (1 - p_t)^γ * log(p_t)
    focal_loss = alpha_t * modulating_factor * ce
    
    # Return mean loss over batch
    return tf.reduce_mean(focal_loss)


def get_class_balanced_focal_loss_fn(class_weights_dict, gamma=2.0):
    """
    Create a class-balanced focal loss function with specific class weights.
    
    Args:
        class_weights_dict: Dictionary mapping class index to weight
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Loss function ready to use in model.compile()
    """
    # Convert dict to list (assuming class indices are 0 to NUM_CLASSES-1)
    alpha_list = [class_weights_dict.get(i, 1.0) for i in range(NUM_CLASSES)]
    
    def loss_fn(y_true, y_pred):
        return class_balanced_focal_loss(y_true, y_pred, alpha=alpha_list, gamma=gamma)
    
    return loss_fn

# ============================================================================
# STREAMING MACRO F1 METRIC (Global TP/FP/FN Accumulation)
# ============================================================================

class StreamingMacroF1(keras.metrics.Metric):
    """
    Streaming Macro F1 Metric that accumulates TP/FP/FN across all batches.
    This matches sklearn's macro F1 calculation on the full dataset.
    """
    def __init__(self, num_classes=5, name='macro_f1_metric', **kwargs):
        super(StreamingMacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Initialize state variables for each class: TP, FP, FN
        self.true_positives = self.add_weight(
            name='tp',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update TP, FP, FN counts for current batch."""
        # Convert to class indices
        y_true_classes = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
        y_pred_classes = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        
        # Vectorized computation
        y_true_one_hot = tf.one_hot(y_true_classes, depth=self.num_classes, dtype=tf.float32)
        y_pred_one_hot = tf.one_hot(y_pred_classes, depth=self.num_classes, dtype=tf.float32)
        
        # TP, FP, FN
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)
        
        # Update state
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        """Calculate macro F1 from accumulated TP/FP/FN."""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_scores = 2.0 * precision * recall / (precision + recall + 1e-8)
        
        macro_f1 = tf.reduce_mean(f1_scores)
        return macro_f1
    
    def reset_state(self):
        """Reset all state variables at the start of each epoch."""
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

# ============================================================================
# SKLEARN MACRO F1 CALLBACK
# ============================================================================

class SklearnMacroF1Callback(Callback):
    """
    Callback to compute sklearn's macro F1 at the end of each epoch on validation set.
    """
    def __init__(self, val_generator, num_classes=5, verbose=1):
        super(SklearnMacroF1Callback, self).__init__()
        self.val_generator = val_generator
        self.num_classes = num_classes
        self.verbose = verbose
        self.sklearn_f1_scores = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute sklearn macro F1 on full validation set."""
        self.val_generator.reset()
        
        y_true_all = []
        y_pred_all = []
        
        for batch_idx in range(len(self.val_generator)):
            batch_x, batch_y = self.val_generator[batch_idx]
            y_pred_batch = self.model.predict(batch_x, verbose=0)
            
            y_true_all.append(np.argmax(batch_y, axis=1))
            y_pred_all.append(np.argmax(y_pred_batch, axis=1))
        
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.sklearn_f1_scores.append(macro_f1)
        
        if self.verbose > 0:
            print(f"\n[SKLEARN] Validation Macro F1: {macro_f1*100:.2f}%")

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_image_efficientnet(img):
    """
    Preprocess image for EfficientNetB3 input with dermoscopy-safe contrast adjustment.
    
    Process:
    1. ImageDataGenerator automatically resizes to (300, 300)
    2. Optional contrast adjustment (0.95-1.05 range)
    3. EfficientNet preprocessing (ImageNet normalization)
    
    EfficientNet preprocessing:
    - Scales pixels to [0, 1] range (divides by 255)
    - Applies ImageNet mean/std normalization
    - This matches exactly what pretrained EfficientNetB3 expects
    
    Args:
        img: numpy array (H, W, 3) in [0, 255] (uint8 or float)
    
    Returns:
        Preprocessed image array (normalized to ImageNet statistics)
    """
    # Ensure input is float32 in [0, 255] range
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Contrast adjustment (dermoscopy-safe: 0.95-1.05 range)
    # This is applied before normalization to preserve clinical features
    # Note: Contrast adjustment is handled by ImageDataGenerator's brightness_range
    # for simplicity, we skip explicit contrast here as it's less critical
    
    # EfficientNet preprocessing (handles ImageNet normalization internally)
    # This applies: scale to [0,1] then ImageNet mean/std normalization
    img_preprocessed = efficientnet_preprocess(img)
    return img_preprocessed

# ============================================================================
# DATA GENERATORS
# ============================================================================

print("\n[1/5] Data generators oluşturuluyor...")

# Training: with dermoscopy-safe augmentation
# Dermatoscopic images require careful augmentation to preserve clinical features
# Note: Keras ImageDataGenerator doesn't have contrast_range parameter
# Brightness adjustment helps with contrast variation to some extent
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image_efficientnet,
    rotation_range=15,         # Limited rotation (dermoscopy-safe: ±15°)
    width_shift_range=0.1,     # 10% horizontal shift
    height_shift_range=0.1,    # 10% vertical shift
    zoom_range=0.1,            # 10% zoom (±10%)
    horizontal_flip=True,      # Horizontal flip (dermoscopy-safe)
    vertical_flip=False,       # NO vertical flip (preserves clinical orientation)
    brightness_range=[0.95, 1.05],  # 5% brightness variation (minimal, dermoscopy-safe)
    # Note: contrast_range not available in Keras ImageDataGenerator
    # Brightness adjustment provides some contrast-like variation
    fill_mode='nearest'        # Fill mode for augmentation
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,  # Only 5 classes (df and vasc excluded)
    shuffle=True
)

# Validation and test: no augmentation, only preprocessing
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image_efficientnet
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image_efficientnet
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\n[INFO] Dataset bilgileri:")
print(f"  Train samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")
print(f"  Number of classes: {train_generator.num_classes}")
print(f"  Class indices: {train_generator.class_indices}")

# Calculate steps per epoch
train_steps = len(train_generator)
val_steps = len(val_generator)
test_steps = len(test_generator)

print(f"\n[INFO] Steps per epoch:")
print(f"  Train steps: {train_steps}")
print(f"  Validation steps: {val_steps}")
print(f"  Test steps: {test_steps}")

# ============================================================================
# CLASS WEIGHTS (for imbalanced dataset)
# ============================================================================

print("\n[2/5] Class weights hesaplanıyor...")

# Get class weights based on training set distribution
class_indices = train_generator.class_indices

# Get actual training labels from generator (more accurate than folder counts)
# This handles cases where files are skipped, extensions differ, or augmentation filtering occurs
all_train_labels = train_generator.classes

# Count samples per class using actual training labels (not folder counts)
# Use np.bincount to get actual distribution from generator
actual_class_counts = np.bincount(all_train_labels, minlength=len(CLASS_NAMES))
class_counts = {}
for class_name in CLASS_NAMES:
    class_idx = class_indices[class_name]
    class_counts[class_name] = int(actual_class_counts[class_idx])

total_samples = len(all_train_labels)
print(f"  Class distribution in training set (from actual generator labels):")
for class_name in CLASS_NAMES:
    count = class_counts.get(class_name, 0)
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    print(f"    {class_name}: {count} ({percentage:.2f}%)")

# Calculate balanced class weights using sklearn

# Compute balanced class weights using sklearn
balanced_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_train_labels),
    y=all_train_labels
)

# Create weight dictionary using sklearn's balanced class weights (data-driven, no arbitrary caps)
# sklearn's compute_class_weight('balanced') automatically calculates optimal weights
# based on class frequency: n_samples / (n_classes * np.bincount(y))
# This is data-driven, clinically justified, and reproducible - no hand-tuned caps needed
class_weights = {}
for idx, class_name in enumerate(CLASS_NAMES):
    class_idx = class_indices[class_name]
    # Find the index in unique classes array
    unique_classes = np.unique(all_train_labels)
    balanced_idx = np.where(unique_classes == class_idx)[0]
    if len(balanced_idx) > 0:
        balanced_weight = float(balanced_weights[balanced_idx[0]])
    else:
        balanced_weight = 1.0
    
    # Use sklearn's balanced weight directly (data-driven, no arbitrary adjustments)
    count = class_counts.get(class_name, 1)
    class_ratio = count / total_samples if total_samples > 0 else 1.0
    
    # Direct use of sklearn's balanced weight (optimal for class imbalance)
    class_weights[class_idx] = balanced_weight
    print(f"    {class_name}: weight = {balanced_weight:.3f} (ratio: {class_ratio*100:.1f}%, count: {count})")

class_weight_dict = class_weights

# Create Class-Balanced Focal Loss function with class weights
# Gamma parameter controls focusing (2.0 = standard for medical imaging)
# Alpha weights: sklearn's balanced weights (data-driven, no arbitrary caps)
# Medical imaging best practice: gamma=1.5-2.0 with sklearn balanced alpha
# IMPORTANT: Class weights are included in the loss function (alpha parameter)
# Therefore, DO NOT use class_weight= in model.fit() - that would cause double weighting
FOCAL_LOSS_GAMMA = 2.0  # Medical imaging standard (1.5-2.0 range)
focal_loss_fn = get_class_balanced_focal_loss_fn(class_weight_dict, gamma=FOCAL_LOSS_GAMMA)
print(f"\n[LOSS] Class-Balanced Focal Loss configured:")
print(f"  Gamma (focusing parameter): {FOCAL_LOSS_GAMMA} (medical imaging standard: 1.5-2.0)")
print(f"  Alpha weights: sklearn balanced weights (data-driven: n_samples / (n_classes * bincount))")
print(f"  Strategy: No arbitrary caps or hand-tuned adjustments - fully data-driven")
print(f"  Note: Focal loss already focuses learning - minimal dropout to avoid gradient vanishing")
print(f"  Note: class_weight= NOT used in fit() to avoid double weighting")

# ============================================================================
# MODEL ARCHITECTURE - PHASE 1: Feature Extraction
# ============================================================================

print("\n[3/5] Model oluşturuluyor (EfficientNetB3)...")

# Base model: EfficientNetB3 (pre-trained on ImageNet)
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model layers (Phase 1)
base_model.trainable = False

# Build model (Phase 1: Minimal regularization - focal loss already focuses learning)
# Phase 1 strategy: Focal loss downweights easy samples and focuses learning
# Adding heavy dropout can cause gradient vanishing for minority classes
# Frozen backbone + BatchNorm provide sufficient regularization
# Small head prevents overfitting without needing additional dropout
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)  # BatchNorm for stability (sufficient regularization)
x = layers.Dense(128, activation='relu')(x)  # Smaller head (reduced from 256→128→5 to 128→5)
x = layers.Dropout(0.2)(x)  # Minimal dropout (only before output, avoid gradient vanishing)
predictions = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions', dtype='float32')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

print(f"[MODEL] Trainable layers (Phase 1): {len([l for l in model.layers if l.trainable])}")
print(f"[MODEL] Frozen layers: {len([l for l in model.layers if not l.trainable])}")

# Compile model
streaming_macro_f1 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list = [
    'accuracy',
    streaming_macro_f1,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

# Phase 1: Use Class-Balanced Focal Loss
# Focal loss focuses on hard examples and handles class imbalance well
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss=focal_loss_fn,  # Class-Balanced Focal Loss
    metrics=metrics_list  # Macro F1 monitored as metric
)

print("[MODEL] Model compiled with Class-Balanced Focal Loss (Phase 1)")
print("[MODEL] Focal Loss: Focuses on hard examples, handles class imbalance")
print("[MODEL] Macro F1 monitored as metric")

# ============================================================================
# CALLBACKS
# ============================================================================

print("\n[4/5] Callbacks oluşturuluyor...")

checkpoint = ModelCheckpoint(
    'models/skin_5class_efficientnetb3_macro_f1_phase1.keras',
    monitor='val_macro_f1_metric',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_macro_f1_metric',
    patience=15,
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.001
)

# ReduceLROnPlateau for Phase 1 (feature extraction with frozen backbone)
# Phase 1 uses higher LR (LEARNING_RATE = 0.0001)
reduce_lr = ReduceLROnPlateau(
    monitor='val_macro_f1_metric',
    factor=0.5,  # Reduce LR by half when plateau is reached
    patience=5,  # Wait 5 epochs before reducing
    min_lr=1e-7,
    verbose=1,
    mode='max'
)

sklearn_macro_f1_callback = SklearnMacroF1Callback(
    val_generator=val_generator,
    num_classes=NUM_CLASSES,
    verbose=1
)

# ============================================================================
# PHASE 1: FEATURE EXTRACTION (Frozen Base Model)
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: FEATURE EXTRACTION (Base Model Frozen)")
print("Using: CLASS-BALANCED FOCAL LOSS + MACRO F1 METRIC")
print("Strategy: Focal loss for hard examples & class imbalance, Macro F1 as metric")
print("="*70)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    # Note: class_weight NOT used here - Class-Balanced Focal Loss already includes class weights in alpha
    callbacks=[checkpoint, early_stopping, reduce_lr, sklearn_macro_f1_callback],
    verbose=1
)

# Load best model from Phase 1
print("\n[INFO] Loading best model from Phase 1...")
model = keras.models.load_model(
    'models/skin_5class_efficientnetb3_macro_f1_phase1.keras',
    custom_objects={
        'StreamingMacroF1': StreamingMacroF1,
        'class_balanced_focal_loss': class_balanced_focal_loss,
        'get_class_balanced_focal_loss_fn': get_class_balanced_focal_loss_fn,
        'focal_loss_fn': focal_loss_fn,
    },
    compile=False
)

# Recompile for Phase 2
streaming_macro_f1_phase1 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_phase1 = [
    'accuracy',
    streaming_macro_f1_phase1,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

# Phase 2: Continue using Class-Balanced Focal Loss
# Focal loss works well for fine-tuning with class imbalance
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=focal_loss_fn,  # Class-Balanced Focal Loss
    metrics=metrics_list_phase1  # Macro F1 monitored as metric
)

# Evaluate Phase 1
print("\n[EVAL] Phase 1 Results:")
test_generator.reset()
results_phase1 = model.evaluate(test_generator, verbose=0)

metric_names = model.metrics_names
loss_idx = metric_names.index('loss')
accuracy_idx = metric_names.index('accuracy')
macro_f1_idx = metric_names.index('macro_f1_metric') if 'macro_f1_metric' in metric_names else -1

print(f"  Test Loss (Categorical Crossentropy): {results_phase1[loss_idx]:.4f}")
print(f"  Test Accuracy: {results_phase1[accuracy_idx]*100:.2f}%")
if macro_f1_idx >= 0:
    print(f"  Test Macro F1: {results_phase1[macro_f1_idx]*100:.2f}%")

# Check if model is learning all classes
test_generator.reset()
y_pred_phase1 = model.predict(test_generator, verbose=0)
y_pred_classes_phase1 = np.argmax(y_pred_phase1, axis=1)
y_true = test_generator.classes

unique_preds = np.unique(y_pred_classes_phase1)
print(f"\n[INFO] Phase 1: Model predicts {len(unique_preds)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds}")
if len(unique_preds) == NUM_CLASSES:
    print("  ✅ EXCELLENT: Model is learning all classes!")
else:
    print(f"  ⚠️  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds)}")

# Calculate Macro F1 using sklearn
macro_f1_sklearn = f1_score(y_true, y_pred_classes_phase1, average='macro', zero_division=0)
print(f"  Macro F1 (sklearn): {macro_f1_sklearn*100:.2f}%")

# ============================================================================
# PHASE 2: FINE-TUNING (Top Layers Only - Safer Approach)
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Top Layers Only - Safer for Medical Imaging)")
print("Using: CLASS-BALANCED FOCAL LOSS + MACRO F1 METRIC")
print("Strategy: Unfreeze only top layers (last 30 layers) to avoid destroying learned representations")
print("Note: Unfreezing all layers can cause instability with small medical datasets")
print("="*70)

# Freeze all layers first
base_model.trainable = False

# Unfreeze only top layers (last 30 layers) - safer approach for medical imaging
# This avoids destroying learned representations in early feature extractors
# and prevents training instability, especially with small medical datasets
NUM_LAYERS_TO_UNFREEZE = 30
total_layers = len(base_model.layers)
layers_to_unfreeze = min(NUM_LAYERS_TO_UNFREEZE, total_layers)

# Unfreeze top layers
for layer in base_model.layers[-layers_to_unfreeze:]:
    layer.trainable = True

# Keep ALL BatchNorm layers in inference mode during fine-tuning for stability
# This prevents BatchNorm statistics from changing dramatically during fine-tuning
# Especially important with small batches (BATCH_SIZE=16) in medical imaging
# BatchNorm layer statistics can be unstable with small batches, freezing prevents this
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

unfrozen_count = len([l for l in base_model.layers if l.trainable])
frozen_count = len([l for l in base_model.layers if not l.trainable])
total_batchnorm_count = len([l for l in base_model.layers if isinstance(l, layers.BatchNormalization)])

print(f"\n[MODEL] Total base model layers: {total_layers}")
print(f"[MODEL] Unfrozen layers: {unfrozen_count} (last {layers_to_unfreeze} layers, excluding all BatchNorm)")
print(f"[MODEL] Frozen layers: {frozen_count} (early feature extractors + all {total_batchnorm_count} BatchNorm layers)")
print(f"[MODEL] BatchNorm layers: ALL {total_batchnorm_count} BatchNorm layers kept in inference mode (trainable=False)")
print(f"[MODEL] Strategy: Gradual fine-tuning of top layers only, BatchNorm frozen (safer for small batches in medical imaging)")

# Recompile with lower learning rate
streaming_macro_f1_phase2 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_phase2 = [
    'accuracy',
    streaming_macro_f1_phase2,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=focal_loss_fn,  # Class-Balanced Focal Loss
    metrics=metrics_list_phase2
)

checkpoint_finetune = ModelCheckpoint(
    'models/skin_5class_efficientnetb3_macro_f1_finetuned.keras',
    monitor='val_macro_f1_metric',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping_finetune = EarlyStopping(
    monitor='val_macro_f1_metric',
    patience=20,
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.002
)

# Separate ReduceLROnPlateau for Phase 2 (fine-tuning)
# Phase 1 and Phase 2 have different LR scales, so separate callback is needed
# Phase 2 uses lower LR (FINE_TUNE_LR), so reduction strategy should be different
reduce_lr_phase2 = ReduceLROnPlateau(
    monitor='val_macro_f1_metric',
    factor=0.3,  # More conservative than Phase 1 (0.5) for fine-tuning
    patience=6,  # Slightly longer patience than Phase 1 (5) for fine-tuning
    min_lr=1e-7,
    verbose=1,
    mode='max'
)

sklearn_macro_f1_callback_phase2 = SklearnMacroF1Callback(
    val_generator=val_generator,
    num_classes=NUM_CLASSES,
    verbose=1
)

# Fine-tune
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    # Note: class_weight NOT used here - Class-Balanced Focal Loss already includes class weights in alpha
    callbacks=[checkpoint_finetune, early_stopping_finetune, reduce_lr_phase2, sklearn_macro_f1_callback_phase2],
    verbose=1
)

# Load best fine-tuned model
print("\n[INFO] Loading best fine-tuned model...")
model = keras.models.load_model(
    'models/skin_5class_efficientnetb3_macro_f1_finetuned.keras',
    custom_objects={
        'StreamingMacroF1': StreamingMacroF1,
        'class_balanced_focal_loss': class_balanced_focal_loss,
        'get_class_balanced_focal_loss_fn': get_class_balanced_focal_loss_fn,
        'focal_loss_fn': focal_loss_fn,
    }
)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

test_generator.reset()
results = model.evaluate(test_generator, verbose=1)

# Extract results
loss = results[loss_idx]
accuracy = results[accuracy_idx]
macro_f1 = results[macro_f1_idx] if macro_f1_idx >= 0 else 0

print(f"\n[RESULTS] Final Test Metrics:")
print(f"  Test Loss (Class-Balanced Focal Loss): {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Test Macro F1: {macro_f1*100:.2f}%")

# Per-class metrics
test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification Report
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE (Classification Report)")
print("="*70)
print(classification_report(
    y_true, 
    y_pred_classes, 
    target_names=CLASS_NAMES, 
    zero_division=0,
    digits=4
))

# Calculate Macro F1 using sklearn
macro_f1_sklearn = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
weighted_f1_sklearn = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)

print(f"\n[RESULTS] Sklearn F1 Scores:")
print(f"  Macro F1: {macro_f1_sklearn*100:.2f}%")
print(f"  Weighted F1: {weighted_f1_sklearn*100:.2f}%")

# Per-class F1 scores
per_class_f1 = f1_score(y_true, y_pred_classes, average=None, zero_division=0)
print(f"\n[RESULTS] Per-Class F1 Scores:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {per_class_f1[i]*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("\n[RESULTS] Confusion Matrix:")
print(cm)

# Check prediction distribution
unique_preds_final = np.unique(y_pred_classes)
print(f"\n[INFO] Final: Model predicts {len(unique_preds_final)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds_final}")
if len(unique_preds_final) == NUM_CLASSES:
    print("  ✅ EXCELLENT: Model predicts all classes!")
else:
    print(f"  ⚠️  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds_final)}")

# Save final model
final_model_path = 'models/skin_disease_model_5class_efficientnetb3_macro_f1.keras'
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss
axes[0, 0].plot(history_phase1.history['loss'], label='Train (Phase 1)')
axes[0, 0].plot(history_phase1.history['val_loss'], label='Val (Phase 1)')
axes[0, 0].plot(history_finetune.history['loss'], label='Train (Phase 2)')
axes[0, 0].plot(history_finetune.history['val_loss'], label='Val (Phase 2)')
axes[0, 0].set_title('Model Loss (Class-Balanced Focal Loss)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(history_phase1.history['accuracy'], label='Train (Phase 1)')
axes[0, 1].plot(history_phase1.history['val_accuracy'], label='Val (Phase 1)')
axes[0, 1].plot(history_finetune.history['accuracy'], label='Train (Phase 2)')
axes[0, 1].plot(history_finetune.history['val_accuracy'], label='Val (Phase 2)')
axes[0, 1].set_title('Model Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Macro F1
axes[1, 0].plot(history_phase1.history['macro_f1_metric'], label='Train (Phase 1)')
axes[1, 0].plot(history_phase1.history['val_macro_f1_metric'], label='Val (Phase 1)')
axes[1, 0].plot(history_finetune.history['macro_f1_metric'], label='Train (Phase 2)')
axes[1, 0].plot(history_finetune.history['val_macro_f1_metric'], label='Val (Phase 2)')
axes[1, 0].set_title('Macro F1 Score')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Macro F1')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix (Test Set)')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

plt.tight_layout()
plot_path = 'models/training_history_skin_5class_efficientnetb3_macro_f1.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Training plots saved to: {plot_path}")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"[FILES]")
print(f"  Final Model: {final_model_path}")
print(f"  Training Plot: {plot_path}")
print(f"\n[METRICS]")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Test Macro F1: {macro_f1*100:.2f}%")
print(f"  Test Macro F1 (sklearn): {macro_f1_sklearn*100:.2f}%")
print(f"  Test Weighted F1: {weighted_f1_sklearn*100:.2f}%")
print(f"\n[PER-CLASS F1]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {per_class_f1[i]*100:.2f}%")
print(f"\n[CLASSES]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {i+1}. {class_name}")
print("\n[IMPROVEMENTS]")
print("  ✅ EfficientNetB3 (medical imaging optimized)")
print("  ✅ RGB input (dermatoscopic images)")
print("  ✅ Class-Balanced Focal Loss (handles class imbalance & hard examples)")
print("  ✅ Macro F1 Metric (monitoring)")
print("  ✅ Equal importance to all classes")
print("  ✅ Medical safety optimized")
print("  ✅ Two-phase training strategy")
if len(unique_preds_final) == NUM_CLASSES:
    print("  ✅ Model predicts all classes - Excellent!")

