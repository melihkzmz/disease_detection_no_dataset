#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bone Disease Detection - 4 CLASS MACRO F1 TRAINING
Model: DenseNet121 (Medical Imaging Optimized)
Optimized for: Class Imbalance + Medical Safety + Professional Standards
Uses: Macro F1 Loss + Macro F1 Metric
Color Mode: Grayscale (X-Ray optimized)
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
try:
    import cv2
    CLAHE_AVAILABLE = True
except ImportError:
    print("[WARNING] OpenCV (cv2) not found. CLAHE will be disabled. Install with: pip install opencv-python")
    CLAHE_AVAILABLE = False

# Enable Mixed Precision Training (reduces memory usage by ~50%)
# NOT: Mixed precision Windows'ta model yükleme sorunlarına yol açıyor
# Bu yüzden kapatıldı - eğitim biraz daha yavaş olacak ama model sorunsuz yüklenecek
# policy = keras.mixed_precision.Policy('mixed_float16')
# keras.mixed_precision.set_global_policy(policy)
# print("[MEMORY] Mixed Precision Training enabled (float16) - Memory usage reduced by ~50%")
print("[MEMORY] Mixed Precision Training DISABLED for Windows compatibility")

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("\n" + "="*70)
print("🦴 BONE DISEASE DETECTION - 4 CLASS MACRO F1 TRAINING")
print("="*70)
print("✅ Model: DenseNet121 (Medical Imaging Optimized)")
print("✅ Optimized for: Class Imbalance + Medical Safety")
print("✅ Uses: Macro F1 Loss + Macro F1 Metric")
print("✅ Color Mode: Grayscale (X-Ray optimized)")
print("✅ Professional Standard for Medical AI")
print("="*70)

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
USE_GPU = False

if len(physical_devices) > 0:
    print(f"\n[GPU] {len(physical_devices)} GPU(s) available")
    for device in physical_devices:
        print(f"  - {device}")
    
    # Try to configure GPU, fallback to CPU if fails
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[GPU] Memory growth enabled")
        
        # Test GPU with a simple operation
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0])
            _ = tf.reduce_sum(test_tensor)
        USE_GPU = True
        print("[GPU] ✅ GPU test passed, using GPU for training")
    except Exception as e:
        print(f"[GPU] ⚠️  GPU configuration failed: {e}")
        print("[GPU] 🔄 Falling back to CPU mode")
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        USE_GPU = False
        # Reinitialize TensorFlow
        tf.config.set_visible_devices([], 'GPU')
else:
    print("\n[CPU] No GPU found, training on CPU")
    USE_GPU = False

# Hyperparameters
TRAIN_DIR = 'datasets/bone/Bone_4Class_Final/train'
VAL_DIR = 'datasets/bone/Bone_4Class_Final/val'
TEST_DIR = 'datasets/bone/Bone_4Class_Final/test'

IMG_SIZE = (384, 384)  # Optimized for medical imaging
BATCH_SIZE = 16  # Reduced from 16 to prevent OOM (Out of Memory) errors
INITIAL_EPOCHS = 150
FINE_TUNE_EPOCHS = 80
LEARNING_RATE = 0.0001  # Reduced from 0.0005 for better stability with Macro F1 loss
FINE_TUNE_LR = 0.00001  # Reduced from 0.00002 to prevent overfitting in Phase 2 (slower, more stable learning)
COLOR_MODE = 'rgb'  # Load as RGB (preprocessing handles grayscale→RGB conversion and CLAHE)

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
print(f"  Color Mode: {COLOR_MODE} (X-Ray optimized)")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")

# ============================================================================
# FOCAL LOSS + MACRO F1 HYBRID LOSS FUNCTION
# ============================================================================

def focal_macro_f1_loss(y_true, y_pred, alpha=0.75, gamma=2.5):
    """
    Focal Loss + Macro F1 Hybrid Loss Function (AGGRESSIVE for tumors)
    
    Combines Focal Loss (for hard examples and class imbalance) 
    with Macro F1 (for balanced class evaluation).
    
    Focal Loss focuses on hard examples and reduces easy example loss.
    Macro F1 ensures all classes (especially minorities) are considered.
    
    UPDATED parameters after data augmentation:
    - alpha=0.75 (higher than default 0.25) - still focuses on hard examples
    - gamma=2.5 (higher than default 2.0) - focuses on hard examples
    - Note: Malignant_Tumor is now the largest class (~45%), so less aggressive class-specific alpha
    
    Args:
        alpha: Weighting factor for rare class (default: 0.75 - aggressive)
        gamma: Focusing parameter (default: 2.5 - more focus on hard examples)
    
    Returns:
        Combined Focal + Macro F1 Loss
    """
    # Clip predictions to prevent numerical issues
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Class-specific alpha values (updated after adding more malignant data)
    # Now Malignant_Tumor is the largest class (~45%), so less aggressive alpha needed
    # Normal: low (reference class)
    # Fracture: medium
    # Benign_Tumor: medium-high (still important to detect)
    # Malignant_Tumor: medium-high (no longer the minority, but still critical)
    alpha_per_class = [0.25, 0.4, 0.5, 0.5]  # [Normal, Fracture, Benign_Tumor, Malignant_Tumor]
    
    # Calculate focal loss component for each sample
    # Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t)
    focal_losses = []
    
    for class_id in range(NUM_CLASSES):
        y_true_class = y_true[:, class_id]
        y_pred_class = y_pred[:, class_id]
        
        # Calculate p_t (probability of true class)
        p_t = y_true_class * y_pred_class + (1.0 - y_true_class) * (1.0 - y_pred_class)
        
        # Focal Loss component with class-specific alpha
        alpha_t = alpha_per_class[class_id] if class_id < len(alpha_per_class) else alpha
        focal_weight = alpha_t * tf.pow((1.0 - p_t), gamma)
        ce = -tf.math.log(p_t + 1e-8)
        focal_loss = focal_weight * ce
        
        focal_losses.append(tf.reduce_mean(focal_loss))
    
    focal_component = tf.reduce_mean(tf.stack(focal_losses))
    
    # Calculate Macro F1 component
    f1_scores = []
    for class_id in range(NUM_CLASSES):
        y_true_class = y_true[:, class_id]
        y_pred_class = y_pred[:, class_id]
        
        tp = tf.reduce_sum(y_true_class * y_pred_class)
        fp = tf.reduce_sum((1.0 - y_true_class) * y_pred_class)
        fn = tf.reduce_sum(y_true_class * (1.0 - y_pred_class))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    
    macro_f1 = tf.reduce_mean(tf.stack(f1_scores))
    macro_f1_component = 1.0 - macro_f1
    
    # Combine: 70% Focal Loss + 30% Macro F1 (adjustable)
    combined_loss = 0.7 * focal_component + 0.3 * macro_f1_component
    
    return combined_loss

# ============================================================================
# SOFT (DIFFERENTIABLE) MACRO F1 LOSS FUNCTION (backward compatibility)
# ============================================================================

def soft_macro_f1_loss(y_true, y_pred):
    """
    Soft (Differentiable) Macro F1 Score Loss Function
    
    Probability-based (soft) F1 score hesaplar - tamamen differentiable.
    Hard predictions (argmax) yerine soft probabilities kullanır.
    
    Advantages:
    - Fully differentiable (gradient flow better)
    - More stable training
    - Better for imbalanced datasets
    - Works directly with probabilities
    
    Returns:
        1 - soft_macro_f1 (loss minimize edilir, F1 maximize edilir)
    """
    # Clip predictions to prevent numerical issues
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Calculate soft F1 for each class
    f1_scores = []
    
    for class_id in range(NUM_CLASSES):
        # True labels for this class (one-hot -> probability)
        y_true_class = y_true[:, class_id]  # Shape: (batch_size,)
        
        # Predicted probability for this class
        y_pred_class = y_pred[:, class_id]  # Shape: (batch_size,)
        
        # Soft True Positives: y_true * y_pred (expected value)
        tp = tf.reduce_sum(y_true_class * y_pred_class)
        
        # Soft False Positives: (1 - y_true) * y_pred
        fp = tf.reduce_sum((1.0 - y_true_class) * y_pred_class)
        
        # Soft False Negatives: y_true * (1 - y_pred)
        fn = tf.reduce_sum(y_true_class * (1.0 - y_pred_class))
        
        # Calculate Soft Precision and Recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # Calculate Soft F1 Score
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    
    # Calculate Macro F1 (average of all class F1 scores)
    soft_macro_f1 = tf.reduce_mean(tf.stack(f1_scores))
    
    # Return loss (1 - F1), so minimizing loss maximizes F1
    return 1.0 - soft_macro_f1

# ============================================================================
# CUSTOM LAYER: Grayscale to RGB Converter
# ============================================================================

class GrayscaleToRGB(layers.Layer):
    """
    Custom layer to convert grayscale (1 channel) to RGB (3 channels).
    This is serializable and can be loaded without Lambda layer issues.
    """
    def __init__(self, **kwargs):
        super(GrayscaleToRGB, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Repeat grayscale channel 3 times to create RGB
        return tf.repeat(inputs, 3, axis=-1)
    
    def get_config(self):
        config = super(GrayscaleToRGB, self).get_config()
        return config

# ============================================================================
# STREAMING MACRO F1 METRIC (Global TP/FP/FN Accumulation)
# ============================================================================

class StreamingMacroF1(keras.metrics.Metric):
    """
    Streaming Macro F1 Metric that accumulates TP/FP/FN across all batches.
    This matches sklearn's macro F1 calculation on the full dataset.
    """
    def __init__(self, num_classes=4, name='macro_f1_metric', **kwargs):
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
        """
        Update TP, FP, FN counts for current batch.
        Vectorized computation for all classes at once.
        """
        # Convert to class indices
        y_true_classes = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
        y_pred_classes = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        
        # Vectorized computation: calculate TP/FP/FN for all classes simultaneously
        # Shape: (num_classes, batch_size)
        y_true_one_hot = tf.one_hot(y_true_classes, depth=self.num_classes, dtype=tf.float32)  # (batch_size, num_classes)
        y_pred_one_hot = tf.one_hot(y_pred_classes, depth=self.num_classes, dtype=tf.float32)  # (batch_size, num_classes)
        
        # True positives: (y_true == class) AND (y_pred == class)
        # Shape: (num_classes,) - sum over batch dimension
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)  # (num_classes,)
        
        # False positives: (y_true != class) AND (y_pred == class)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)  # (num_classes,)
        
        # False negatives: (y_true == class) AND (y_pred != class)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)  # (num_classes,)
        
        # Update state variables (vectorized update)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        """
        Calculate macro F1 from accumulated TP/FP/FN.
        Vectorized computation for all classes.
        """
        # Vectorized calculation for all classes at once
        # Shape: (num_classes,)
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_scores = 2.0 * precision * recall / (precision + recall + 1e-8)
        
        # Macro F1: average of per-class F1 scores
        macro_f1 = tf.reduce_mean(f1_scores)
        return macro_f1
    
    def reset_state(self):
        """
        Reset all state variables at the start of each epoch.
        Vectorized reset.
        """
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


# ============================================================================
# SKLEARN MACRO F1 CALLBACK (Optional but Recommended)
# ============================================================================

class SklearnMacroF1Callback(Callback):
    """
    Callback to compute sklearn's macro F1 at the end of each epoch on validation set.
    This provides ground truth comparison to ensure metric correctness.
    """
    def __init__(self, val_generator, num_classes=4, verbose=1):
        super(SklearnMacroF1Callback, self).__init__()
        self.val_generator = val_generator
        self.num_classes = num_classes
        self.verbose = verbose
        
        # Store sklearn F1 scores
        self.sklearn_f1_scores = []
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Compute sklearn macro F1 on full validation set.
        """
        # Reset generator
        self.val_generator.reset()
        
        # Collect all predictions and true labels
        y_true_all = []
        y_pred_all = []
        
        for batch_idx in range(len(self.val_generator)):
            batch_x, batch_y = self.val_generator[batch_idx]
            y_pred_batch = self.model.predict(batch_x, verbose=0)
            
            y_true_all.append(np.argmax(batch_y, axis=1))
            y_pred_all.append(np.argmax(y_pred_batch, axis=1))
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        
        # Calculate sklearn macro F1
        sklearn_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.sklearn_f1_scores.append(sklearn_macro_f1)
        
        # Add to logs
        logs = logs or {}
        logs['val_sklearn_macro_f1'] = sklearn_macro_f1
        
        if self.verbose > 0:
            print(f'\n[Sklearn Macro F1] Epoch {epoch + 1}: {sklearn_macro_f1:.4f} ({sklearn_macro_f1*100:.2f}%)')


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def apply_clahe_grayscale(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale images.
    CLAHE improves contrast in X-Ray images by enhancing local contrast.
    
    Args:
        img: numpy array of shape (H, W, 1) with values in [0, 255]
    
    Returns:
        Enhanced image with CLAHE applied
    """
    if not CLAHE_AVAILABLE:
        return img
    
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img
    
    # Extract single channel for grayscale
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1:
        img_2d = img_uint8[:, :, 0]
    elif len(img_uint8.shape) == 2:
        img_2d = img_uint8
    else:
        # If RGB, convert to grayscale first
        img_2d = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    img_clahe = clahe.apply(img_2d)
    
    # Reshape back to (H, W, 1) if needed
    if len(img.shape) == 3:
        img_clahe = np.expand_dims(img_clahe, axis=-1)
    
    return img_clahe


def preprocess_image_clahe_normalize(img):
    """
    Combined preprocessing: CLAHE + Official DenseNet121 ImageNet Preprocessing
    
    Process:
    1. Apply CLAHE for contrast enhancement (if grayscale and available)
    2. Convert grayscale to RGB if needed (for preprocessing function)
    3. Apply official DenseNet121 ImageNet preprocessing (matches pretrained weights)
    
    Official preprocessing:
    - Scales pixels to [0, 1] range
    - Applies ImageNet mean/std normalization
    - This matches exactly what the pretrained DenseNet121 expects
    
    Args:
        img: numpy array with values in [0, 255] (uint8 or float)
               Shape: (H, W) or (H, W, 1) for grayscale, (H, W, 3) for RGB
    
    Returns:
        Preprocessed image ready for DenseNet121 (float32, normalized)
    """
    # Ensure input is in [0, 255] range and uint8 format
    if img.max() <= 1.0:
        # If already normalized to [0, 1], scale back to [0, 255]
        img = (img * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Detect if image is grayscale (by shape, not COLOR_MODE)
    # ImageDataGenerator with COLOR_MODE='rgb' will convert grayscale to RGB automatically,
    # but we check the actual image to apply CLAHE if it's effectively grayscale
    is_grayscale = False
    if len(img.shape) == 2:
        # (H, W) - definitely grayscale
        is_grayscale = True
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            # (H, W, 1) - grayscale
            is_grayscale = True
        elif img.shape[2] == 3:
            # (H, W, 3) - check if it's actually grayscale (all channels same)
            # If it's a grayscale image loaded as RGB, all channels will be identical
            if np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
                is_grayscale = True
    
    # Apply CLAHE if grayscale and available
    if is_grayscale and CLAHE_AVAILABLE:
        img = apply_clahe_grayscale(img)
        # Ensure still uint8 after CLAHE
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert grayscale to RGB if needed (preprocessing function expects RGB)
    if len(img.shape) == 2:
        # (H, W) -> (H, W, 3)
        img = np.stack([img] * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        # (H, W, 1) -> (H, W, 3)
        img = np.repeat(img, 3, axis=-1)
    # If already RGB (H, W, 3), keep as is
    
    # Apply official DenseNet121 ImageNet preprocessing
    # This handles: scaling to [0,1] and ImageNet mean/std normalization
    img_preprocessed = densenet_preprocess(img)
    
    return img_preprocessed


# ============================================================================
# DATA GENERATORS
# ============================================================================

print("\n[DATA] Creating X-ray optimized data generators...")
print(f"  [PREPROCESSING] CLAHE: {'Enabled (auto-detect grayscale)' if CLAHE_AVAILABLE else 'Disabled'}")
print(f"  [PREPROCESSING] Normalization: Official DenseNet121 ImageNet preprocessing (matches pretrained weights)")
print(f"  [DATA] Loading images as RGB (preprocessing handles grayscale detection and conversion)")

# Enhanced augmentation for medical imaging (X-Ray specific)
# Aggressively increased augmentation to combat severe overfitting while preserving medical image integrity
# Preprocessing: CLAHE (Contrast Enhancement) + Official DenseNet121 ImageNet preprocessing
if COLOR_MODE == 'grayscale':
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image_clahe_normalize,  # CLAHE + Official ImageNet preprocessing
        rotation_range=22,         # Increased from 15 to 22 (more rotation diversity - max recommended for X-Ray)
        width_shift_range=0.2,     # Increased from 0.15 to 0.2 (more translation)
        height_shift_range=0.2,    # Increased from 0.15 to 0.2 (more translation)
        shear_range=0.11,          # Increased from 0.08 to 0.11 (more geometric variation)
        zoom_range=0.22,           # Increased from 0.15 to 0.22 (more zoom diversity)
        horizontal_flip=False,     # X-Ray için flip yok (anatomical correctness)
        vertical_flip=False,       # X-Ray için flip yok (anatomical correctness)
        fill_mode='constant',
        cval=0.0,                  # Black background for X-Ray
        brightness_range=[0.85, 1.15],  # Increased from [0.9, 1.1] to [0.85, 1.15] (more intensity variation)
    )
else:
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image_clahe_normalize,  # Official ImageNet preprocessing (CLAHE auto-detects grayscale)
        rotation_range=22,         # Increased from 15 to 22 (more rotation diversity)
        width_shift_range=0.2,     # Increased from 0.15 to 0.2 (more translation)
        height_shift_range=0.2,    # Increased from 0.15 to 0.2 (more translation)
        shear_range=0.11,          # Increased from 0.08 to 0.11 (more geometric variation)
        zoom_range=0.22,           # Increased from 0.15 to 0.22 (more zoom diversity)
        channel_shift_range=0.15,  # Added for RGB channel variation (color/intensity shifts)
        horizontal_flip=False,     # X-Ray için flip yok (anatomical correctness)
        vertical_flip=False,       # X-Ray için flip yok (anatomical correctness)
        fill_mode='constant',
        cval=0.0,                  # Black background for X-Ray
        brightness_range=[0.85, 1.15],  # Increased from [0.9, 1.1] to [0.85, 1.15] (more intensity variation)
    )

# Validation and test: Apply same preprocessing (CLAHE + [-1, +1] normalization)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_image_clahe_normalize)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image_clahe_normalize)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True,
    seed=42,
    color_mode=COLOR_MODE
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    color_mode=COLOR_MODE
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    color_mode=COLOR_MODE
)

print(f"\n[DATA] Generators created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# Data verification
print("\n[DATA VERIFICATION] Checking data diversity...")
print("  Class indices mapping:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"    Index {i}: {class_name}")
print("  Generator class indices:", train_generator.class_indices)

# Calculate class distribution
print("\n[DATA] Class distribution:")
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)

for i, class_name in enumerate(CLASS_NAMES):
    percentage = (class_counts[i] / total_samples * 100) if total_samples > 0 else 0
    print(f"  {class_name}: {class_counts[i]} ({percentage:.1f}%)")

max_count = np.max(class_counts)
min_count = np.min(class_counts)
imbalance_ratio = max_count / min_count if min_count > 0 else 0
print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}:1")

# Calculate class weights for imbalanced dataset
# UPDATED: After adding more malignant data, Malignant_Tumor is now the largest class (~45%)
# So we use more balanced weights instead of aggressive weighting
print("\n[CLASS WEIGHTS] Computing balanced class weights (updated after data augmentation)...")
class_weights_balanced = compute_class_weight(
    'balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_generator.classes
)

# Balanced weighting strategy (Malignant_Tumor is no longer the minority)
# Updated weights based on current class distribution
class_weight_dict = {}
for i, class_name in enumerate(CLASS_NAMES):
    balanced_weight = float(class_weights_balanced[i])
    
    # Balanced weighting - all classes are reasonably represented now
    if 'Malignant_Tumor' in class_name:
        # Malignant_Tumor: now largest class (~45%), use balanced weight with slight emphasis
        class_weight_dict[i] = min(balanced_weight * 1.2, 2.0)  # Max 2.0x, slightly higher than balanced
    elif 'Benign_Tumor' in class_name:
        # Benign_Tumor: use balanced weight with slight emphasis (still important)
        class_weight_dict[i] = min(balanced_weight * 1.3, 2.5)  # Max 2.5x
    elif 'Normal' in class_name:
        # Normal: reference class (smallest now at ~21%)
        class_weight_dict[i] = balanced_weight  # Use balanced weight
    elif 'Fracture' in class_name:
        # Fracture: slightly below average (~17%)
        class_weight_dict[i] = min(balanced_weight * 1.1, 2.0)  # Max 2.0x
    else:
        # Others: use balanced weight
        class_weight_dict[i] = balanced_weight
    
    print(f"  {class_name}: {class_weight_dict[i]:.2f}")

print("\n  ✅ Balanced class weights applied (updated for new data distribution)")
print("  📊 Strategy: All classes now reasonably represented, balanced weighting used")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n[MODEL] Building DenseNet121 model...")

# Preprocessing function converts grayscale to RGB and applies ImageNet preprocessing
# So model always receives RGB input (3 channels)
# Let TensorFlow automatically place model on GPU if available, otherwise CPU
base_model = DenseNet121(
    input_shape=(*IMG_SIZE, 3),  # RGB input (preprocessing handles grayscale→RGB conversion)
    include_top=False,
    weights='imagenet'
)

print("  ✅ Model expects RGB input (3 channels)")
print("  ✅ Preprocessing handles grayscale detection, CLAHE, and RGB conversion")
print("  ✅ Official DenseNet121 ImageNet preprocessing applied")

# Phase 1: Unfreeze top layers
# DenseNet121 için optimal unfreeze stratejisi
base_model.trainable = True
# DenseNet121'de ~400+ layer var, top 150'yi unfreeze et
freeze_until_phase1 = len(base_model.layers) - 150
for layer in base_model.layers[:freeze_until_phase1]:
    layer.trainable = False

print(f"\n[MODEL] Phase 1 Configuration:")
print(f"  Model: DenseNet121")
print(f"  Input Shape: {(*IMG_SIZE, 3)} (RGB)")
print(f"  Unfrozen layers: {len([l for l in base_model.layers if l.trainable])}")
print(f"  Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")

# Build model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Increased from 0.3 (batch size 16 allows higher dropout to combat overfitting)
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  # Increased L2 from 0.0001 to 0.001 (penalizes large weights to prevent overfitting)
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Increased from 0.3 (batch size 16 allows higher dropout)
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  # Increased L2 from 0.0001 to 0.001 (encourages evenly distributed weights)
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # Increased from 0.3 (slightly less aggressive near output layer)
    layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')  # float32 for mixed precision
])

# ============================================================================
# COMPILE MODEL - MACRO F1 LOSS
# ============================================================================

print("\n[MODEL] Compiling with CATEGORICAL CROSSENTROPY + CLASS WEIGHTS...")

# Prepare metrics list with streaming macro F1 metric
streaming_macro_f1 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list = [
    'accuracy',
    streaming_macro_f1,  # Streaming metric that accumulates TP/FP/FN globally (matches sklearn)
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',  # Stable loss function for batch size 8
    metrics=metrics_list
)

print("\n[MODEL] Model architecture:")
model.summary()
print(f"\n[MODEL] Total parameters: {model.count_params():,}")
print(f"\n[MODEL] ✅ Using CATEGORICAL CROSSENTROPY + CLASS WEIGHTS (stable gradients)")
print(f"[MODEL] ✅ Streaming Macro F1 metric: Accumulates TP/FP/FN globally (matches sklearn)")
print(f"[MODEL] ✅ Sklearn Macro F1 callback: Validates metric correctness each epoch")

# ============================================================================
# CALLBACKS
# ============================================================================

os.makedirs('models', exist_ok=True)

checkpoint_initial = ModelCheckpoint(
    'models/bone_4class_densenet121_macro_f1_initial.keras',
    monitor='val_macro_f1_metric',  # ← MACRO F1 izleniyor!
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_macro_f1_metric',  # ← MACRO F1 izleniyor!
    patience=25,  # Increased from 20 to allow more epochs for improvement (prevent too early stopping)
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.003  # Reduced from 0.01 to allow smaller improvements (Macro F1 can be sensitive)
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_macro_f1_metric',  # ← MACRO F1 izleniyor!
    factor=0.3,
    patience=15,
    min_lr=1e-8,
    verbose=1,
    mode='max',
    cooldown=5
)

# ============================================================================
# PHASE 1: INITIAL TRAINING
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (Top 150 Layers Unfrozen)")
print("Using: CATEGORICAL CROSSENTROPY + CLASS WEIGHTS + MACRO F1 METRIC")
print("Strategy: Stable gradients with class weights for imbalance handling")
print("Note: Macro F1 is monitored as metric, not used as loss (stable training)")
print("="*70)

# Add sklearn macro F1 callback for validation
sklearn_macro_f1_callback = SklearnMacroF1Callback(
    val_generator=val_generator,
    num_classes=NUM_CLASSES,
    verbose=1
)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight_dict,  # Use class weights with Macro F1 for better minority class learning
    callbacks=[checkpoint_initial, early_stopping, reduce_lr, sklearn_macro_f1_callback],
    verbose=1
)

# Load best model from phase 1
print("\n[INFO] Loading best model from Phase 1...")
model = keras.models.load_model(
    'models/bone_4class_densenet121_macro_f1_initial.keras',
    custom_objects={
        'GrayscaleToRGB': GrayscaleToRGB,  # Custom layer
        'StreamingMacroF1': StreamingMacroF1,  # Streaming macro F1 metric
        'focal_macro_f1_loss': focal_macro_f1_loss,  # Backward compatibility
        'soft_macro_f1_loss': soft_macro_f1_loss,  # Backward compatibility
        'macro_f1_loss': soft_macro_f1_loss,  # Backward compatibility
    }
)

# Check intermediate results
print("\n[EVAL] Phase 1 Results:")
test_generator.reset()
results_phase1 = model.evaluate(test_generator, verbose=0)

# Find metric indices
metric_names = model.metrics_names
print(f"  Metric names: {metric_names}")

loss_idx = metric_names.index('loss')
accuracy_idx = metric_names.index('accuracy')
macro_f1_idx = metric_names.index('macro_f1_metric') if 'macro_f1_metric' in metric_names else -1

print(f"\n  Test Loss (Macro F1 Loss): {results_phase1[loss_idx]:.4f}")
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

# Calculate Macro F1 using sklearn for verification
macro_f1_sklearn = f1_score(y_true, y_pred_classes_phase1, average='macro', zero_division=0)
print(f"  Macro F1 (sklearn): {macro_f1_sklearn*100:.2f}%")

# ============================================================================
# PHASE 2: FINE-TUNING
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (All Layers Unfrozen)")
print("Using: CATEGORICAL CROSSENTROPY + CLASS WEIGHTS + MACRO F1 METRIC")
print("Strategy: Stable gradients with class weights for imbalance handling")
print("="*70)

base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = True

print(f"\n[MODEL] Unfrozen layers: {len([l for l in base_model.layers if l.trainable])}")
print(f"[MODEL] Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")

# Recompile with lower learning rate (reduced to prevent overfitting)
# Create new streaming metric instance for Phase 2
streaming_macro_f1_phase2 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_phase2 = [
    'accuracy',
    streaming_macro_f1_phase2,  # Streaming metric for Phase 2
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',  # Stable loss function (same as Phase 1)
    metrics=metrics_list_phase2
)

checkpoint_finetune = ModelCheckpoint(
    'models/bone_4class_densenet121_macro_f1_finetuned.keras',
    monitor='val_macro_f1_metric',  # ← MACRO F1 izleniyor!
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping_finetune = EarlyStopping(
    monitor='val_macro_f1_metric',  # ← MACRO F1 izleniyor!
    patience=20,  # Reduced from 60 to stop earlier when overfitting starts (Phase 2 is more prone to overfitting)
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.002  # Increased from 0.001 to require more significant improvement
)

# Add sklearn macro F1 callback for Phase 2 validation
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
    class_weight=class_weight_dict,  # Use class weights with Macro F1 for better minority class learning
    callbacks=[checkpoint_finetune, early_stopping_finetune, reduce_lr, sklearn_macro_f1_callback_phase2],
    verbose=1
)

# Load best fine-tuned model
print("\n[INFO] Loading best fine-tuned model...")
model = keras.models.load_model(
    'models/bone_4class_densenet121_macro_f1_finetuned.keras',
    custom_objects={
        'GrayscaleToRGB': GrayscaleToRGB,  # Custom layer
        'StreamingMacroF1': StreamingMacroF1,  # Streaming macro F1 metric
        'focal_macro_f1_loss': focal_macro_f1_loss,  # Backward compatibility
        'soft_macro_f1_loss': soft_macro_f1_loss,  # Backward compatibility
        'macro_f1_loss': soft_macro_f1_loss,  # Backward compatibility
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
print(f"  Test Loss (Macro F1 Loss): {loss:.4f}")
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
final_model_path = 'models/bone_disease_model_4class_densenet121_macro_f1.keras'
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_initial = range(1, len(history_initial.history['loss']) + 1)
epochs_finetune = range(
    len(history_initial.history['loss']) + 1,
    len(history_initial.history['loss']) + len(history_finetune.history['loss']) + 1
)

# Macro F1 Metric
axes[0, 0].plot(
    epochs_initial, 
    history_initial.history.get('macro_f1_metric', []), 
    'b-', label='Train (Phase 1)', linewidth=2
)
axes[0, 0].plot(
    epochs_initial, 
    history_initial.history.get('val_macro_f1_metric', []), 
    'b--', label='Val (Phase 1)', linewidth=2
)
axes[0, 0].plot(
    epochs_finetune, 
    history_finetune.history.get('macro_f1_metric', []), 
    'r-', label='Train (Phase 2)', linewidth=2
)
axes[0, 0].plot(
    epochs_finetune, 
    history_finetune.history.get('val_macro_f1_metric', []), 
    'r--', label='Val (Phase 2)', linewidth=2
)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Macro F1 Score', fontsize=12)
axes[0, 0].set_title('Macro F1 Score (Primary Metric)', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(epochs_initial, history_initial.history['accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[0, 1].plot(epochs_initial, history_initial.history['val_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[0, 1].plot(epochs_finetune, history_finetune.history['accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[0, 1].plot(epochs_finetune, history_finetune.history['val_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Loss (Macro F1 Loss)
axes[1, 0].plot(epochs_initial, history_initial.history['loss'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_initial, history_initial.history['val_loss'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_finetune, history_finetune.history['loss'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[1, 0].plot(epochs_finetune, history_finetune.history['val_loss'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Loss (Macro F1 Loss)', fontsize=12)
axes[1, 0].set_title('Model Loss (Macro F1 Loss)', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=CLASS_NAMES, 
    yticklabels=CLASS_NAMES, 
    ax=axes[1, 1],
    cbar_kws={'label': 'Count'}
)
axes[1, 1].set_xlabel('Predicted', fontsize=12)
axes[1, 1].set_ylabel('True', fontsize=12)
axes[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
plt.setp(axes[1, 1].get_yticklabels(), rotation=0)

plt.tight_layout()
plot_path = 'models/training_history_bone_4class_densenet121_macro_f1.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"[SUCCESS] Training plots saved to: {plot_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"\n[FILES]")
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
print(f"\n[IMPROVEMENTS]")
print(f"  ✅ DenseNet121 (medical imaging optimized)")
print(f"  ✅ Grayscale input (X-Ray optimized)")
print(f"  ✅ Macro F1 Loss (professional standard)")
print(f"  ✅ Macro F1 Metric (monitoring)")
print(f"  ✅ Equal importance to all classes")
print(f"  ✅ Medical safety optimized")
print(f"  ✅ Two-phase training strategy")
if len(unique_preds_final) == NUM_CLASSES:
    print(f"\n  ✅ Model predicts all classes - Excellent!")
else:
    print(f"\n  ⚠️  Model does not predict all classes")
print("="*70 + "\n")

