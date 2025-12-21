#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KEMIK HASTALIKLARI TESPIT API
Flask API for Bone Disease Detection (4 Classes)
eye_disease_api.py'den adapte edildi
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import io
try:
    import cv2
    CLAHE_AVAILABLE = True
except ImportError:
    print("[UYARI] OpenCV (cv2) bulunamadi. CLAHE devre disi. Yüklemek için: pip install opencv-python")
    CLAHE_AVAILABLE = False
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import base64

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

app = Flask(__name__)
CORS(app)  # Frontend'den istek kabul etmek için

# Model configuration
# Keras 3.x için: .h5 formatı destekleniyor (önerilen)
# Sırayla kontrol et: .h5 > .keras
import os
# SavedModel formatında kaydedilmiş model (Windows uyumlu)
MODEL_PATH_SAVEDMODEL = 'models/bone_disease_model_4class_densenet121_macro_f1_savedmodel'
MODEL_PATH_KERAS = 'models/bone_disease_model_4class_densenet121_macro_f1.keras'
# Önce SavedModel'i dene, yoksa .keras'ı dene
MODEL_PATH = MODEL_PATH_SAVEDMODEL if os.path.exists(MODEL_PATH_SAVEDMODEL) else MODEL_PATH_KERAS
CLASS_NAMES = [
    'Normal',
    'Fracture',
    'Benign_Tumor',
    'Malignant_Tumor'
]

CLASS_NAMES_TR = {
    'Normal': 'Normal',
    'Fracture': 'Kırık',
    'Benign_Tumor': 'İyi Huylu Tümör',
    'Malignant_Tumor': 'Kötü Huylu Tümör'
}

CLASS_DESCRIPTIONS = {
    'Normal': 'Normal kemik yapısı - Anomali tespit edilmedi',
    'Fracture': 'Kırık - Kemik bütünlüğü bozulmuş',
    'Benign_Tumor': 'İyi huylu tümör - Kanserli olmayan anormal büyüme',
    'Malignant_Tumor': 'Kötü huylu tümör - Kanserli anormal büyüme, acil tıbbi müdahale gerekebilir'
}

IMG_SIZE = (384, 384)  # Eğitimde kullanılan boyut (train_bone_4class_macro_f1.py ile aynı)

# ============================================================================
# ÖZEL KATMAN VE METRİK SINIFLARI - Model yüklenirken gerekli
# ============================================================================

class GrayscaleToRGB(keras.layers.Layer):
    """
    Custom layer to convert grayscale (1 channel) to RGB (3 channels).
    Model yüklenirken gerekli - eğitim scriptinden alındı
    """
    def __init__(self, **kwargs):
        super(GrayscaleToRGB, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Repeat grayscale channel 3 times to create RGB
        return tf.repeat(inputs, 3, axis=-1)
    
    def get_config(self):
        config = super(GrayscaleToRGB, self).get_config()
        return config

class StreamingMacroF1(keras.metrics.Metric):
    """
    Streaming Macro F1 Metric - Model yüklenirken gerekli
    Eğitim scriptinden alındı
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
        """Update TP, FP, FN counts for current batch."""
        # Convert to class indices
        y_true_classes = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
        y_pred_classes = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        
        # Vectorized computation
        y_true_one_hot = tf.one_hot(y_true_classes, depth=self.num_classes, dtype=tf.float32)
        y_pred_one_hot = tf.one_hot(y_pred_classes, depth=self.num_classes, dtype=tf.float32)
        
        # True positives, false positives, false negatives
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
        """Reset all state variables."""
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

print("\n" + "="*70)
print("KEMIK HASTALIKLARI TESPIT API")
print("="*70)

# Load model
model = None
model_type = None  # 'keras', 'savedmodel', 'h5'

try:
    print(f"\n[YUKLENIYOR] Model: {MODEL_PATH}")
    # Özel katman ve metrik sınıflarını custom_objects ile belirt
    custom_objects = {
        'GrayscaleToRGB': GrayscaleToRGB,
        'StreamingMacroF1': StreamingMacroF1,
    }
    
    # SavedModel, .h5 veya .keras formatında yükle
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        if os.path.isdir(MODEL_PATH) or MODEL_PATH.endswith('_savedmodel'):
            # SavedModel formatı
            print(f"[YONTEM] SavedModel formatı kullanılıyor")
            saved_model = tf.saved_model.load(MODEL_PATH)
            # SavedModel'in serve fonksiyonunu bul
            if hasattr(saved_model, 'signatures') and 'serving_default' in saved_model.signatures:
                model = saved_model.signatures['serving_default']
                print("[BASARILI] Model yuklendi (SavedModel format - serving_default signature)")
            else:
                # Direkt callable olarak kullan
                model = saved_model
                print("[BASARILI] Model yuklendi (SavedModel format - callable)")
            model_type = 'savedmodel'
            # Test prediction için input shape'i bilinmiyor, atlanacak
            input_shape = (384, 384)  # Varsayılan
        elif MODEL_PATH.endswith('.h5'):
            print(f"[YONTEM] H5 formatı kullanılıyor")
            model = keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
            model_type = 'h5'
            print("[BASARILI] Model yuklendi (H5 format)")
            # Test prediction
            try:
                input_shape = model.input_shape[1:3] if hasattr(model, 'input_shape') and model.input_shape else (384, 384)
                test_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
                test_output = model.predict(test_input, verbose=0)
                print(f"[TEST] Model tahmin yapabiliyor: input shape {input_shape}, output shape {test_output.shape}")
            except Exception as test_err:
                print(f"[UYARI] Test prediction atlandi: {str(test_err)[:100]}")
        else:
            # .keras formatı
            print(f"[YONTEM] Keras formatı kullanılıyor")
            model = keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False
            )
            model_type = 'keras'
            print("[BASARILI] Model yuklendi (Keras format)")
            # Test prediction
            try:
                input_shape = model.input_shape[1:3] if hasattr(model, 'input_shape') and model.input_shape else (384, 384)
                test_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
                test_output = model.predict(test_input, verbose=0)
                print(f"[TEST] Model tahmin yapabiliyor: input shape {input_shape}, output shape {test_output.shape}")
            except Exception as test_err:
                print(f"[UYARI] Test prediction atlandi: {str(test_err)[:100]}")
    
    print(f"[BASARILI] Model yuklendi!")
    print(f"[BILGI] Model tipi: {model_type}")
    print(f"[BILGI] Sinif sayisi: {len(CLASS_NAMES)}")
    print(f"[BILGI] Siniflar: {', '.join(CLASS_NAMES)}")
    
except Exception as e:
    print(f"[HATA] Model yuklenemedi: {e}")
    print(f"[BILGI] Lutfen model dosyasinin yolunu kontrol edin: {MODEL_PATH}")
    error_msg = str(e)
    if len(error_msg) > 500:
        print(f"[HATA DETAY] {error_msg[:500]}...")
    sys.exit(1)

def apply_clahe_grayscale(img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale image"""
    if not CLAHE_AVAILABLE:
        return img
    
    # Ensure uint8
    if img.max() <= 1.0:
        img_uint8 = (img * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
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
    
    # Reshape back to (H, W, 1) if needed (preserve original shape)
    if len(img_uint8.shape) == 3:
        img_clahe = np.expand_dims(img_clahe, axis=-1)
    
    return img_clahe


def preprocess_image(image):
    """
    Preprocess image for model input - Eğitim scriptindeki ile AYNI
    
    Process:
    1. Resize to (384, 384)
    2. Convert PIL Image to numpy array
    3. Apply CLAHE if grayscale (matches training)
    4. Convert grayscale to RGB if needed
    5. Apply official DenseNet121 ImageNet preprocessing
    """
    # Resize to training size
    image = image.resize(IMG_SIZE)
    
    # Convert PIL Image to numpy array (values in [0, 255], uint8)
    img_array = np.array(image)
    
    # Detect if image is grayscale
    is_grayscale = False
    if len(img_array.shape) == 2:
        is_grayscale = True
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 1:
            is_grayscale = True
        elif img_array.shape[2] == 3:
            # Check if it's actually grayscale (all channels same)
            if np.allclose(img_array[:,:,0], img_array[:,:,1]) and np.allclose(img_array[:,:,1], img_array[:,:,2]):
                is_grayscale = True
    
    # Apply CLAHE if grayscale and available (matches training)
    if is_grayscale and CLAHE_AVAILABLE:
        img_array = apply_clahe_grayscale(img_array)
        # Ensure still uint8 after CLAHE
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert grayscale to RGB if needed (model expects RGB input)
    if len(img_array.shape) == 2:
        # (H, W) -> (H, W, 3)
        img_array = np.stack([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        # (H, W, 1) -> (H, W, 3)
        img_array = np.repeat(img_array, 3, axis=-1)
    # If already RGB (H, W, 3), keep as is
    
    # Apply official DenseNet121 ImageNet preprocessing
    # This handles: scaling to [0,1] and ImageNet mean/std normalization
    # Matches exactly what the pretrained DenseNet121 expects
    img_preprocessed = densenet_preprocess(img_array.astype(np.float32))
    
    # Add batch dimension
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    
    return img_preprocessed

# ============================================================================
# GRAD-CAM FUNCTIONS
# ============================================================================

def get_gradcam_model(base_model):
    """Grad-CAM için model oluştur - son convolutional layer'ı al"""
    # DenseNet121'in son convolutional block'u
    layer_names = [layer.name for layer in base_model.layers]
    
    # Son convolutional layer'ı bul - DenseNet121 için daha spesifik
    last_conv_layer_name = None
    
    # DenseNet121'de son convolutional layer genelde 'relu' ile biten son conv block
    # Önce 'conv5_block16_concat' veya benzer bir layer'ı bul
    for name in reversed(layer_names):
        if 'conv5_block16' in name and ('concat' in name or 'relu' in name):
            last_conv_layer_name = name
            break
    
    # İkinci seçenek: 'bn' (batch norm) öncesi son conv
    if last_conv_layer_name is None:
        for name in reversed(layer_names):
            if 'conv' in name.lower() and 'relu' in name.lower():
                # DenseNet block sonları genelde relu ile biter
                try:
                    layer = base_model.get_layer(name)
                    if len(layer.output_shape) == 4:  # (B, H, W, C)
                        last_conv_layer_name = name
                        break
                except:
                    continue
    
    # Fallback: Son 4D output veren layer
    if last_conv_layer_name is None:
        for i in range(len(base_model.layers) - 1, -1, -1):
            layer = base_model.layers[i]
            try:
                if len(layer.output_shape) == 4:  # (B, H, W, C)
                    last_conv_layer_name = layer.name
                    break
            except:
                continue
    
    if last_conv_layer_name is None:
        # Default DenseNet121
        last_conv_layer_name = 'conv5_block16_concat'
        print(f"[GRAD-CAM] UYARI: Default layer kullanılıyor: {last_conv_layer_name}")
    else:
        print(f"[GRAD-CAM] Son convolutional layer bulundu: {last_conv_layer_name}")
    
    # Grad-CAM model
    try:
        gradcam_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
    except Exception as e:
        print(f"[GRAD-CAM] Model oluşturma hatası: {e}")
        raise
    
    return gradcam_model, last_conv_layer_name

def compute_gradcam(model, img_array, pred_index=None, model_type=None):
    """Grad-CAM hesapla"""
    try:
        # Model tipi kontrolü
        if model_type == 'savedmodel':
            raise NotImplementedError("SavedModel formatı için Grad-CAM henüz desteklenmiyor. Keras model formatı (.keras veya .h5) kullanın.")
        
        # Keras model kontrolü
        if isinstance(model, keras.Model) or (hasattr(model, 'layers') and hasattr(model, 'predict')):
            # Full Keras model - base model'i bul
            base_model = None
            
            # Model'in ilk layer'ını kontrol et (genelde base model wrapper)
            # DenseNet121 için: model.layers[0] genelde base model wrapper'dır
            if len(model.layers) > 0:
                first_layer = model.layers[0]
                # Eğer ilk layer'da 'densenet' veya çok sayıda layer varsa, bu base model olabilir
                if hasattr(first_layer, 'layers') and len(first_layer.layers) > 10:
                    base_model = first_layer
                    print(f"[GRAD-CAM] Base model ilk layer'da bulundu: {first_layer.name} ({len(first_layer.layers)} layers)")
                else:
                    # Direkt base model olabilir - tüm model'i kullan
                    base_model = model
                    print(f"[GRAD-CAM] Model direkt base model olarak kullanılıyor ({len(model.layers)} layers)")
            else:
                base_model = model
            
            # Grad-CAM model oluştur
            print(f"[GRAD-CAM] Grad-CAM model oluşturuluyor...")
            gradcam_model, last_conv_layer_name = get_gradcam_model(base_model)
            
            img_tensor = tf.constant(img_array, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, predictions = gradcam_model(img_tensor, training=False)
                
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                else:
                    pred_index = tf.constant(pred_index, dtype=tf.int64)
                
                class_channel = predictions[:, pred_index]
            
            # Gradient hesapla
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Gradient'in None olup olmadığını kontrol et
            if grads is None:
                raise ValueError("Gradient hesaplanamadı - model trainable değil olabilir")
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weighted combination of activation maps
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # ReLU ve normalize
            heatmap = tf.maximum(heatmap, 0)
            heatmap_max = tf.reduce_max(heatmap)
            
            # Heatmap'in düzgün hesaplandığını kontrol et
            if heatmap_max < 1e-7:
                print("[GRAD-CAM] UYARI: Heatmap tüm değerleri sıfır veya çok küçük")
                raise ValueError("Heatmap hesaplanamadı - tüm değerler sıfır")
            
            heatmap = heatmap / heatmap_max
            heatmap = heatmap.numpy()
            
            # NaN veya Inf kontrolü
            if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
                print("[GRAD-CAM] UYARI: Heatmap NaN veya Inf içeriyor")
                raise ValueError("Heatmap geçersiz (NaN/Inf)")
            
            # Resize to original image size
            original_height = img_array.shape[1]
            original_width = img_array.shape[2]
            
            if CLAHE_AVAILABLE:
                heatmap = cv2.resize(heatmap, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
            else:
                from PIL import Image as PILImage
                heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8), mode='L')
                heatmap_pil = heatmap_pil.resize((original_width, original_height), PILImage.Resampling.LANCZOS)
                heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0
            
            return heatmap, img_array[0]
        else:
            # Diğer formatlar - desteklenmiyor
            raise TypeError(f"Model tipi desteklenmiyor. Model: {type(model)}, hasattr(layers): {hasattr(model, 'layers')}, hasattr(predict): {hasattr(model, 'predict')}")
    except Exception as e:
        print(f"[GRAD-CAM] Hesaplama hatası: {e}")
        import traceback
        traceback.print_exc()
        # Hata durumunda None döndür - çağıran kod bunu handle etsin
        raise

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Heatmap'i görüntü üzerine yerleştir"""
    # Heatmap normalize kontrolü
    if heatmap.max() - heatmap.min() < 1e-6:
        print(f"[OVERLAY] UYARI: Heatmap düz (min: {heatmap.min()}, max: {heatmap.max()})")
    
    # Heatmap'i 0-255 aralığına normalize et
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    
    # Colormap uygula
    if CLAHE_AVAILABLE:
        heatmap_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
        heatmap_rgb_normalized = heatmap_rgb.astype(np.float32) / 255.0
    else:
        # Basit grayscale to RGB
        heatmap_rgb_normalized = np.stack([heatmap_normalized] * 3, axis=-1)
    
    # Image normalize
    if img.max() > 1.0:
        img_normalized = img.astype(np.float32) / 255.0
    else:
        img_normalized = img.astype(np.float32)
    
    # Resize heatmap to match image dimensions
    if heatmap_rgb_normalized.shape[:2] != img_normalized.shape[:2]:
        if CLAHE_AVAILABLE:
            heatmap_rgb_normalized = cv2.resize(
                heatmap_rgb_normalized, 
                (img_normalized.shape[1], img_normalized.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            from PIL import Image as PILImage
            heatmap_pil = PILImage.fromarray((heatmap_rgb_normalized * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize((img_normalized.shape[1], img_normalized.shape[0]), PILImage.Resampling.LANCZOS)
            heatmap_rgb_normalized = np.array(heatmap_pil).astype(np.float32) / 255.0
    
    # Overlay - alpha blending
    superimposed = img_normalized * (1 - alpha) + heatmap_rgb_normalized * alpha
    superimposed = np.clip(superimposed, 0, 1)
    
    return superimposed

@app.route('/')
def status():
    """API status endpoint"""
    return jsonify({
        "status": "API calisiyor",
        "model": "DenseNet121 - Bone Disease Detection (4 Classes)",
        "version": "1.0",
        "classes": CLASS_NAMES,
        "classes_tr": [CLASS_NAMES_TR[c] for c in CLASS_NAMES],
        "num_classes": len(CLASS_NAMES),
        "image_size": IMG_SIZE,
        "endpoints": {
            "GET /": "API durumu",
            "POST /predict": "Goruntu tahmini (multipart/form-data, field: 'image')",
            "GET /classes": "Tum siniflari listele"
        }
    })

@app.route('/classes')
def list_classes():
    """List all disease classes with descriptions"""
    classes_info = []
    for cls in CLASS_NAMES:
        classes_info.append({
            "name": cls,
            "name_tr": CLASS_NAMES_TR.get(cls, cls),
            "description": CLASS_DESCRIPTIONS.get(cls, "No description available")
        })
    return jsonify({
        "total_classes": len(CLASS_NAMES),
        "classes": classes_info
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict bone disease from uploaded image"""
    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({"error": "Goruntu gonderilmedi"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "Dosya secilmedi"}), 400
    
    # File content'i sakla (Grad-CAM için tekrar kullanacağız)
    file_content = file.read()
    file_stream = io.BytesIO(file_content)
    
    try:
        # Read and preprocess image
        file_stream.seek(0)
        image = Image.open(file_stream).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Predict
        # SavedModel veya Keras model prediction
        if hasattr(model, 'predict'):
            # Keras model
            predictions = model.predict(processed_image, verbose=0)
        else:
            # SavedModel - callable veya signature function
            # Input için TensorFlow tensor oluştur
            input_tensor = tf.constant(processed_image, dtype=tf.float32)
            
            # Signature function ise dict döner
            if callable(model):
                predictions_tensor = model(input_tensor)
            else:
                predictions_tensor = model(input_tensor)
            
            # TensorFlow tensor'ı numpy array'e çevir
            if isinstance(predictions_tensor, dict):
                # Signature function dict döner, genelde 'output_0' veya ilk değer
                output_key = list(predictions_tensor.keys())[0]
                predictions = predictions_tensor[output_key].numpy()
            elif hasattr(predictions_tensor, 'numpy'):
                predictions = predictions_tensor.numpy()
            else:
                predictions = np.array(predictions_tensor)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(predictions[0][top_idx])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_results = []
        for i in top_3_indices:
            top_3_results.append({
                "class": CLASS_NAMES[i],
                "class_tr": CLASS_NAMES_TR.get(CLASS_NAMES[i], CLASS_NAMES[i]),
                "description": CLASS_DESCRIPTIONS.get(CLASS_NAMES[i], ""),
                "confidence": float(predictions[0][i]),
                "percentage": f"{predictions[0][i]*100:.2f}%"
            })
        
        # All predictions
        all_predictions = []
        for i, cls in enumerate(CLASS_NAMES):
            all_predictions.append({
                "class": cls,
                "class_tr": CLASS_NAMES_TR.get(cls, cls),
                "confidence": float(predictions[0][i]),
                "percentage": f"{predictions[0][i]*100:.2f}%"
            })
        
        result = {
            "success": True,
            "prediction": {
                "class": top_class,
                "class_tr": CLASS_NAMES_TR.get(top_class, top_class),
                "description": CLASS_DESCRIPTIONS.get(top_class, ""),
                "confidence": top_confidence,
                "percentage": f"{top_confidence*100:.2f}%"
            },
            "top_3": top_3_results,
            "all_predictions": all_predictions,
            "note": "Bu sonuclar yalnizca arastirma ve egitim amaclidir. Klinik tani icin degildir. Mutlaka uzman doktora basvurun."
        }
        
        # Grad-CAM isteğe bağlı
        if request.form.get('with_gradcam', 'false').lower() == 'true':
            try:
                # Original image'ı sakla (preprocessing öncesi)
                file_stream.seek(0)
                original_image = Image.open(file_stream).convert('RGB')
                original_array = np.array(original_image)
                
                # Grad-CAM hesapla
                print(f"[GRAD-CAM] Grad-CAM hesaplanıyor... Model tipi: {model_type}")
                heatmap, _ = compute_gradcam(model, processed_image, pred_index=int(top_idx), model_type=model_type)
                
                # Heatmap kontrolü - eğer düz ise (tüm değerler aynı), uyarı ver
                if heatmap is not None:
                    heatmap_unique = np.unique(heatmap)
                    if len(heatmap_unique) <= 2:  # Çok az farklı değer varsa
                        print(f"[GRAD-CAM] UYARI: Heatmap düz görünüyor (unique değer sayısı: {len(heatmap_unique)})")
                    else:
                        print(f"[GRAD-CAM] Heatmap başarıyla hesaplandı (min: {heatmap.min():.4f}, max: {heatmap.max():.4f}, unique: {len(heatmap_unique)})")
                
                # Overlay heatmap
                gradcam_img = overlay_heatmap(original_array, heatmap)
                
                # PIL Image'a çevir ve base64 encode
                gradcam_img_uint8 = (gradcam_img * 255).astype(np.uint8)
                gradcam_pil = Image.fromarray(gradcam_img_uint8)
                
                # Base64 encode
                buffer = io.BytesIO()
                gradcam_pil.save(buffer, format='PNG')
                gradcam_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                result["gradcam"] = f"data:image/png;base64,{gradcam_base64}"
            except NotImplementedError as e:
                # SavedModel formatı için özel mesaj
                print(f"[GRAD-CAM] Hata: {e}")
                result["gradcam_error"] = "Grad-CAM şu an sadece Keras model formatı (.keras veya .h5) için destekleniyor. SavedModel formatı için henüz desteklenmiyor."
            except Exception as e:
                print(f"[GRAD-CAM] Hata: {e}")
                import traceback
                traceback.print_exc()
                result["gradcam_error"] = f"Grad-CAM hesaplanamadı: {str(e)}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Tahmin hatasi: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("API SERVER BASLATILIYOR")
    print("="*70)
    print(f"\n[SERVER] Calisiyor: http://localhost:5002")
    print(f"[API] Durum: http://localhost:5002/")
    print(f"[API] Tahmin: POST http://localhost:5002/predict")
    print(f"[API] Siniflar: http://localhost:5002/classes")
    print("\n" + "="*70)
    print("\nFrontend'den kullanmak icin analyze.html dosyasini acin")
    print("Not: CORS etkin, localhost'tan gelen istekler kabul edilir")
    print("\n" + "="*70 + "\n")
    
    app.run(host='127.0.0.1', port=5002, debug=True)
