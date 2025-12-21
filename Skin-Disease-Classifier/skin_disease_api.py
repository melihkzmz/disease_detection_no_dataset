#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DERI HASTALIKLARI TESPIT API
Flask API for Skin Disease Detection (5 Classes)
Model: EfficientNetB3 - Macro F1 Optimized
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
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
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
# SavedModel formatında kaydedilmiş model (Keras 3.x uyumlu - önerilen)
MODEL_PATH_SAVEDMODEL = 'models/skin_disease_model_5class_efficientnetb3_macro_f1_savedmodel'
MODEL_PATH_KERAS = 'models/skin_disease_model_5class_efficientnetb3_macro_f1.keras'
# Önce SavedModel'i dene, yoksa .keras'ı dene (bone_disease_api.py ile aynı strateji)
MODEL_PATH = MODEL_PATH_SAVEDMODEL if os.path.exists(MODEL_PATH_SAVEDMODEL) else MODEL_PATH_KERAS

CLASS_NAMES = [
    'akiec',   # Actinic Keratoses
    'bcc',     # Basal Cell Carcinoma
    'bkl',     # Benign Keratosis
    'mel',     # Melanoma
    'nv'       # Melanocytic Nevi
]

CLASS_NAMES_TR = {
    'akiec': 'Aktinik Keratoz',
    'bcc': 'Bazal Hücreli Karsinom',
    'bkl': 'İyi Huylu Keratoz',
    'mel': 'Melanom',
    'nv': 'Melanositik Nevüs (Ben)'
}

CLASS_DESCRIPTIONS = {
    'akiec': 'Aktinik Keratoz - Güneş hasarına bağlı kanser öncesi lezyon, düzenli takip gerekebilir',
    'bcc': 'Bazal Hücreli Karsinom - En yaygın cilt kanseri türü, genellikle yavaş büyür ve nadiren yayılır',
    'bkl': 'İyi Huylu Keratoz - Kanserli olmayan cilt lezyonu, genellikle zararsızdır',
    'mel': 'Melanom - En tehlikeli cilt kanseri türü, erken teşhis kritiktir. Acil tıbbi değerlendirme önerilir',
    'nv': 'Melanositik Nevüs (Ben) - Genellikle zararsız ben, ancak değişiklik gösterirse kontrol edilmelidir'
}

IMG_SIZE = (300, 300)  # EfficientNetB3 input size

# ============================================================================
# CUSTOM METRIC CLASS - Model yüklenirken gerekli
# ============================================================================

class StreamingMacroF1(keras.metrics.Metric):
    """
    Streaming Macro F1 Metric - Model yüklenirken gerekli
    Eğitim scriptinden alındı
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
print("DERI HASTALIKLARI TESPIT API")
print("="*70)

# Load model
model = None
model_type = None  # 'keras', 'savedmodel', 'h5'

try:
    print(f"\n[YUKLENIYOR] Model: {MODEL_PATH}")
    custom_objects = {
        'StreamingMacroF1': StreamingMacroF1,
    }
    
    # SavedModel, .h5 veya .keras formatında yükle (bone_disease_api.py ile aynı strateji)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        if os.path.isdir(MODEL_PATH) or MODEL_PATH.endswith('_savedmodel'):
            # SavedModel formatı (Keras 3.x uyumlu - BONE API'NIN KULLANDIGI YONTEM)
            print(f"[YONTEM] SavedModel formatı kullanılıyor (Keras 3.x uyumlu)")
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
            input_shape = IMG_SIZE  # Varsayılan
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
                input_shape = model.input_shape[1:3] if hasattr(model, 'input_shape') and model.input_shape else IMG_SIZE
                test_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
                test_output = model.predict(test_input, verbose=0)
                print(f"[TEST] Model tahmin yapabiliyor: input shape {input_shape}, output shape {test_output.shape}")
            except Exception as test_err:
                print(f"[UYARI] Test prediction atlandi: {str(test_err)[:100]}")
        else:
            # .keras formatı (Keras 3.x ile sorunlu olabilir)
            print(f"[YONTEM] Keras formatı kullanılıyor (Keras 3.x uyumluluk sorunları olabilir)")
            try:
                # Try tf.keras first (more compatible)
                import tensorflow.keras as tf_keras
                model = tf_keras.models.load_model(
                    MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False
                )
                model_type = 'keras'
                print("[BASARILI] Model yuklendi (tf.keras ile)")
            except Exception as e1:
                print(f"[UYARI] tf.keras ile yukleme basarisiz: {str(e1)[:200]}")
                # Fallback to standard keras
                model = keras.models.load_model(
                    MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                model_type = 'keras'
                print("[BASARILI] Model yuklendi (keras ile)")
            
            # Test prediction
            try:
                input_shape = model.input_shape[1:3] if hasattr(model, 'input_shape') and model.input_shape else IMG_SIZE
                test_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
                test_output = model.predict(test_input, verbose=0)
                print(f"[TEST] Model tahmin yapabiliyor: input shape {input_shape}, output shape {test_output.shape}")
            except Exception as test_err:
                print(f"[UYARI] Test prediction atlandi: {str(test_err)[:100]}")
    
    print(f"[BASARILI] Model yuklendi!")
    print(f"[BILGI] Model tipi: {model_type}")
    print(f"[BILGI] Sinif sayisi: {len(CLASS_NAMES)}")
    print(f"[BILGI] Siniflar: {', '.join(CLASS_NAMES)}")
    print(f"[BILGI] Image size: {IMG_SIZE}")
    
    # Test prediction
    try:
        # Use EfficientNet preprocessing for test
        test_img = np.random.randint(0, 255, (IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        test_img_float = test_img.astype(np.float32)
        test_input = efficientnet_preprocess(test_img_float)
        test_input = np.expand_dims(test_input, axis=0)
        
        test_output = model.predict(test_input, verbose=0)
        print(f"[TEST] Model tahmin yapabiliyor: output shape {test_output.shape}")
        print(f"[TEST] Output sum: {test_output.sum():.4f} (should be ~1.0 for softmax)")
        if test_output.shape[1] != len(CLASS_NAMES):
            print(f"[UYARI] Model output shape ({test_output.shape[1]}) sinif sayisi ({len(CLASS_NAMES)}) ile uyusmuyor!")
    except Exception as test_err:
        error_msg = str(test_err)
        # Truncate very long error messages
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        print(f"[UYARI] Test prediction atlandi: {error_msg}")
    
except Exception as e:
    error_msg = str(e)
    # Truncate very long error messages (like verbose JSON dumps)
    if len(error_msg) > 1000:
        error_msg = error_msg[:1000] + "\n... (error message truncated)"
        print(f"\n[HATA] Model yuklenemedi (uzun hata mesaji kesildi)")
    else:
        print(f"\n[HATA] Model yuklenemedi: {error_msg}")
    
    print(f"[BILGI] Lutfen model dosyasinin yolunu kontrol edin: {MODEL_PATH}")
    print(f"[BILGI] Model dosyasi mevcut mu: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        print(f"[BILGI] Model dosya boyutu: {file_size:.2f} MB")
    
    # Check TensorFlow/Keras version
    print(f"[BILGI] TensorFlow version: {tf.__version__}")
    print(f"[BILGI] Keras version: {keras.__version__}")
    
    # Provide helpful suggestions
    print(f"\n[COZUM ONERILERI]")
    print(f"1. Model dosyasinin gecerli oldugundan emin olun")
    print(f"2. TensorFlow/Keras versiyon uyumlulugunu kontrol edin")
    print(f"3. Model dosyasini yeniden egiterek kaydedin")
    print(f"4. Alternatif: Model'i SavedModel formatina donusturun")
    
    # Only show full traceback if error is short
    if len(str(e)) < 500:
        import traceback
        traceback.print_exc()
    
    sys.exit(1)

def preprocess_image_efficientnet(image):
    """
    Preprocess image for EfficientNetB3 input - Eğitim scriptindeki ile AYNI
    
    Process:
    1. Resize to (300, 300)
    2. Convert PIL Image to numpy array
    3. Apply EfficientNet preprocessing (ImageNet normalization)
    """
    # Resize to training size
    image = image.resize(IMG_SIZE)
    
    # Convert PIL Image to numpy array (values in [0, 255], uint8)
    img_array = np.array(image)
    
    # Ensure RGB format
    if len(img_array.shape) == 2:
        # Grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA to RGB
        img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        # Single channel to RGB
        img_array = np.repeat(img_array, 3, axis=-1)
    
    # Ensure float32 in [0, 255] range
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    
    # EfficientNet preprocessing (handles ImageNet normalization internally)
    # This applies: scale to [0,1] then ImageNet mean/std normalization
    img_preprocessed = efficientnet_preprocess(img_array)
    
    # Add batch dimension
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    
    return img_preprocessed

# ============================================================================
# GRAD-CAM FUNCTIONS (Optional - for visualization)
# ============================================================================

def get_gradcam_model(base_model):
    """Grad-CAM için model oluştur - son convolutional layer'ı al"""
    # EfficientNetB3'ün son convolutional block'u
    layer_names = [layer.name for layer in base_model.layers]
    
    # Son convolutional layer'ı bul
    last_conv_layer_name = None
    
    # EfficientNetB3'te son conv layer genelde 'top_conv' veya 'block7a_expand_activation'
    for name in reversed(layer_names):
        if 'top_conv' in name or 'block7' in name:
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
        # Default EfficientNetB3
        last_conv_layer_name = 'top_conv'
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

def compute_gradcam(model, img_array, pred_index=None):
    """Grad-CAM hesapla"""
    try:
        # Base model'i bul (EfficientNetB3)
        base_model = None
        
        # Model'in ilk layer'ını kontrol et
        if len(model.layers) > 0:
            first_layer = model.layers[0]
            # EfficientNetB3 genelde ilk layer'da olur
            if hasattr(first_layer, 'layers') and len(first_layer.layers) > 10:
                base_model = first_layer
                print(f"[GRAD-CAM] Base model ilk layer'da bulundu: {first_layer.name}")
            else:
                base_model = model
                print(f"[GRAD-CAM] Model direkt base model olarak kullanılıyor")
        else:
            base_model = model
        
        # Grad-CAM model oluştur
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
        
        if grads is None:
            raise ValueError("Gradient hesaplanamadı")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination of activation maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU ve normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)
        
        if heatmap_max < 1e-7:
            raise ValueError("Heatmap hesaplanamadı")
        
        heatmap = heatmap / heatmap_max
        heatmap = heatmap.numpy()
        
        # Resize to original image size
        original_height = img_array.shape[1]
        original_width = img_array.shape[2]
        
        from PIL import Image as PILImage
        heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8), mode='L')
        heatmap_pil = heatmap_pil.resize((original_width, original_height), PILImage.Resampling.LANCZOS)
        heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0
        
        return heatmap, img_array[0]
    except Exception as e:
        print(f"[GRAD-CAM] Hesaplama hatası: {e}")
        import traceback
        traceback.print_exc()
        raise

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Heatmap'i görüntü üzerine yerleştir"""
    # Heatmap normalize
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    
    # Colormap uygula (basit RGB)
    heatmap_rgb_normalized = np.stack([heatmap_normalized] * 3, axis=-1)
    
    # Image normalize
    if img.max() > 1.0:
        img_normalized = img.astype(np.float32) / 255.0
    else:
        img_normalized = img.astype(np.float32)
    
    # Resize heatmap to match image dimensions
    if heatmap_rgb_normalized.shape[:2] != img_normalized.shape[:2]:
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
        "model": "EfficientNetB3 - Skin Disease Detection (5 Classes)",
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
    """Predict skin disease from uploaded image"""
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
        processed_image = preprocess_image_efficientnet(image)
        
        # Predict
        # SavedModel veya Keras model prediction (bone_disease_api.py ile aynı)
        if model_type == 'savedmodel':
            # SavedModel - callable veya signature function
            input_tensor = tf.constant(processed_image, dtype=tf.float32)
            
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
        else:
            # Keras model
            predictions = model.predict(processed_image, verbose=0)
        
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
                # Original image'ı sakla
                file_stream.seek(0)
                original_image = Image.open(file_stream).convert('RGB')
                original_array = np.array(original_image)
                
                # Grad-CAM hesapla (sadece Keras model için)
                if model_type == 'savedmodel':
                    result["gradcam_error"] = "Grad-CAM şu an sadece Keras model formatı (.keras veya .h5) için destekleniyor. SavedModel formatı için henüz desteklenmiyor."
                else:
                    print(f"[GRAD-CAM] Grad-CAM hesaplanıyor...")
                    heatmap, _ = compute_gradcam(model, processed_image, pred_index=int(top_idx))
                
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
    print(f"\n[SERVER] Calisiyor: http://localhost:5003")
    print(f"[API] Durum: http://localhost:5003/")
    print(f"[API] Tahmin: POST http://localhost:5003/predict")
    print(f"[API] Siniflar: http://localhost:5003/classes")
    print("\n" + "="*70)
    print("\nFrontend'den kullanmak icin analyze.html dosyasini acin")
    print("Not: CORS etkin, localhost'tan gelen istekler kabul edilir")
    print("\n" + "="*70 + "\n")
    
    app.run(host='127.0.0.1', port=5003, debug=True)

