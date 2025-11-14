#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mendeley Eye Disease Detection - Flask API
Serves trained EfficientNetB3 model for eye disease classification
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import io

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

app = Flask(__name__)

# Model configuration
MODEL_PATH = 'models/eye_disease_model.keras'
CLASS_NAMES = [
    'Diabetic_Retinopathy',
    'Disc_Edema',
    'Glaucoma',
    'Macular_Scar',
    'Myopia',
    'Normal',
    'Pterygium',
    'Retinal_Detachment',
    'Retinitis_Pigmentosa'
]

CLASS_DESCRIPTIONS = {
    'Diabetic_Retinopathy': 'Diabetic Retinopathy - damage to blood vessels in retina due to diabetes',
    'Disc_Edema': 'Disc Edema - swelling of optic nerve',
    'Glaucoma': 'Glaucoma - damage to optic nerve, often from high eye pressure',
    'Macular_Scar': 'Macular Scar - scarring in the central part of retina',
    'Myopia': 'Myopia - nearsightedness',
    'Normal': 'Normal - healthy eye',
    'Pterygium': 'Pterygium - growth of tissue on white part of eye',
    'Retinal_Detachment': 'Retinal Detachment - retina pulls away from back of eye',
    'Retinitis_Pigmentosa': 'Retinitis Pigmentosa - genetic disorder causing retina breakdown'
}

IMG_SIZE = (224, 224)

print("\n" + "="*70)
print("MENDELEY EYE DISEASE DETECTION API")
print("="*70)

# Load model
model = None
model_accuracy = None
model_top3_accuracy = None

try:
    print(f"\n[LOADING] Model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print(f"[SUCCESS] Model loaded!")
    print(f"[INFO] Classes: {len(CLASS_NAMES)}")
    
    # Try to get test accuracy if test data available
    TEST_DIR = 'datasets/Eye_Mendeley/test'
    if os.path.exists(TEST_DIR):
        try:
            print(f"[EVAL] Evaluating model on test set...")
            test_ds = tf.keras.utils.image_dataset_from_directory(
                TEST_DIR,
                labels='inferred',
                label_mode='categorical',
                image_size=IMG_SIZE,
                shuffle=False,
                batch_size=32
            )
            loss, accuracy, top_3_accuracy = model.evaluate(test_ds, verbose=0)
            model_accuracy = f"{accuracy*100:.2f}%"
            model_top3_accuracy = f"{top_3_accuracy*100:.2f}%"
            print(f"[METRICS] Accuracy: {model_accuracy}, Top-3: {model_top3_accuracy}")
        except Exception as e:
            print(f"[WARNING] Could not evaluate model: {e}")
            model_accuracy = "N/A"
            model_top3_accuracy = "N/A"
    else:
        model_accuracy = "N/A (test data not found)"
        model_top3_accuracy = "N/A (test data not found)"
        
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print(f"[INFO] Please ensure the model file exists at: {MODEL_PATH}")
    print(f"[INFO] Train the model first using: python train_mendeley_eye.py")
    sys.exit(1)

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize
    image = image.resize(IMG_SIZE)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def status():
    """API status endpoint"""
    return jsonify({
        "status": "API is running",
        "model": "EfficientNetB3 (Mendeley Eye Disease Dataset)",
        "version": "1.0",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "test_accuracy": model_accuracy,
        "top_3_accuracy": model_top3_accuracy,
        "image_size": IMG_SIZE,
        "endpoints": {
            "GET /": "API status",
            "POST /predict": "Image prediction (multipart/form-data, field: 'image')",
            "GET /classes": "List all disease classes",
            "GET /web": "Web interface"
        }
    })

@app.route('/classes')
def list_classes():
    """List all disease classes with descriptions"""
    classes_info = []
    for cls in CLASS_NAMES:
        classes_info.append({
            "name": cls,
            "description": CLASS_DESCRIPTIONS.get(cls, "No description available")
        })
    return jsonify({
        "total_classes": len(CLASS_NAMES),
        "classes": classes_info
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict eye disease from uploaded image"""
    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Predict
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
                "description": CLASS_DESCRIPTIONS.get(CLASS_NAMES[i], ""),
                "confidence": float(predictions[0][i]),
                "confidence_percent": f"{predictions[0][i]*100:.2f}%"
            })
        
        # All predictions
        all_predictions = []
        for i, cls in enumerate(CLASS_NAMES):
            all_predictions.append({
                "class": cls,
                "confidence": float(predictions[0][i]),
                "confidence_percent": f"{predictions[0][i]*100:.2f}%"
            })
        
        return jsonify({
            "success": True,
            "prediction": {
                "class": top_class,
                "description": CLASS_DESCRIPTIONS.get(top_class, ""),
                "confidence": top_confidence,
                "confidence_percent": f"{top_confidence*100:.2f}%"
            },
            "top_3_predictions": top_3_results,
            "all_predictions": all_predictions,
            "note": "This is for research purposes only. Not for clinical diagnosis."
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/web')
def web_interface():
    """Web interface for testing"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eye Disease Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 50px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .stat-box {
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                background: #f8f9fa;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #e9ecef;
                border-color: #764ba2;
            }
            input[type="file"] {
                display: none;
            }
            .upload-btn {
                background: #667eea;
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s;
            }
            .upload-btn:hover {
                background: #764ba2;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            #preview {
                max-width: 400px;
                max-height: 400px;
                margin: 20px auto;
                display: none;
                border-radius: 10px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }
            .prediction-main {
                font-size: 28px;
                color: #667eea;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .confidence {
                font-size: 20px;
                color: #28a745;
                margin-bottom: 15px;
            }
            .description {
                color: #666;
                margin-bottom: 20px;
                font-style: italic;
            }
            .top3 {
                margin-top: 20px;
            }
            .top3-item {
                padding: 15px;
                margin: 10px 0;
                background: white;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .top3-item:first-child {
                border-left: 4px solid #28a745;
            }
            .top3-item:nth-child(2) {
                border-left: 4px solid #ffc107;
            }
            .top3-item:nth-child(3) {
                border-left: 4px solid #17a2b8;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .warning {
                background: #fff3cd;
                border: 1px solid #ffc107;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                color: #856404;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👁️ Eye Disease Detection</h1>
            <div class="subtitle">
                Mendeley Dataset • EfficientNetB3 • 9 Disease Types
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">""" + str(len(CLASS_NAMES)) + """</div>
                    <div class="stat-label">Disease Classes</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + (model_accuracy if model_accuracy else "N/A") + """</div>
                    <div class="stat-label">Test Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + (model_top3_accuracy if model_top3_accuracy else "N/A") + """</div>
                    <div class="stat-label">Top-3 Accuracy</div>
                </div>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                <h3>📤 Upload Fundus Image</h3>
                <p>Click to select a retinal fundus image</p>
                <input type="file" id="imageInput" accept="image/*">
                <button class="upload-btn">Choose Image</button>
            </div>
            
            <img id="preview" alt="Preview">
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>
            
            <div id="result"></div>
            
            <div class="warning">
                ⚠️ <strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
                It is NOT intended for clinical diagnosis. Always consult a qualified ophthalmologist 
                for proper medical advice.
            </div>
        </div>
        
        <script>
            document.getElementById('imageInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    // Show preview
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview').src = e.target.result;
                        document.getElementById('preview').style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                    
                    // Send to API
                    const formData = new FormData();
                    formData.append('image', file);
                    
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('result').innerHTML = '';
                    
                    fetch('/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('loading').style.display = 'none';
                        
                        if (data.success) {
                            let html = '<div class="result">';
                            html += '<h2>🔍 Prediction Results</h2>';
                            html += '<div class="prediction-main">📊 ' + data.prediction.class + '</div>';
                            html += '<div class="confidence">✅ Confidence: ' + data.prediction.confidence_percent + '</div>';
                            html += '<div class="description">' + data.prediction.description + '</div>';
                            
                            html += '<div class="top3"><h3>Top 3 Predictions:</h3>';
                            data.top_3_predictions.forEach((pred, index) => {
                                html += '<div class="top3-item">';
                                html += '<div>';
                                html += '<strong>' + (index + 1) + '. ' + pred.class + '</strong><br>';
                                html += '<small>' + pred.description + '</small>';
                                html += '</div>';
                                html += '<div><strong>' + pred.confidence_percent + '</strong></div>';
                                html += '</div>';
                            });
                            html += '</div></div>';
                            
                            document.getElementById('result').innerHTML = html;
                        } else {
                            document.getElementById('result').innerHTML = 
                                '<div class="result" style="border-left-color: #dc3545;">' +
                                '<h3 style="color: #dc3545;">❌ Error</h3>' +
                                '<p>' + (data.error || 'Unknown error') + '</p>' +
                                '</div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('result').innerHTML = 
                            '<div class="result" style="border-left-color: #dc3545;">' +
                            '<h3 style="color: #dc3545;">❌ Error</h3>' +
                            '<p>Failed to connect to API: ' + error + '</p>' +
                            '</div>';
                    });
                }
            });
        </script>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING API SERVER")
    print("="*70)
    print(f"\n[SERVER] Running on: http://localhost:5001")
    print(f"[WEB] Web interface: http://localhost:5001/web")
    print(f"[API] Status endpoint: http://localhost:5001/")
    print(f"[API] Prediction endpoint: POST http://localhost:5001/predict")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)

