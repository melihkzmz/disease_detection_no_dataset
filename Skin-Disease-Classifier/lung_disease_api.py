#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKCIGER HASTALIKLARI TESPIT API
Flask API for Lung Disease Detection
"""

import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

# UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Flask app
app = Flask(__name__)

# Model ve parametreler
MODEL_PATH = "models/lung_disease_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['COVID-19', 'Non-COVID (Pnomoni)', 'Normal']

# Model yukle
print("Model yukleniyor...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model yuklendi! {len(CLASS_NAMES)} sinif - Accuracy: %85.14")
except Exception as e:
    print(f"HATA: Model yuklenemedi: {e}")
    model = None

# ============================================================================
# YARDIMCI FONKSIYONLAR
# ============================================================================

def preprocess_image(image):
    """Goruntu on isleme"""
    # RGB'ye cevir
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Boyutlandir
    image = image.resize(IMG_SIZE)
    
    # Array'e cevir ve normalize et
    img_array = np.array(image) / 255.0
    
    # Batch dimension ekle
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_lung_disease(image):
    """Akciger hastaligi tahmini"""
    if model is None:
        return {"error": "Model yuklu degil"}
    
    try:
        # On isleme
        processed_image = preprocess_image(image)
        
        # Tahmin
        predictions = model.predict(processed_image, verbose=0)
        
        # Sonuclari hazirla
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            confidence = float(predictions[0][i])
            results.append({
                "class": class_name,
                "confidence": confidence,
                "percentage": f"{confidence * 100:.2f}%"
            })
        
        # Confidence'a gore sirala
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "success": True,
            "prediction": results[0]["class"],
            "confidence": results[0]["percentage"],
            "all_predictions": results
        }
    
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Ana sayfa"""
    return jsonify({
        "status": "OK",
        "message": "Akciger Hastaliklari Tespit API",
        "version": "1.0",
        "model": "MobileNetV2",
        "accuracy": "85.14%",
        "classes": CLASS_NAMES,
        "endpoints": {
            "GET /": "API durumu",
            "POST /predict": "Goruntu tahmini (multipart/form-data, field: 'image')",
            "GET /web": "Web arayuzu"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin endpoint'i"""
    if 'image' not in request.files:
        return jsonify({"error": "Goruntu bulunamadi. 'image' field'i gerekli."}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "Dosya secilmedi"}), 400
    
    try:
        # Goruntu yukle
        image = Image.open(io.BytesIO(file.read()))
        
        # Tahmin yap
        result = predict_lung_disease(image)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Tahmin hatasi: {str(e)}"}), 500

@app.route('/web')
def web():
    """Web arayuzu"""
    html = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Akciger Hastaliklari Tespit</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        
        .upload-area.dragover {
            background: #f0f2ff;
            border-color: #764ba2;
            transform: scale(1.02);
        }
        
        #file-input {
            display: none;
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .upload-text {
            color: #667eea;
            font-size: 16px;
            font-weight: 600;
        }
        
        .upload-hint {
            color: #999;
            font-size: 12px;
            margin-top: 10px;
        }
        
        #preview-container {
            margin: 20px 0;
            text-align: center;
            display: none;
        }
        
        #preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
            margin-top: 10px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        #result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .result-success {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        
        .result-error {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .prediction-main {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .prediction-class {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .prediction-confidence {
            font-size: 18px;
            color: #666;
        }
        
        .all-predictions {
            margin-top: 15px;
        }
        
        .prediction-item {
            background: white;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .prediction-label {
            font-weight: 600;
            color: #333;
        }
        
        .prediction-value {
            color: #667eea;
            font-weight: 600;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .info-title {
            font-weight: 600;
            color: #2196F3;
            margin-bottom: 8px;
        }
        
        .class-list {
            list-style: none;
            padding-left: 0;
        }
        
        .class-list li {
            padding: 5px 0;
            color: #555;
        }
        
        .class-list li:before {
            content: "• ";
            color: #2196F3;
            font-weight: bold;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🫁 Akciger Hastaliklari Tespit</h1>
        <p class="subtitle">X-Ray goruntusu yukleyin ve analiz edin</p>
        
        <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
            <div class="upload-icon">📤</div>
            <div class="upload-text">Goruntu secmek icin tiklayin</div>
            <div class="upload-hint">veya goruntuyu buraya surukleyin</div>
            <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
        </div>
        
        <div id="preview-container">
            <img id="preview-image" alt="Preview">
        </div>
        
        <button id="predict-btn" onclick="predictDisease()" disabled>Analiz Et</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analiz ediliyor...</p>
        </div>
        
        <div id="result"></div>
        
        <div class="info-box">
            <div class="info-title">Model Bilgileri</div>
            <ul class="class-list">
                <li>COVID-19: Koronavirus</li>
                <li>Non-COVID: Pnomoni</li>
                <li>Normal: Saglikli Akciger</li>
            </ul>
            <p style="margin-top: 10px; color: #666; font-size: 13px;">
                Model Dogrulugu: <strong>85.14%</strong> | MobileNetV2
            </p>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        // Drag & drop
        const uploadArea = document.getElementById('upload-area');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            }, false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }, false);
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Lutfen bir goruntu dosyasi secin!');
                return;
            }
            
            selectedFile = file;
            
            // Preview
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('preview-image').src = e.target.result;
                document.getElementById('preview-container').style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Enable button
            document.getElementById('predict-btn').disabled = false;
            document.getElementById('result').style.display = 'none';
        }
        
        async function predictDisease() {
            if (!selectedFile) {
                alert('Lutfen bir goruntu secin!');
                return;
            }
            
            // UI update
            document.getElementById('predict-btn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // Form data
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data);
                }
            } catch (error) {
                showError('Baglanti hatasi: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predict-btn').disabled = false;
            }
        }
        
        function showResult(data) {
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result-success';
            
            let html = '<div class="result-title">Analiz Sonucu</div>';
            
            // Ana tahmin
            html += '<div class="prediction-main">';
            html += `<div class="prediction-class">${data.prediction}</div>`;
            html += `<div class="prediction-confidence">Guven: ${data.confidence}</div>`;
            html += '</div>';
            
            // Tum tahminler
            html += '<div class="all-predictions">';
            html += '<div style="font-weight: 600; margin-bottom: 10px; text-align: center;">Detayli Sonuclar:</div>';
            data.all_predictions.forEach(pred => {
                html += '<div class="prediction-item">';
                html += `<span class="prediction-label">${pred.class}</span>`;
                html += `<span class="prediction-value">${pred.percentage}</span>`;
                html += '</div>';
            });
            html += '</div>';
            
            resultDiv.innerHTML = html;
            resultDiv.style.display = 'block';
        }
        
        function showError(message) {
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result-error';
            resultDiv.innerHTML = `
                <div class="result-title">Hata</div>
                <p style="text-align: center; color: #721c24;">${message}</p>
            `;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
    """
    return render_template_string(html)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" AKCIGER HASTALIKLARI TESPIT API")
    print(" Lung Disease Detection API")
    print("=" * 70)
    print(f"Siniflar:")
    for i, class_name in enumerate(CLASS_NAMES, 1):
        print(f"  {i}. {class_name}")
    print(f"\nAPI calistiriliyor...")
    print(f"Adres: http://localhost:5000")
    print(f"Endpoint'ler:")
    print(f"  GET  / - API durumu")
    print(f"  POST /predict - Goruntu tahmini")
    print(f"  GET  /web - Web arayuzu")
    print(f"\nWeb arayuzu: http://localhost:5000/web")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=False)

