# 🚀 Google Colab Training Guide - Mendeley Eye Disease Detection

**Model:** EfficientNetB3 + Transfer Learning + Fine-tuning  
**Dataset:** Mendeley Eye Disease Dataset (18,363 images, 9 classes)  
**GPU:** Tesla T4 (Free Tier)  
**Expected Time:** ~2-3 hours

---

## 📋 Step-by-Step Instructions

### **1. Prepare Dataset (Local Computer)**

```bash
# Navigate to project
cd C:\Users\melih\dev\disease_detection\Skin-Disease-Classifier

# Create ZIP of organized dataset
Compress-Archive -Path datasets/Eye_Mendeley -DestinationPath Eye_Mendeley.zip
```

**Result:** `Eye_Mendeley.zip` (~2-3 GB, 18,363 images)

---

### **2. Upload Notebook to Google Colab**

1. Go to: https://colab.research.google.com/
2. **Upload notebook:**
   - Click: `File` → `Upload notebook`
   - Select: `train_eye_colab.ipynb` (I'll create it below)
3. **Enable GPU:**
   - Click: `Runtime` → `Change runtime type`
   - Select: `T4 GPU`
   - Click: `Save`

---

### **3. Upload Dataset to Colab**

**Option A: Direct Upload (10-15 minutes)**
- Run the cell that says "Upload from local"
- Select `Eye_Mendeley.zip`
- Wait for upload to complete

**Option B: Google Drive (Faster if you have time)**
1. Upload `Eye_Mendeley.zip` to Google Drive (overnight)
2. In Colab, mount Drive and copy the file
3. Much faster for subsequent runs

---

### **4. Run All Cells**

Click: `Runtime` → `Run all`

**Training will proceed automatically:**
- ✅ Phase 1: Initial training (~1 hour, 50 epochs)
- ✅ Phase 2: Fine-tuning (~1 hour, 30 epochs)
- ✅ Evaluation & Visualization (~5 minutes)

---

### **5. Monitor Progress**

You'll see:
```
Epoch 1/50
525/525 [==============================] - 120s 228ms/step
  - loss: 1.2345
  - accuracy: 0.6543
  - val_accuracy: 0.7123
  - top_3_accuracy: 0.9012
```

**Expected Accuracy:**
- Phase 1: ~55-65% validation accuracy
- Phase 2: ~65-75% validation accuracy 🎯
- Top-3: ~90-95% (very good for medical!)

---

### **6. Download Trained Model**

After training completes:
```python
# Last cell will download automatically
files.download('eye_disease_model.keras')
files.download('confusion_matrix.png')
files.download('training_history.png')
```

**Save to:** `C:\Users\melih\dev\disease_detection\Skin-Disease-Classifier\models\eye_disease_model.keras`

---

## 📊 Expected Results

### **Baseline (ODIR-5K, MobileNetV2):**
```
Test Accuracy: 38.27%
Top-3 Accuracy: 82.69%
```

### **Mendeley + EfficientNetB3 (GPU):**
```
Test Accuracy: 65-75% 🚀 (+27-37%)
Top-3 Accuracy: 90-95% 🚀 (+7-12%)
```

---

## 🎯 Why This Will Work Better

| Factor | Old (ODIR-5K) | New (Mendeley + GPU) | Improvement |
|--------|---------------|----------------------|-------------|
| **Dataset Size** | 6,392 images | 18,363 images | +187% |
| **Data Quality** | Multi-label mess | Single-label clean | ✅ Much better |
| **Model** | MobileNetV2 (light) | EfficientNetB3 (powerful) | +30% capacity |
| **Hardware** | CPU (slow) | GPU T4 (fast) | 10x faster |
| **Training** | No fine-tuning | 2-phase fine-tuning | +10-15% accuracy |
| **Domain Expertise** | No | Yes (hospital-verified) | ✅ More reliable |

---

## 🔧 Troubleshooting

### **Problem: "No GPU available"**
**Solution:**
- Click: `Runtime` → `Change runtime type`
- Select: `T4 GPU`
- Click: `Save`
- Restart runtime

### **Problem: "Out of memory"**
**Solution:**
- Reduce `BATCH_SIZE` from 32 to 16:
  ```python
  BATCH_SIZE = 16  # Instead of 32
  ```

### **Problem: "Upload too slow"**
**Solution:**
- Use Google Drive option instead
- Or upload overnight

### **Problem: "Session disconnected"**
**Solution:**
- Colab free tier: 12 hours max
- Keep browser tab open
- Use Google Colab Pro ($10/month) for longer sessions

---

## 📱 Mobile Version (Optional - for later)

After training, convert to TensorFlow Lite:
```python
# In Colab, after training
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('eye_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Download
files.download('eye_disease_model.tflite')
```

---

## 🎓 Learning Points

### **Why 2 Phases?**
1. **Phase 1 (Frozen):** Learn disease-specific features quickly
2. **Phase 2 (Fine-tune):** Adapt low-level features (edges, textures) to fundus images

### **Why Class Weights?**
```python
Pterygium: 90 images (0.5%) → Weight: 13.45
Diabetic_Retinopathy: 4,296 images (23.4%) → Weight: 0.28
```
Without weights, model would ignore rare diseases!

### **Why Top-3 Accuracy Matters?**
- Doctor gets 3 suggestions
- Real diagnosis: combine with patient history, symptoms
- 95% chance correct diagnosis in top-3 = very useful!

---

## 🏆 Success Criteria

**Minimum Acceptable:**
- ✅ Test Accuracy > 60%
- ✅ Top-3 Accuracy > 85%
- ✅ No class with 0% recall

**Target (Achievable):**
- 🎯 Test Accuracy: 65-75%
- 🎯 Top-3 Accuracy: 90-95%
- 🎯 All classes > 40% recall

**Excellent (Hopeful):**
- 🌟 Test Accuracy: >75%
- 🌟 Top-3 Accuracy: >95%
- 🌟 All classes > 60% recall

---

## 📞 Need Help?

Common Colab questions:
- GPU quota exhausted? Wait 24h or use Colab Pro
- Runtime keeps disconnecting? Keep browser tab active
- Upload failed? Try smaller batches or Google Drive

---

## ✅ Checklist

Before starting:
- [ ] `Eye_Mendeley.zip` created and ready
- [ ] Google account logged in
- [ ] Colab notebook uploaded
- [ ] GPU runtime selected
- [ ] At least 2-3 hours available (don't close browser!)

After training:
- [ ] Model downloaded (`eye_disease_model.keras`)
- [ ] Plots downloaded (confusion matrix, training history)
- [ ] Model moved to `models/` folder
- [ ] Ready to run Flask API

---

## 🚀 Next: Flask API

After downloading trained model:
```bash
# Copy model to project
cp ~/Downloads/eye_disease_model.keras models/

# Run API
python eye_disease_api.py

# Test
python test_eye_api.py
```

---

**Ready? Let's create the Colab notebook file!** 🎉

