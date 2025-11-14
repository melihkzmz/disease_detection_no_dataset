# 👁️ Eye Disease Detection System

**Model:** EfficientNetB3 + Transfer Learning + Fine-tuning  
**Dataset:** Mendeley Eye Disease Dataset  
**Classes:** 9 eye diseases  
**Training:** 2-phase (Initial + Fine-tuning)

---

## 📊 Dataset Information

### **Source**
[Mendeley Eye Disease Image Dataset](https://data.mendeley.com/datasets/s9bfhswzjb/1)

### **Statistics**
```
Total Images: 18,363
├── Training:   16,788 (91.4%)
├── Validation:    781 (4.3%)
└── Test:          794 (4.3%)

Original Images:  5,234
Augmented Images: 13,129
```

### **Disease Classes**

| # | Class | Training | Val | Test | Total | % |
|---|-------|----------|-----|------|-------|---|
| 1 | Diabetic Retinopathy | 3,843 | 226 | 227 | 4,296 | 23.4% |
| 2 | Glaucoma | 3,569 | 202 | 203 | 3,974 | 21.6% |
| 3 | Normal | 3,062 | 153 | 155 | 3,370 | 18.4% |
| 4 | Myopia | 2,264 | 75 | 75 | 2,414 | 13.1% |
| 5 | Macular Scar | 1,820 | 66 | 68 | 1,954 | 10.6% |
| 6 | Retinitis Pigmentosa | 801 | 20 | 22 | 843 | 4.6% |
| 7 | Disc Edema | 735 | 19 | 20 | 774 | 4.2% |
| 8 | Retinal Detachment | 610 | 18 | 20 | 648 | 3.5% |
| 9 | Pterygium | 84 | 2 | 4 | 90 | 0.5% |

**Note:** Dataset is imbalanced. Class weights are used during training.

---

## 🏗️ Model Architecture

### **Base Model: EfficientNetB3**
```
Input: (224, 224, 3)
├── EfficientNetB3 (ImageNet pretrained)
│   └── 300+ layers, frozen initially
├── GlobalAveragePooling2D
├── BatchNormalization
├── Dropout(0.5)
├── Dense(512, relu, L2 regularization)
├── BatchNormalization
├── Dropout(0.3)
├── Dense(256, relu, L2 regularization)
├── Dropout(0.2)
└── Dense(9, softmax)

Total Parameters: ~12M
Trainable (Phase 1): ~2M
Trainable (Phase 2): ~6M
```

### **Why EfficientNetB3?**
- ✅ Better accuracy than MobileNetV2
- ✅ Efficient architecture (compound scaling)
- ✅ Pre-trained on ImageNet
- ✅ Good balance: speed vs accuracy

---

## 🎯 Training Strategy

### **Phase 1: Initial Training (Base Frozen)**
```python
Epochs: 50
Learning Rate: 0.001
Base Model: Frozen (only top layers train)
Duration: ~1-2 hours (CPU), ~20 min (GPU)

Goal: Learn disease-specific features quickly
```

### **Phase 2: Fine-tuning (Partial Unfreeze)**
```python
Epochs: 30
Learning Rate: 0.0001 (10x lower)
Base Model: Top 50 layers unfrozen
Duration: ~1-1.5 hours (CPU), ~15 min (GPU)

Goal: Adapt low-level features to fundus images
```

### **Data Augmentation**
```python
Training:
- Rotation: ±20°
- Width/Height shift: 20%
- Shear: 15%
- Zoom: 20%
- Horizontal flip
- Brightness: 80-120%

Validation/Test: No augmentation (rescale only)
```

### **Class Weights (for Imbalanced Data)**
```python
Pterygium: 13.45 (rare, 90 samples)
Diabetic_Retinopathy: 0.28 (common, 4,296 samples)
```

---

## 📈 Expected Performance

### **Comparison: ODIR-5K vs Mendeley**

| Metric | ODIR-5K (Old) | Mendeley (New) | Improvement |
|--------|---------------|----------------|-------------|
| **Test Accuracy** | 38.27% | **65-75%** | +27-37% 🚀 |
| **Top-3 Accuracy** | 82.69% | **90-95%** | +7-12% 🚀 |
| **Dataset Size** | 6,392 | 18,363 | +187% |
| **Label Quality** | Multi-label | Single-label | ✅ Cleaner |
| **Domain Expert** | No | Yes | ✅ More reliable |

### **Per-Class Performance (Expected)**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Diabetic Retinopathy | 0.75-0.85 | 0.70-0.80 | 0.72-0.82 |
| Glaucoma | 0.70-0.80 | 0.65-0.75 | 0.67-0.77 |
| Normal | 0.80-0.90 | 0.75-0.85 | 0.77-0.87 |
| Myopia | 0.60-0.70 | 0.55-0.65 | 0.57-0.67 |
| Macular Scar | 0.55-0.65 | 0.50-0.60 | 0.52-0.62 |
| Retinitis Pigmentosa | 0.50-0.60 | 0.45-0.55 | 0.47-0.57 |
| Disc Edema | 0.50-0.60 | 0.45-0.55 | 0.47-0.57 |
| Retinal Detachment | 0.45-0.55 | 0.40-0.50 | 0.42-0.52 |
| Pterygium | 0.40-0.50 | 0.30-0.40 | 0.34-0.44 |

**Note:** Rare diseases (Pterygium, Retinal Detachment) harder to detect due to limited samples.

---

## 🚀 Usage

### **1. Training**

#### **Option A: Local (CPU - 6-8 hours)**
```bash
cd Skin-Disease-Classifier
python train_mendeley_eye.py
```

#### **Option B: Google Colab (GPU - 2-3 hours)** ⭐ Recommended
1. Upload dataset: `Eye_Mendeley.zip`
2. Open Colab notebook
3. Enable GPU (T4)
4. Run all cells
5. Download trained model

See: `COLAB_TRAINING_GUIDE.md`

### **2. Flask API**

```bash
# Start API server
python eye_disease_api.py

# API runs on: http://localhost:5001
```

**Endpoints:**
- `GET /` - API status
- `GET /classes` - List all diseases
- `POST /predict` - Predict from image
- `GET /web` - Web interface

### **3. Test API**

```bash
python test_eye_api.py
```

### **4. Web Interface**

Open browser: http://localhost:5001/web

---

## 💻 API Examples

### **Python**
```python
import requests

url = "http://localhost:5001/predict"
files = {'image': open('fundus_image.jpg', 'rb')}
response = requests.post(url, files=files)
data = response.json()

print(f"Prediction: {data['prediction']['class']}")
print(f"Confidence: {data['prediction']['confidence_percent']}")
```

### **cURL**
```bash
curl -X POST \
  -F "image=@fundus_image.jpg" \
  http://localhost:5001/predict
```

### **JavaScript (Fetch)**
```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('http://localhost:5001/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 📦 Files Structure

```
Skin-Disease-Classifier/
├── datasets/
│   └── Eye_Mendeley/          # Organized dataset
│       ├── train/             # 16,788 images
│       ├── val/               # 781 images
│       └── test/              # 794 images
├── models/
│   ├── eye_disease_model.keras           # Final model
│   ├── eye_disease_initial.keras         # Phase 1 checkpoint
│   ├── eye_disease_finetuned.keras       # Phase 2 checkpoint
│   └── training_history_mendeley_eye.png # Training plots
├── organize_mendeley_eye.py   # Dataset organization script
├── train_mendeley_eye.py      # Training script
├── eye_disease_api.py         # Flask API
├── test_eye_api.py            # API test script
├── test_gpu.py                # GPU detection test
└── EYE_DISEASE_README.md      # This file
```

---

## 🔧 Requirements

```txt
tensorflow>=2.12.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
flask>=3.0.0
numpy>=1.23.0,<1.24.0
pandas>=2.0.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🎓 Medical Context

### **Disease Descriptions**

**1. Diabetic Retinopathy**
- Damage to blood vessels in retina due to diabetes
- Leading cause of blindness in working-age adults
- Early detection crucial for treatment

**2. Glaucoma**
- Optic nerve damage, often from high eye pressure
- "Silent thief of sight" - no early symptoms
- Second leading cause of blindness worldwide

**3. Normal**
- Healthy eye with no detected abnormalities
- Baseline for comparison

**4. Myopia (Nearsightedness)**
- Distant objects appear blurry
- Can lead to more serious conditions if severe
- Increasing prevalence globally

**5. Macular Scar**
- Scarring in central retina (macula)
- Affects central vision
- Can result from various conditions

**6. Retinitis Pigmentosa**
- Genetic disorder causing retina breakdown
- Progressive vision loss (night blindness → tunnel vision)
- Currently no cure, but treatments can slow progression

**7. Disc Edema (Papilledema)**
- Swelling of optic nerve
- Can indicate increased intracranial pressure
- Requires urgent investigation

**8. Retinal Detachment**
- Retina pulls away from back of eye
- Medical emergency requiring immediate surgery
- Can cause permanent vision loss if untreated

**9. Pterygium**
- Growth of tissue on white part of eye
- Often caused by UV exposure
- Can affect vision if grows over cornea

### **Clinical Workflow Integration**

```
Patient Visit → Fundus Photography → AI Screening → Ophthalmologist Review → Diagnosis
                                          ↓
                                    Flag high-risk cases
                                    Reduce screening time
                                    Assist decision-making
```

**AI Role:** Screening tool, NOT diagnostic tool
- Help prioritize urgent cases
- Reduce ophthalmologist workload
- Assist in underserved areas

---

## ⚠️ Important Disclaimers

### **Research Use Only**
- ✅ Educational purposes
- ✅ Research studies
- ✅ Proof-of-concept demonstrations
- ❌ NOT for clinical diagnosis
- ❌ NOT FDA/CE approved
- ❌ NOT a replacement for ophthalmologist

### **Limitations**
1. **Data Bias:** Trained on specific populations
2. **Image Quality:** Requires good fundus photos
3. **Rare Diseases:** Lower accuracy (Pterygium: 0.5% of data)
4. **Edge Cases:** May fail on unusual presentations
5. **No Clinical Context:** Doesn't consider patient history

### **Medical Validation Needed**
For clinical use, system requires:
- ✅ IRB approval
- ✅ Clinical trials
- ✅ FDA/regulatory approval
- ✅ Integration with DICOM/PACS
- ✅ HIPAA compliance
- ✅ Ongoing monitoring & updates

---

## 📊 Evaluation Metrics

### **Primary Metrics**
- **Accuracy:** Overall correctness
- **Top-3 Accuracy:** Correct answer in top 3 predictions

### **Per-Class Metrics**
- **Precision:** Of predicted positives, how many are correct?
- **Recall (Sensitivity):** Of actual positives, how many detected?
- **F1-Score:** Harmonic mean of precision & recall

### **Why Top-3 Matters in Medical AI?**
- Ophthalmologist considers differential diagnosis (multiple possibilities)
- Top-3 gives alternative diagnoses
- 95% Top-3 accuracy = very useful clinical tool

---

## 🔬 Future Improvements

### **Model Enhancements**
1. **Ensemble:** Combine EfficientNetB3 + ResNet50 + DenseNet
2. **Attention Mechanism:** Focus on relevant retina regions
3. **Multi-task Learning:** Predict disease + severity simultaneously
4. **Few-shot Learning:** Better handle rare diseases

### **Data Improvements**
1. **More Data:** Collect additional rare disease samples
2. **External Validation:** Test on different datasets
3. **Multi-center:** Include diverse populations
4. **Longitudinal:** Track disease progression

### **Deployment**
1. **TensorFlow Lite:** Mobile deployment
2. **ONNX:** Cross-platform compatibility
3. **TensorFlow.js:** Browser-based inference
4. **TensorFlow Serving:** Production-grade serving

### **Clinical Features**
1. **Severity Grading:** Mild/Moderate/Severe
2. **Heatmaps:** Visualize AI decision regions
3. **Uncertainty Estimation:** Flag low-confidence cases
4. **Integration:** DICOM support, HL7 FHIR

---

## 📚 References

### **Dataset**
```
Rashid, M. R., Sharmin, S., Khatun, T., Hasan, M. Z., & Uddin, M. S. (2024).
Eye Disease Image Dataset. Mendeley Data, V1.
DOI: 10.17632/s9bfhswzjb.1
```

### **Model Architecture**
```
Tan, M., & Le, Q. (2019).
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
ICML 2019.
```

### **Medical Background**
- [WHO - Blindness and vision impairment](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)
- [NEI - Eye Disease Statistics](https://www.nei.nih.gov/learn-about-eye-health)

---

## 🤝 Contributing

This is a research/educational project. Contributions welcome:
- Better preprocessing techniques
- Model architecture improvements
- Additional disease classes
- Performance optimizations

---

## 📧 Contact & Support

For questions or issues:
1. Check training logs: `training_log.txt`
2. Review this README
3. See `COLAB_TRAINING_GUIDE.md` for GPU training

---

## ✅ Quick Start Checklist

**For Training:**
- [ ] Dataset organized (`datasets/Eye_Mendeley/`)
- [ ] Requirements installed
- [ ] Choose CPU (6-8h) or GPU Colab (2-3h)
- [ ] Run training script
- [ ] Wait for model (`models/eye_disease_model.keras`)

**For API:**
- [ ] Model trained and saved
- [ ] Flask installed
- [ ] Run `python eye_disease_api.py`
- [ ] Open http://localhost:5001/web
- [ ] Upload fundus image
- [ ] View predictions!

---

**Last Updated:** October 30, 2025  
**Version:** 1.0  
**Status:** Training in progress... 🚀

