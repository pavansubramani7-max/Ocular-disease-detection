# 👁️ Ocular Disease Detection System

AI-powered detection of 7 ocular diseases using a 5-model deep learning ensemble with Grad-CAM visualization.

---

## 🎯 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| EfficientNetB3 | 74.50% | 0.9483 |
| ResNet50 | 78.33% | 0.9639 |
| DenseNet121 | 79.33% | 0.9634 |
| MobileNetV2 | 75.79% | 0.9611 |
| VGG16 | 81.58% | 0.9693 |
| **Ensemble** | **80.96%** | **0.9694** |

> Trained on 10,449 retinal fundus images across 7 disease classes.
> Target accuracy (92.75%) achievable with full dataset on Kaggle GPU.

---

## 🔬 Diseases Detected
- 👴 Age-related Macular Degeneration
- 🌫️ Cataracts
- 🩸 Diabetic Retinopathy
- 👁️ Glaucoma
- 💉 Hypertension
- 🔍 Myopia
- ✅ Normal

---

## 🚀 Setup & Run

### 1. Install Python 3.11
Download from https://python.org/downloads

### 2. Install Dependencies
```bash
py -3.11 -m pip install streamlit tensorflow opencv-python pillow numpy scikit-learn matplotlib seaborn
```

### 3. Add Model Files
Download trained `.keras` files from Kaggle output and place in `saved_models/`:
- `efficientnetb3_model.keras`
- `resnet50_model.keras`
- `densenet121_model.keras`
- `mobilenetv2_model.keras`
- `vgg16_model.keras`
- `class_indices.json`

### 4. Run the App
```bash
py -3.11 -m streamlit run app.py
```
App opens at http://localhost:8501

---

## 🏋️ GPU Training on Kaggle (to get 90%+ accuracy)

1. Go to [kaggle.com](https://kaggle.com) → New Notebook
2. Add datasets:
   - `manan1717/ocular-disease-dataset`
   - `nurmukhammed7/ocular-diseases`
   - `alaaelmor/ocular-disease`
3. Settings → Accelerator → **GPU T4 x2**
4. Settings → Internet → **ON**
5. Paste `kaggle_train.py` into a cell
6. Click **Save & Run All**
7. Download `.keras` files from Output tab → place in `saved_models/`

---

## 📁 Project Structure
```
ocular-disease-detection/
├── app.py                  # Main Streamlit app
├── kaggle_train.py         # GPU training script for Kaggle
├── config.py               # Paths and settings
├── evaluate.py             # Generate evaluation metrics
├── train.py                # Local training script
├── requirements.txt        # Dependencies
├── src/
│   ├── ensemble.py         # Ensemble predictor (5 models)
│   ├── gradcam.py          # Grad-CAM visualization
│   └── models.py           # Model architectures
└── saved_models/
    └── class_indices.json  # Class label mapping
```

---

## ✨ Features
- 🤖 5-model ensemble prediction
- 🔥 Grad-CAM heatmaps for all 5 models
- 👤 Patient name, age and timestamp
- 📊 Probability bars for all 7 classes
- 📈 Evaluation page with confusion matrix + ROC curves
- 💊 Disease info: symptoms, treatment, severity

---

## 🛠️ Tech Stack
- TensorFlow / Keras
- Streamlit
- OpenCV
- Python 3.11

---

## 👨‍💻 Author
**Pavan Subramani**
GitHub: [@pavansubramani7-max](https://github.com/pavansubramani7-max)
