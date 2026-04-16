# Ocular Disease Detection System

AI-powered detection of 7 ocular diseases using deep learning ensemble.

## Diseases Detected
- Age-related Macular Degeneration
- Cataracts
- Diabetic Retinopathy
- Glaucoma
- Hypertension
- Myopia
- Normal

## Setup Instructions

### 1. Install Python 3.11
Download from https://python.org/downloads

### 2. Create Virtual Environment
```bash
python -m venv ocular_venv
call ocular_venv\Scripts\activate.bat   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```
App opens at http://localhost:8501

---

## GPU Training on Kaggle (to get 92%+ accuracy)

1. Zip your dataset folder (ageDegeneration/cataract/diabetes/glaucoma/hypertension/myopia/normal)
2. Upload to Kaggle Datasets
3. Create new Kaggle Notebook → Settings → GPU T4 x2
4. Add your dataset
5. Paste `kaggle_train.py` into a cell → Run All
6. Download output .h5 files → place in `saved_models/`

## Models
- EfficientNetB3 — Target: 91.19%
- ResNet50       — Target: 91.95%
- DenseNet121    — Target: 90.80%
- MobileNetV3    — Target: 91.57%
- VGG16          — Target: 84.29%
- Ensemble       — Target: 92.75%

## Project Structure
```
ocular-disease-detection/
├── app.py                  # Main Streamlit app
├── kaggle_train.py         # GPU training script for Kaggle
├── config.py               # Paths and settings
├── requirements.txt        # Dependencies
├── src/
│   └── ensemble.py         # Ensemble predictor
└── saved_models/
    ├── class_indices.json
    ├── efficientnet_model.h5
    ├── mobilenet_model.h5
    └── resnet_model.h5
```
