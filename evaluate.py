"""
Generate validation_results.json for the Evaluation page in app.py
Run: py -3.11 evaluate.py
"""

import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import config

print("Loading models...")
models = {}
model_names = ["efficientnetb3", "resnet50", "densenet121", "mobilenetv2", "vgg16"]
for name in model_names:
    for ext in [".keras", ".h5"]:
        path = os.path.join(config.SAVED_MODELS_DIR, name + "_model" + ext)
        if os.path.exists(path):
            try:
                models[name] = keras.models.load_model(path, compile=False)
                print(f"  Loaded {name}")
                break
            except Exception as e:
                print(f"  Failed {name}: {e}")

if not models:
    print("No models found!")
    exit()

# ── Load class indices ────────────────────────────────────
with open(config.CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())
NUM_CLASSES  = len(class_names)
print(f"\nClasses: {class_names}")

# ── Load images from dataset ──────────────────────────────
DATASET_DIR = r"C:\ocular_combined"
FOLDER_MAP  = {
    "ageDegeneration" : "Age-related Macular Degeneration",
    "cataract"        : "Cataracts",
    "diabetes"        : "Diabetic Retinopathy",
    "glaucoma"        : "Glaucoma",
    "hypertension"    : "Hypertension",
    "myopia"          : "Myopia",
    "normal"          : "Normal",
}

all_paths, all_labels = [], []
for folder, cls_name in FOLDER_MAP.items():
    cls_dir = os.path.join(DATASET_DIR, folder)
    if not os.path.exists(cls_dir):
        print(f"  WARNING: {cls_dir} not found")
        continue
    idx = class_names.index(cls_name)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".jpg",".jpeg",".png")):
            all_paths.append(os.path.join(cls_dir, fname))
            all_labels.append(idx)

print(f"\nTotal images found: {len(all_paths)}")

if len(all_paths) == 0:
    print("No images found! Check DATASET_DIR path.")
    exit()

# ── Use 20% as validation set ─────────────────────────────
_, val_paths, _, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.20,
    random_state=42, stratify=all_labels)
print(f"Validation set: {len(val_paths)} images")

# ── Preprocess images ─────────────────────────────────────
IMG_SIZE = config.IMAGE_SIZE
print(f"\nPreprocessing {len(val_paths)} images at {IMG_SIZE}x{IMG_SIZE}...")

val_images = []
valid_labels = []
for i, (path, label) in enumerate(zip(val_paths, val_labels)):
    try:
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        val_images.append(np.array(img) / 255.0)
        valid_labels.append(label)
    except Exception:
        continue
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(val_paths)}")

val_images = np.array(val_images, dtype=np.float32)
val_labels = np.array(valid_labels)
print(f"Preprocessed: {val_images.shape}")

# ── Predict with ensemble ─────────────────────────────────
print("\nRunning predictions...")
all_probs = []
for name, model in models.items():
    print(f"  Predicting with {name}...")
    preds = model.predict(val_images, batch_size=32, verbose=0)
    all_probs.append(preds)
    acc = accuracy_score(val_labels, np.argmax(preds, axis=1))
    print(f"    {name} accuracy: {acc:.4f}")

ensemble_probs = np.mean(all_probs, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc   = accuracy_score(val_labels, ensemble_preds)
print(f"\nEnsemble accuracy: {ensemble_acc:.4f}")
print("\nClassification Report:")
print(classification_report(val_labels, ensemble_preds, target_names=class_names))

# ── Save results ──────────────────────────────────────────
os.makedirs(config.RESULTS_DIR, exist_ok=True)
results = {
    "y_true"      : val_labels.tolist(),
    "y_pred"      : ensemble_preds.tolist(),
    "y_pred_proba": ensemble_probs.tolist(),
    "class_names" : class_names,
    "accuracy"    : float(ensemble_acc),
    "num_models"  : len(models),
}
out_path = os.path.join(config.RESULTS_DIR, "validation_results.json")
with open(out_path, "w") as f:
    json.dump(results, f)
print(f"\nSaved → {out_path}")
print("Now open the app and go to Evaluation page!")
