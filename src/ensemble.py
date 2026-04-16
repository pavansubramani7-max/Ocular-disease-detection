import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os


DEFAULT_WEIGHTS = {
    "efficientnetb3": 0.25,
    "resnet50"      : 0.25,
    "densenet121"   : 0.20,
    "mobilenetv2"   : 0.15,
    "vgg16"         : 0.15,
}


class EnsemblePredictor:
    def __init__(self, class_indices_path, model_dir="saved_models",
                 efficientnet_path=None, mobilenet_path=None, resnet_path=None,
                 weights=None):

        self.class_indices = {}
        self.models        = {}
        self.weights       = weights or DEFAULT_WEIGHTS

        self._load_all_models(model_dir)
        self._load_class_indices(class_indices_path)

        # Legacy attributes for gradcam in app.py
        self.efficientnet = self.models.get("efficientnetb3")
        self.mobilenet    = self.models.get("mobilenetv2")
        self.resnet       = self.models.get("resnet50")

    def _load_all_models(self, model_dir):
        # Try .keras first (new format), then .h5 (legacy)
        model_names = [
            "efficientnetb3",
            "resnet50",
            "densenet121",
            "mobilenetv2",
            "vgg16",
        ]

        for name in model_names:
            loaded = False
            for ext in [".keras", ".h5"]:
                path = os.path.join(model_dir, name + "_model" + ext)
                if os.path.exists(path):
                    try:
                        self.models[name] = keras.models.load_model(
                            path, compile=False)
                        print(f"Loaded {name} from {path}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Warning: could not load {name}: {e}")
            if not loaded:
                print(f"Warning: {name} not found in {model_dir}")

        if not self.models:
            raise RuntimeError(
                "No model files found in saved_models/. "
                "Download .keras files from Kaggle output.")

    def _load_class_indices(self, path):
        default_indices = {
            "Age-related Macular Degeneration": 0,
            "Cataracts"                       : 1,
            "Diabetic Retinopathy"            : 2,
            "Glaucoma"                        : 3,
            "Hypertension"                    : 4,
            "Myopia"                          : 5,
            "Normal"                          : 6,
        }
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.class_indices = json.load(f)
            except Exception:
                self.class_indices = default_indices
        else:
            self.class_indices = default_indices

    def predict(self, image, return_proba=True):
        img   = np.expand_dims(image, axis=0)
        n     = len(self.class_indices) or 7
        acc   = np.zeros(n, dtype=np.float64)
        w_sum = 0.0

        for name, model in self.models.items():
            w = self.weights.get(name, 0.20)
            try:
                pred = model.predict(img, verbose=0)[0]
                if len(pred) == n:
                    acc   += pred * w
                    w_sum += w
            except Exception as e:
                print(f"Warning: prediction failed for {name}: {e}")

        if w_sum == 0:
            raise RuntimeError("All model predictions failed.")

        probs       = acc / w_sum
        probs       = probs / probs.sum()
        pred_idx    = int(np.argmax(probs))
        class_names = list(self.class_indices.keys())
        pred_class  = class_names[pred_idx]
        pred_dict   = {name: float(probs[i]) for i, name in enumerate(class_names)}
        return pred_dict, pred_class

    def predict_batch(self, images):
        return [self.predict(img) for img in images]

    @property
    def loaded_model_names(self):
        return list(self.models.keys())
