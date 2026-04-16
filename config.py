import os

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH     = r"C:\ocular_combined"
IMG_SIZE         = 224
IMAGE_SIZE       = 224
BATCH_SIZE       = 16
NUM_CLASSES      = 7
RANDOM_SEED      = 42
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
EPOCHS           = 20

CLASS_LABEL_MAP = {
    "ageDegeneration": "Age-related Macular Degeneration",
    "cataract":        "Cataracts",
    "diabetes":        "Diabetic Retinopathy",
    "glaucoma":        "Glaucoma",
    "hypertension":    "Hypertension",
    "myopia":          "Myopia",
    "normal":          "Normal",
}
CLASSES = list(CLASS_LABEL_MAP.values())

MODEL_PATHS = {
    "efficientnet" : os.path.join(SAVED_MODELS_DIR, "efficientnet_model.h5"),
    "mobilenet"    : os.path.join(SAVED_MODELS_DIR, "mobilenet_model.h5"),
    "resnet"       : os.path.join(SAVED_MODELS_DIR, "resnet_model.h5"),
}

CLASS_INDICES_PATH = os.path.join(SAVED_MODELS_DIR, "class_indices.json")

for _d in [SAVED_MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)
