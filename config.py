import os

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH     = r"C:\ocular_combined"
IMG_SIZE         = 224
IMAGE_SIZE       = 224
BATCH_SIZE       = 32
NUM_CLASSES      = 7
RANDOM_SEED      = 42
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
EPOCHS           = 20

CLASS_LABEL_MAP = {
    "ageDegeneration": "Age-related Macular Degeneration",
    "cataract"       : "Cataracts",
    "diabetes"       : "Diabetic Retinopathy",
    "glaucoma"       : "Glaucoma",
    "hypertension"   : "Hypertension",
    "myopia"         : "Myopia",
    "normal"         : "Normal",
}
CLASSES = list(CLASS_LABEL_MAP.values())

# New .keras model paths
MODEL_PATHS = {
    "efficientnetb3": os.path.join(SAVED_MODELS_DIR, "efficientnetb3_model.keras"),
    "resnet50"      : os.path.join(SAVED_MODELS_DIR, "resnet50_model.keras"),
    "densenet121"   : os.path.join(SAVED_MODELS_DIR, "densenet121_model.keras"),
    "mobilenetv2"   : os.path.join(SAVED_MODELS_DIR, "mobilenetv2_model.keras"),
    "vgg16"         : os.path.join(SAVED_MODELS_DIR, "vgg16_model.keras"),
}

CLASS_INDICES_PATH = os.path.join(SAVED_MODELS_DIR, "class_indices.json")

for _d in [SAVED_MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)
