"""
Ocular Disease Detection - CPU-Optimized Training
- Smaller image size (128px) for faster CPU processing
- Fewer epochs with aggressive early stopping
- Lighter augmentation
- MobileNetV2 only for fast baseline, then EfficientNetB0 + ResNet50
"""

import os, gc, json, time, math, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import config

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ── CPU-optimized config ──────────────────────────────────────────────────────
IMG_SIZE    = 128      # Reduced from 224 — 3x faster on CPU
BATCH_SIZE  = 32       # Larger batch = fewer steps per epoch
EPOCHS_P1   = 8        # Reduced from 15
EPOCHS_P2   = 12       # Reduced from 25
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
AUTOTUNE    = tf.data.AUTOTUNE

VALID_CLASSES = sorted(config.CLASS_LABEL_MAP.keys())
NUM_CLASSES   = len(VALID_CLASSES)
CLASS_TO_IDX  = {cls: i for i, cls in enumerate(VALID_CLASSES)}

# ── Load image paths ──────────────────────────────────────────────────────────
all_paths, all_labels = [], []
for cls in VALID_CLASSES:
    cls_dir = os.path.join(config.DATASET_PATH, cls)
    if not os.path.exists(cls_dir):
        print("WARNING: missing:", cls_dir)
        continue
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_paths.append(os.path.join(cls_dir, fname))
            all_labels.append(CLASS_TO_IDX[cls])

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)

print("\nTotal images:", len(all_paths))
for cls, idx in CLASS_TO_IDX.items():
    print("  %-20s : %d" % (cls, np.sum(all_labels == idx)))

# Save class indices
readable = {config.CLASS_LABEL_MAP[cls]: idx for cls, idx in CLASS_TO_IDX.items()}
with open(config.CLASS_INDICES_PATH, "w") as f:
    json.dump(readable, f, indent=2)
print("\nSaved class_indices.json")

# ── Stratified split ──────────────────────────────────────────────────────────
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.20, random_state=config.RANDOM_SEED, stratify=all_labels)
print("Train:", len(train_paths), "| Val:", len(val_paths))

# ── Class weights ─────────────────────────────────────────────────────────────
cw_arr        = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(cw_arr))
dr_idx  = CLASS_TO_IDX.get("diabetes", -1)
ht_idx  = CLASS_TO_IDX.get("hypertension", -1)
if dr_idx  >= 0: class_weights[dr_idx]  *= 2.0
if ht_idx  >= 0: class_weights[ht_idx]  *= 1.5
print("Class weights:", {k: round(v, 3) for k, v in class_weights.items()})

steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)

# ── Lightweight augmentation (CPU-friendly) ───────────────────────────────────
def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 1.0)

def parse_train(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = augment(img)
    return img, tf.one_hot(label, NUM_CLASSES)

def parse_val(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, NUM_CLASSES)

def make_dataset(paths, labels, train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(parse_train if train else parse_val, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, train=True)
val_ds   = make_dataset(val_paths,   val_labels,   train=False)

# ── Classification head ───────────────────────────────────────────────────────
def build_head(x, prefix):
    x = layers.Dense(256, name=prefix + "_d1")(x)
    x = layers.BatchNormalization(name=prefix + "_bn1")(x)
    x = layers.Activation("relu", name=prefix + "_act1")(x)
    x = layers.Dropout(0.4, name=prefix + "_drop1")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name=prefix + "_out")(x)

# ── Models (B0 instead of B3 — much faster on CPU) ───────────────────────────
def build_efficientnet(trainable=False):
    inp      = layers.Input(shape=INPUT_SHAPE, name="eff_in")
    x        = layers.Rescaling(255.0)(inp)
    backbone = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = trainable
    x = layers.GlobalAveragePooling2D()(backbone.output)
    return Model(inputs=inp, outputs=build_head(x, "eff"), name="EfficientNetB0")

def build_mobilenet(trainable=False):
    inp      = layers.Input(shape=INPUT_SHAPE, name="mob_in")
    backbone = MobileNetV2(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    backbone.trainable = trainable
    x = layers.GlobalAveragePooling2D()(backbone(inp, training=False))
    return Model(inputs=inp, outputs=build_head(x, "mob"), name="MobileNetV2")

def _resnet_preprocess(x):
    return keras.applications.resnet50.preprocess_input(x * 255.0)

def build_resnet(trainable=False):
    inp      = layers.Input(shape=INPUT_SHAPE, name="res_in")
    x        = layers.Lambda(_resnet_preprocess, name="res_preprocess")(inp)
    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = trainable
    x = layers.GlobalAveragePooling2D()(backbone.output)
    return Model(inputs=inp, outputs=build_head(x, "res"), name="ResNet50")

# ── Loss & compile ────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / NUM_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn

def cosine_lr(initial_lr, epochs):
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.01)

def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=focal_loss(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def get_callbacks(save_path):
    return [
        ModelCheckpoint(save_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

# ── Two-phase training ────────────────────────────────────────────────────────
def train_two_phase(builder_fn, model_name, save_path):
    print("\n" + "="*55)
    print("  Training: " + model_name)
    print("="*55)

    print("  PHASE 1 - Frozen | %d epochs" % EPOCHS_P1)
    model = compile_model(builder_fn(trainable=False), cosine_lr(3e-3, EPOCHS_P1))
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_P1, class_weight=class_weights,
                   callbacks=get_callbacks(save_path), verbose=1)
    best1 = max(h1.history["val_accuracy"])
    print("  Phase 1: %.1f min | best val_acc=%.4f" % ((time.time()-t0)/60, best1))

    print("  PHASE 2 - Fine-tune | %d epochs" % EPOCHS_P2)
    for layer in model.layers:
        if hasattr(layer, "layers"):
            n = len(layer.layers)
            cutoff = int(n * 0.70)   # unfreeze only top 30%
            for i, sub in enumerate(layer.layers):
                sub.trainable = False if isinstance(sub, layers.BatchNormalization) else (i >= cutoff)

    model = compile_model(model, cosine_lr(5e-5, EPOCHS_P2))
    t0 = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_P2, class_weight=class_weights,
                   callbacks=get_callbacks(save_path), verbose=1)
    best2 = max(h2.history["val_accuracy"])
    print("  Phase 2: %.1f min | best val_acc=%.4f" % ((time.time()-t0)/60, best2))

    model.save(save_path)
    print("  Saved ->", save_path)
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    return max(best1, best2)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    results = {}
    results["mobilenet"]    = train_two_phase(build_mobilenet,    "mobilenet",    config.MODEL_PATHS["mobilenet"])
    results["efficientnet"] = train_two_phase(build_efficientnet, "efficientnet", config.MODEL_PATHS["efficientnet"])
    results["resnet"]       = train_two_phase(build_resnet,       "resnet",       config.MODEL_PATHS["resnet"])

    print("\n" + "="*55)
    print("  ALL MODELS DONE in %.1f min" % ((time.time()-t_start)/60))
    print("="*55)
    for name, acc in results.items():
        print("  %-15s best val_acc = %.4f" % (name, acc))

    # Simple ensemble evaluation (no TTA to save time)
    print("\n  ENSEMBLE EVALUATION")
    idx2cls   = {v: k for k, v in CLASS_TO_IDX.items()}
    cls_names = [config.CLASS_LABEL_MAP[idx2cls[i]] for i in range(NUM_CLASSES)]
    all_probs = []
    true_labels = np.argmax(
        np.concatenate([y.numpy() for _, y in val_ds], axis=0), axis=1)

    for name in ["mobilenet", "efficientnet", "resnet"]:
        path = config.MODEL_PATHS[name]
        if not os.path.exists(path):
            continue
        m = keras.models.load_model(path, compile=False)
        p = m.predict(val_ds, verbose=0)
        all_probs.append(p)
        print("  %-15s Acc=%.4f" % (name, accuracy_score(true_labels, np.argmax(p, axis=1))))
        del m; gc.collect()

    if len(all_probs) >= 2:
        ensemble_preds = np.argmax(np.mean(all_probs, axis=0), axis=1)
        print("\n  ENSEMBLE Acc=%.4f" % accuracy_score(true_labels, ensemble_preds))
        print("\n" + classification_report(true_labels, ensemble_preds, target_names=cls_names))

    print("\nRun the app: streamlit run app.py")
