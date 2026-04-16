import subprocess
subprocess.run(["pip", "install", "-q", "scikit-learn"], check=True)

import os, gc, json, time, math, warnings, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121, MobileNetV2, VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
warnings.filterwarnings("ignore")

tf.keras.mixed_precision.set_global_policy("mixed_float16")

print("=" * 60)
print("TensorFlow :", tf.__version__)
print("GPUs       :", tf.config.list_physical_devices("GPU"))
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────
OUTPUT_DIR   = "/kaggle/working/saved_models"
COMBINED_DIR = "/kaggle/working/combined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
RANDOM_SEED = 42
EPOCHS_P1   = 15
EPOCHS_P2   = 20

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

CLASS_LABEL_MAP = {
    "ageDegeneration" : "Age-related Macular Degeneration",
    "cataract"        : "Cataracts",
    "diabetes"        : "Diabetic Retinopathy",
    "glaucoma"        : "Glaucoma",
    "hypertension"    : "Hypertension",
    "myopia"          : "Myopia",
    "normal"          : "Normal",
}
VALID_CLASSES = sorted(CLASS_LABEL_MAP.keys())
NUM_CLASSES   = len(VALID_CLASSES)
CLASS_TO_IDX  = {cls: i for i, cls in enumerate(VALID_CLASSES)}

# ── Wipe and recreate combined dir ────────────────────────
if os.path.exists(COMBINED_DIR):
    shutil.rmtree(COMBINED_DIR)
for cls in VALID_CLASSES:
    os.makedirs(os.path.join(COMBINED_DIR, cls), exist_ok=True)

# ── Copy helper ───────────────────────────────────────────
def copy_images(src_dir, target_cls, label):
    dst_dir = os.path.join(COMBINED_DIR, target_cls)
    if not os.path.exists(src_dir):
        print(f"  WARNING: not found: {src_dir}"); return 0
    existing = len(os.listdir(dst_dir))
    count = 0
    for f in os.listdir(src_dir):
        if not f.lower().endswith((".jpg",".jpeg",".png")): continue
        ext = os.path.splitext(f)[1]
        shutil.copy2(os.path.join(src_dir, f),
                     os.path.join(dst_dir, f"{label}_{existing+count}{ext}"))
        count += 1
    print(f"  {label} → {target_cls}: {count}")
    return count

# ── Dataset 1: manan1717 (A,C,D,G,H,M,N folders) ─────────
print("\nLoading manan1717...")
BASE1 = "/kaggle/input/datasets/manan1717/ocular-disease-dataset/preprocessed"
MAP1  = {"A":"ageDegeneration","C":"cataract","D":"diabetes",
         "G":"glaucoma","H":"hypertension","M":"myopia","N":"normal"}
for folder, cls in MAP1.items():
    copy_images(os.path.join(BASE1, folder), cls, f"ds1_{folder}")

# ── Dataset 2: nurmukhammed7 ──────────────────────────────
print("\nLoading nurmukhammed7...")
BASE2 = "/kaggle/input/datasets/nurmukhammed7/ocular-diseases/OCULAR_DISEASES"
MAP2  = {"amd":"ageDegeneration","cataract":"cataract","diabetes":"diabetes",
         "glaucoma":"glaucoma","hypertension":"hypertension","myopia":"myopia","normal":"normal"}
for folder, cls in MAP2.items():
    copy_images(os.path.join(BASE2, folder), cls, f"ds2_{folder}")

# ── Dataset 3: alaaelmor ─────────────────────────────────
print("\nLoading alaaelmor...")
BASE3 = "/kaggle/input/datasets/alaaelmor/ocular-disease/dataset/dataset"
MAP3  = {"ARMD":"ageDegeneration","cataract":"cataract",
         "diabetic_retinopathy":"diabetes","glaucoma":"glaucoma","normal":"normal"}
for folder, cls in MAP3.items():
    copy_images(os.path.join(BASE3, folder), cls, f"ds3_{folder}")

# ── Count combined ────────────────────────────────────────
print("\nCombined counts:")
grand = 0
for cls in sorted(VALID_CLASSES):
    n = len(os.listdir(os.path.join(COMBINED_DIR, cls)))
    grand += n
    print(f"  {cls:20s}: {n}")
print(f"  {'TOTAL':20s}: {grand}")

# ── Save class indices ────────────────────────────────────
readable = {CLASS_LABEL_MAP[cls]: idx for cls, idx in CLASS_TO_IDX.items()}
with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
    json.dump(readable, f, indent=2)
print("\nclass_indices.json saved")

# ── Load all images ───────────────────────────────────────
all_paths, all_labels = [], []
for cls in VALID_CLASSES:
    cls_dir = os.path.join(COMBINED_DIR, cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".jpg",".jpeg",".png")):
            all_paths.append(os.path.join(cls_dir, fname))
            all_labels.append(CLASS_TO_IDX[cls])

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
print(f"\nTotal images: {len(all_paths)}")

# ── Split ─────────────────────────────────────────────────
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.20,
    random_state=RANDOM_SEED, stratify=all_labels)
print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

# ── Class weights ─────────────────────────────────────────
cw_arr        = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(cw_arr))
for cls_name, boost in [("hypertension",2.5),("ageDegeneration",1.5),("myopia",1.5)]:
    idx = CLASS_TO_IDX.get(cls_name, -1)
    if idx >= 0: class_weights[idx] *= boost
print("Class weights:", {k: round(v,3) for k,v in class_weights.items()})

AUTOTUNE    = tf.data.AUTOTUNE
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# ── Data pipeline ─────────────────────────────────────────
def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.05)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    crop_size = tf.random.uniform([], int(IMG_SIZE*0.85), IMG_SIZE, dtype=tf.int32)
    img = tf.image.random_crop(img, [crop_size, crop_size, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.clip_by_value(img, 0.0, 1.0)

def parse_train(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return augment(img), tf.one_hot(label, NUM_CLASSES)

def parse_val(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, NUM_CLASSES)

def make_dataset(paths, labels, train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(parse_train if train else parse_val, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(train_paths, train_labels, train=True)
val_ds   = make_dataset(val_paths,   val_labels,   train=False)

# ── SE block + head ───────────────────────────────────────
def se_block(x, ratio=16):
    ch = x.shape[-1]
    s  = layers.GlobalAveragePooling2D()(x)
    s  = layers.Dense(max(ch // ratio, 8), activation="relu")(s)
    s  = layers.Dense(ch, activation="sigmoid")(s)
    s  = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])

def build_head(x, prefix):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, name=f"{prefix}_d1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}_act1")(x)
    x = layers.Dropout(0.4, name=f"{prefix}_drop1")(x)
    x = layers.Dense(256, name=f"{prefix}_d2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}_act2")(x)
    x = layers.Dropout(0.3, name=f"{prefix}_drop2")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax",
                        dtype="float32", name=f"{prefix}_out")(x)

# ── Model builders ────────────────────────────────────────
def build_efficientnetb3(trainable=False):
    inp = layers.Input(shape=INPUT_SHAPE, name="effb3_in")
    x   = layers.Rescaling(255.0)(inp)
    bb  = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=x)
    bb.trainable = trainable
    return Model(inputs=inp, outputs=build_head(se_block(bb.output), "effb3"), name="EfficientNetB3")

def build_resnet50(trainable=False):
    inp = layers.Input(shape=INPUT_SHAPE, name="res50_in")
    bb  = ResNet50(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    bb.trainable = trainable
    x = bb(keras.applications.resnet50.preprocess_input(inp * 255.0), training=False)
    return Model(inputs=inp, outputs=build_head(se_block(x), "res50"), name="ResNet50")

def build_densenet121(trainable=False):
    inp = layers.Input(shape=INPUT_SHAPE, name="den121_in")
    bb  = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    bb.trainable = trainable
    x = bb(keras.applications.densenet.preprocess_input(inp * 255.0), training=False)
    return Model(inputs=inp, outputs=build_head(se_block(x), "den121"), name="DenseNet121")

def build_mobilenetv2(trainable=False):
    inp = layers.Input(shape=INPUT_SHAPE, name="mob_in")
    bb  = MobileNetV2(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    bb.trainable = trainable
    x = bb(keras.applications.mobilenet_v2.preprocess_input(inp * 255.0), training=False)
    return Model(inputs=inp, outputs=build_head(se_block(x), "mob"), name="MobileNetV2")

def build_vgg16(trainable=False):
    inp = layers.Input(shape=INPUT_SHAPE, name="vgg_in")
    bb  = VGG16(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    bb.trainable = trainable
    x = bb(keras.applications.vgg16.preprocess_input(inp * 255.0), training=False)
    return Model(inputs=inp, outputs=build_head(x, "vgg"), name="VGG16")

# ── Loss ──────────────────────────────────────────────────
def focal_loss(gamma=2.0, smoothing=0.1):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-8, 1.0)
        y_true = tf.cast(y_true, tf.float32) * (1.0 - smoothing) + smoothing / NUM_CLASSES
        ce     = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn

def get_metrics():
    return ["accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")]

# ── Model configs — ONLY 2 NEW MODELS ───────────────────
MODEL_CONFIGS = {
    "densenet121" : (build_densenet121, 3e-3, 5e-5, 0.30),
    "vgg16"       : (build_vgg16,       3e-3, 5e-5, 0.30),
}

def get_callbacks(save_path, best_so_far=None):
    return [
        ModelCheckpoint(save_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1,
                        initial_value_threshold=best_so_far),
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=4, min_lr=1e-7, verbose=1),
    ]

# ── Training ──────────────────────────────────────────────
def train_model(model_name):
    builder, p1_lr, p2_lr, unfreeze_pct = MODEL_CONFIGS[model_name]
    save_path = os.path.join(OUTPUT_DIR, model_name + "_model.h5")

    print("\n" + "=" * 60)
    print(f"  Training: {model_name}")
    print("=" * 60)

    print(f"  PHASE 1 — Frozen | {EPOCHS_P1} epochs | lr={p1_lr}")
    model = builder(trainable=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p1_lr, clipnorm=1.0),
                  loss=focal_loss(), metrics=get_metrics())
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_P1, class_weight=class_weights,
                   callbacks=get_callbacks(save_path, None), verbose=1)
    best1 = max(h1.history["val_accuracy"])
    print(f"  Phase 1: {(time.time()-t0)/60:.1f} min | acc={best1:.4f} | "
          f"auc={max(h1.history['val_auc']):.4f} | "
          f"rec={max(h1.history['val_recall']):.4f} | "
          f"prec={max(h1.history['val_precision']):.4f}")

    print(f"  PHASE 2 — Fine-tune top {int(unfreeze_pct*100)}% | {EPOCHS_P2} epochs | lr={p2_lr}")
    for layer in model.layers:
        if hasattr(layer, "layers"):
            n = len(layer.layers)
            cutoff = int(n * (1.0 - unfreeze_pct))
            for i, sub in enumerate(layer.layers):
                sub.trainable = False if isinstance(sub, layers.BatchNormalization) \
                                else (i >= cutoff)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p2_lr, clipnorm=1.0),
                  loss=focal_loss(), metrics=get_metrics())
    t0 = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_P2, class_weight=class_weights,
                   callbacks=get_callbacks(save_path, best1), verbose=1)
    best2 = max(h2.history["val_accuracy"])
    print(f"  Phase 2: {(time.time()-t0)/60:.1f} min | acc={best2:.4f} | "
          f"auc={max(h2.history['val_auc']):.4f} | "
          f"rec={max(h2.history['val_recall']):.4f} | "
          f"prec={max(h2.history['val_precision']):.4f}")

    best = max(best1, best2)
    print(f"  BEST val_acc={best:.4f} saved → {save_path}")
    del model; gc.collect(); tf.keras.backend.clear_session()
    return best

# ── Train all 5 ───────────────────────────────────────────
t_start = time.time()
results = {}
for name in MODEL_CONFIGS:
    if (time.time() - t_start) / 3600 > 7.5:
        print(f"\n  TIME LIMIT — skipping {name}")
        continue
    try:
        results[name] = train_model(name)
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        import traceback; traceback.print_exc()
        results[name] = 0.0

print("\n" + "=" * 60)
print(f"  ALL DONE in {(time.time()-t_start)/60:.1f} min")
print("=" * 60)
for name, acc in results.items():
    print(f"  {name:20s}  val_acc={acc:.4f}")

# ── Evaluation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FULL EVALUATION")
print("=" * 60)

idx2cls         = {v: k for k, v in CLASS_TO_IDX.items()}
cls_names       = [CLASS_LABEL_MAP[idx2cls[i]] for i in range(NUM_CLASSES)]
true_val        = np.argmax(np.concatenate([y.numpy() for _, y in val_ds], axis=0), axis=1)
true_val_onehot = np.eye(NUM_CLASSES)[true_val]

def print_report(name, preds, probs):
    acc = accuracy_score(true_val, preds)
    auc = roc_auc_score(true_val_onehot, probs, multi_class="ovr", average="macro")
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"{'='*60}")
    print(classification_report(true_val, preds, target_names=cls_names, digits=3))
    print("  Per-class AUC:")
    for i, c in enumerate(cls_names):
        print(f"    {c:42s}: {roc_auc_score(true_val_onehot[:,i], probs[:,i]):.4f}")

all_probs = []
for name in MODEL_CONFIGS:
    path = os.path.join(OUTPUT_DIR, name + "_model.h5")
    if not os.path.exists(path):
        print(f"  SKIP {name} — not found"); continue
    m     = keras.models.load_model(path, compile=False)
    p     = m.predict(val_ds, verbose=0)
    preds = np.argmax(p, axis=1)
    all_probs.append(p)
    print_report(name, preds, p)
    del m; gc.collect()

if len(all_probs) >= 2:
    ep = np.argmax(np.mean(all_probs, axis=0), axis=1)
    print_report("ENSEMBLE", ep, np.mean(all_probs, axis=0))

# ── Files to download ─────────────────────────────────────
print("\n" + "=" * 60)
print("  FILES TO DOWNLOAD:")
print("=" * 60)
for fn in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, fn)
    print(f"  {fn:45s} {os.path.getsize(fp)/1e6:.1f} MB")
print("""
AFTER DOWNLOADING:
  1. Put all .keras files + class_indices.json → saved_models/
  2. streamlit run app.py
""")
