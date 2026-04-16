"""
Rebuilds ResNet model without Lambda layer and transfers weights from saved model.
Run once: python fix_resnet.py
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import config

IMG_SIZE    = 160
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = config.NUM_CLASSES

# Load old model with custom object scope to handle anonymous lambda
def old_preprocess(x):
    return keras.applications.resnet50.preprocess_input(x * 255.0)

print("Loading old ResNet model...")
with keras.utils.custom_object_scope({"old_preprocess": old_preprocess}):
    old_model = keras.models.load_model(
        config.MODEL_PATHS["resnet"], compile=False)

print("Old model loaded. Rebuilding without Lambda...")

# Build new ResNet using Rescaling instead of Lambda
class ResNetPreprocess(layers.Layer):
    def call(self, x):
        x = x * 255.0
        # ResNet50 preprocess: subtract ImageNet mean, BGR
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = x[..., ::-1]  # RGB to BGR
        x = x - mean
        return x

def se_block(x, ratio=16):
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(ch // ratio, activation="relu")(se)
    se = layers.Dense(ch, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, ch))(se)
    return layers.Multiply()([x, se])

def build_head(x, prefix):
    x = layers.Dense(512, name=prefix + "_d1")(x)
    x = layers.BatchNormalization(name=prefix + "_bn1")(x)
    x = layers.Activation("relu", name=prefix + "_act1")(x)
    x = layers.Dropout(0.45, name=prefix + "_drop1")(x)
    x = layers.Dense(256, name=prefix + "_d2")(x)
    x = layers.BatchNormalization(name=prefix + "_bn2")(x)
    x = layers.Activation("relu", name=prefix + "_act2")(x)
    x = layers.Dropout(0.35, name=prefix + "_drop2")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name=prefix + "_out")(x)

inp      = layers.Input(shape=INPUT_SHAPE, name="res_in")
x        = ResNetPreprocess(name="res_preprocess")(inp)
backbone = ResNet50(include_top=False, weights=None, input_tensor=x)
x        = se_block(backbone.output)
x        = layers.GlobalAveragePooling2D()(x)
new_model = Model(inputs=inp, outputs=build_head(x, "res"), name="ResNet50_fixed")

# Copy weights from old model
print("Copying weights...")
old_weights = old_model.get_weights()
new_model.set_weights(old_weights)

# Save fixed model
new_model.save(config.MODEL_PATHS["resnet"])
print("Fixed ResNet saved ->", config.MODEL_PATHS["resnet"])

# Verify it loads cleanly
test = keras.models.load_model(config.MODEL_PATHS["resnet"], compile=False)
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
out = test.predict(dummy, verbose=0)
print("Verification passed. Output shape:", out.shape)
print("Done! Now run: streamlit run app.py")
