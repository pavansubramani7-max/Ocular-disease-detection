"""Test Grad-CAM for MobileNet"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import config
from src.ensemble import ResNetPreprocess

custom_objects = {"ResNetPreprocess": ResNetPreprocess, "loss_fn": lambda y_true, y_pred: y_pred}
with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(config.MODEL_PATHS.get('mobilenet'), compile=False)

img = np.random.rand(1, 160, 160, 3).astype(np.float32)
img_tensor = tf.constant(img)

print("Model input:", model.input)
print("Model output:", model.output)
print()

# Print all layer outputs in the main model graph
for i, layer in enumerate(model.layers):
    try:
        print(f"[{i}] {layer.name} -> output: {layer.output}")
    except Exception as e:
        print(f"[{i}] {layer.name} -> ERROR: {e}")

print("\n--- Trying grad model with multiply output ---")
try:
    # Use multiply output (after SE attention) as the feature map
    multiply_out = model.get_layer('multiply').output
    grad_model = tf.keras.Model(inputs=model.input, outputs=[multiply_out, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    print("SUCCESS with multiply output! grads shape:", grads.shape)
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- Trying grad model with backbone output ---")
try:
    backbone_out = model.get_layer('mobilenetv2_1.00_160').output
    grad_model = tf.keras.Model(inputs=model.input, outputs=[backbone_out, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    print("SUCCESS with backbone output! grads shape:", grads.shape)
except Exception as e:
    print(f"FAILED: {e}")
