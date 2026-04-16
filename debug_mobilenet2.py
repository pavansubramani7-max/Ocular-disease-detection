"""Debug MobileNet SE block connections"""
import tensorflow as tf
from tensorflow import keras
import config
from src.ensemble import ResNetPreprocess

custom_objects = {"ResNetPreprocess": ResNetPreprocess, "loss_fn": lambda y_true, y_pred: y_pred}
with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(config.MODEL_PATHS.get('mobilenet'), compile=False)

print("Main model layers with inbound nodes:")
for i, layer in enumerate(model.layers):
    try:
        inbound = [n.inbound_layers for n in layer._inbound_nodes]
        print(f"[{i}] {layer.name:40s} <- {[l.name if hasattr(l,'name') else str(l) for n in layer._inbound_nodes for l in (n.inbound_layers if isinstance(n.inbound_layers, list) else [n.inbound_layers])]}")
    except:
        print(f"[{i}] {layer.name:40s} <- (no inbound info)")
