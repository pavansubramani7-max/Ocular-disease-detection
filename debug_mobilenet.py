"""Debug MobileNet structure"""
import tensorflow as tf
from tensorflow import keras
import config
from src.ensemble import ResNetPreprocess

# Load MobileNet
custom_objects = {
    "ResNetPreprocess": ResNetPreprocess,
    "loss_fn": lambda y_true, y_pred: y_pred,
}

mobilenet_path = config.MODEL_PATHS.get('mobilenet')
print(f"Loading from: {mobilenet_path}")

with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(mobilenet_path, compile=False)

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
model.summary()

print("\n" + "="*80)
print("LAYER DETAILS")
print("="*80)
for i, layer in enumerate(model.layers):
    print(f"\n[{i}] {layer.name} - {layer.__class__.__name__}")
    if hasattr(layer, 'layers'):
        print(f"    -> Nested model with {len(layer.layers)} layers")
        print(f"    -> Input shape: {layer.input_shape}")
        print(f"    -> Output shape: {layer.output_shape}")
        
        # Find last conv in nested model
        for j, sub in enumerate(reversed(layer.layers)):
            if 'Conv' in sub.__class__.__name__:
                print(f"    -> Last Conv: [{len(layer.layers)-j-1}] {sub.name} - {sub.__class__.__name__}")
                print(f"       Output shape: {sub.output_shape}")
                break
