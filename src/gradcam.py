"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    """Generate Grad-CAM heatmaps for model interpretability"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM
        
        Args:
            model: Keras model
            layer_name: Name of the layer to compute gradients for
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name:
                return layer.name
        return None
    
    def generate_heatmap(self, image, class_idx, eps=1e-8):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Input image (H, W, 3)
            class_idx: Class index for which to generate heatmap
            eps: Small value to avoid division by zero
        
        Returns:
            heatmap: Normalized heatmap (H, W)
        """
        # Create model that outputs the target layer activations and predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the activations
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (H, W, 3)
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Transparency of heatmap
        
        Returns:
            overlaid_image: Image with heatmap overlay (H, W, 3)
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to color
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert image to uint8 if needed
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Overlay
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid
    
    def visualize(self, image, class_idx, class_name=None):
        """
        Visualize Grad-CAM
        
        Args:
            image: Input image
            class_idx: Class index
            class_name: Class name for title
        """
        heatmap = self.generate_heatmap(image, class_idx)
        overlaid = self.overlay_heatmap(image, heatmap)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlaid)
        plt.title(f'Heatmap Overlay (Class: {class_name or class_idx})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
