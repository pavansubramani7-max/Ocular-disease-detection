"""
Model architectures: EfficientNet, MobileNet, ResNet with LSTM
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np

class EfficientNetModel:
    """EfficientNet model for image classification"""
    
    @staticmethod
    def build(num_classes, input_shape=(224, 224, 3)):
        """Build EfficientNetB0 model"""
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class MobileNetModel:
    """MobileNet model for image classification"""
    
    @staticmethod
    def build(num_classes, input_shape=(224, 224, 3)):
        """Build MobileNetV2 model"""
        base_model = applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class ResNetModel:
    """ResNet model for image classification"""
    
    @staticmethod
    def build(num_classes, input_shape=(224, 224, 3)):
        """Build ResNet50 model"""
        base_model = applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class LSTMSequenceModel:
    """LSTM model for temporal/sequence analysis"""
    
    @staticmethod
    def build(num_classes, sequence_length=10, feature_dim=256):
        """Build LSTM model"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(sequence_length, feature_dim)),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
