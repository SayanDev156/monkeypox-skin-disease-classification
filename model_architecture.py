from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from vit_keras import vit

    VIT_KERAS_AVAILABLE = True
except ImportError:
    VIT_KERAS_AVAILABLE = False


@dataclass
class ModelConfig:
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 5
    dropout: float = 0.35


class VisionTransformerClassifier:
    def __init__(self, config: ModelConfig, use_pretrained_vit: bool = True):
        self.config = config
        self.use_pretrained_vit = use_pretrained_vit
        self.model = self._build_model()

    def _build_fallback_transformer(self) -> keras.Model:
        inputs = keras.Input(shape=(*self.config.image_size, 3))
        patch_size = 16
        projection_dim = 128
        num_heads = 4
        transformer_layers = 6

        patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
        patches = layers.Reshape((-1, projection_dim))(patches)

        positions = tf.range(start=0, limit=patches.shape[1], delta=1)
        position_embedding = layers.Embedding(input_dim=patches.shape[1], output_dim=projection_dim)(positions)
        x = patches + position_embedding

        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
            x2 = layers.Add()([attention_output, x])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = layers.Dense(projection_dim * 4, activation="gelu")(x3)
            x3 = layers.Dropout(self.config.dropout)(x3)
            x3 = layers.Dense(projection_dim)(x3)
            x = layers.Add()([x3, x2])

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation="gelu")(x)
        x = layers.Dropout(self.config.dropout)(x)
        outputs = layers.Dense(self.config.num_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs, name="vit_fallback_classifier")

    def _build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(*self.config.image_size, 3))

        if self.use_pretrained_vit and VIT_KERAS_AVAILABLE:
            vit_backbone = vit.vit_b16(
                image_size=self.config.image_size[0],
                pretrained=True,
                include_top=False,
                pretrained_top=False,
            )
            vit_backbone.trainable = False
            x = vit_backbone(inputs)
            x = layers.Dense(512, activation="gelu")(x)
            x = layers.Dropout(self.config.dropout)(x)
            outputs = layers.Dense(self.config.num_classes, activation="softmax")(x)
            model = keras.Model(inputs, outputs, name="vit_b16_classifier")
            model.vit_backbone = vit_backbone
            return model

        return self._build_fallback_transformer()

    def set_fine_tune(self, unfreeze_last_blocks: int = 20):
        if hasattr(self.model, "vit_backbone"):
            backbone = self.model.vit_backbone
            backbone.trainable = True
            if unfreeze_last_blocks > 0 and len(backbone.layers) > unfreeze_last_blocks:
                freeze_until = len(backbone.layers) - unfreeze_last_blocks
                for layer in backbone.layers[:freeze_until]:
                    layer.trainable = False

    def compile(self, learning_rate: float):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

    def save(self, model_path: str):
        self.model.save(model_path)


def create_model(
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    use_pretrained_vit: bool = True,
) -> VisionTransformerClassifier:
    config = ModelConfig(image_size=image_size, num_classes=num_classes)
    return VisionTransformerClassifier(config, use_pretrained_vit=use_pretrained_vit)
