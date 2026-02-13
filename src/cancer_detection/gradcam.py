from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf


def infer_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            try:
                return infer_last_conv_layer(layer)
            except ValueError:
                continue
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def generate_gradcam(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_index: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Create Grad-CAM heatmap for a single image tensor of shape (1, H, W, 3)."""
    if last_conv_layer_name is None:
        last_conv_layer_name = infer_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_tensor)
        if class_index is None:
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_index = tf.argmax(predictions[0])
                class_channel = predictions[:, class_index]
        else:
            class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()
