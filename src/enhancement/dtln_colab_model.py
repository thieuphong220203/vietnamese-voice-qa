"""Serializable Keras layers used by the fine-tuned full DTLN model.

The training notebook saves these layers under the ``DTLN`` package.  Keeping
their registrations here lets inference load a ``.keras`` artifact without
requiring the notebook that originally defined the model.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="DTLN")
class STFT(keras.layers.Layer):
    """Return the exact framed RFFT features used by the training notebook."""

    def call(self, inputs):
        import tensorflow as tf

        spectrum = tf.signal.rfft(tf.signal.frame(inputs, 512, 128))
        return tf.abs(spectrum), tf.math.angle(spectrum)


@keras.saving.register_keras_serializable(package="DTLN")
class LogMagnitude(keras.layers.Layer):
    """Stable log-magnitude feature used by the first DTLN stage."""

    def call(self, inputs):
        return ops.log(inputs + 1e-7)


@keras.saving.register_keras_serializable(package="DTLN")
class InstantLayerNormalization(keras.layers.Layer):
    """Normalize every time frame independently over its feature dimension."""

    def build(self, input_shape):
        feature_count = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(feature_count,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(feature_count,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        mean = ops.mean(inputs, axis=-1, keepdims=True)
        variance = ops.mean(ops.square(inputs - mean), axis=-1, keepdims=True)
        normalized = (inputs - mean) / ops.sqrt(variance + 1e-7)
        return normalized * self.gamma + self.beta


@keras.saving.register_keras_serializable(package="DTLN")
class InverseFFT(keras.layers.Layer):
    """Reconstruct time-domain frames from magnitude and phase."""

    def call(self, inputs):
        import tensorflow as tf

        magnitude, phase = inputs
        spectrum = tf.cast(magnitude, tf.complex64) * tf.exp(
            tf.complex(tf.zeros_like(phase), phase)
        )
        return tf.signal.irfft(spectrum, fft_length=[512])


@keras.saving.register_keras_serializable(package="DTLN")
class OverlapAdd(keras.layers.Layer):
    """Reconstruct a waveform from 512-sample frames at a 128-sample hop."""

    def call(self, inputs):
        import tensorflow as tf

        return tf.signal.overlap_and_add(inputs, frame_step=128)
