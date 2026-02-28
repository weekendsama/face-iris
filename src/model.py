from __future__ import annotations

from collections import defaultdict, deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import ModelConfig
from .dsh import fit_dsh_projections


class SmoothBinaryLayer(layers.Layer):
    """Differentiable approximation during training, hard threshold at inference."""

    def __init__(self, bits: int, alpha: float, name: str | None = None) -> None:
        super().__init__(name=name)
        self.bits = bits
        self.alpha = alpha

    def build(self, input_shape: tf.TensorShape) -> None:
        feat_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(feat_dim, self.bits),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.bits,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def soft_call(self, inputs: tf.Tensor) -> tf.Tensor:
        logits = tf.matmul(inputs, self.kernel) + self.bias
        return tf.sigmoid(self.alpha * logits)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        probs = self.soft_call(inputs)
        if training:
            return probs
        return tf.cast(probs >= 0.5, tf.float32)

    def set_projections(self, weights: tf.Tensor, bias: tf.Tensor, mix: float = 1.0) -> None:
        new_kernel = tf.cast(weights, self.kernel.dtype)
        new_bias = tf.cast(bias, self.bias.dtype)
        mix = tf.cast(tf.clip_by_value(mix, 0.0, 1.0), self.kernel.dtype)
        self.kernel.assign((1.0 - mix) * self.kernel + mix * new_kernel)
        self.bias.assign((1.0 - mix) * self.bias + mix * new_bias)


class AdaFaceHead(layers.Layer):
    """Cosine classifier used by AdaFace-style margin losses."""

    def __init__(self, num_classes: int, name: str | None = None) -> None:
        super().__init__(name=name)
        self.num_classes = num_classes

    def build(self, input_shape: tf.TensorShape) -> None:
        feat_dim = int(input_shape[-1])
        self.class_weights = self.add_weight(
            name="class_weights",
            shape=(feat_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, embeddings: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        norms = tf.norm(embeddings, axis=-1, keepdims=True)
        normalized_embeddings = tf.math.l2_normalize(embeddings, axis=-1)
        normalized_weights = tf.math.l2_normalize(self.class_weights, axis=0)
        cosine_logits = tf.matmul(normalized_embeddings, normalized_weights)
        cosine_logits = tf.clip_by_value(cosine_logits, -1.0 + 1e-7, 1.0 - 1e-7)
        return cosine_logits, norms


def _build_backbone(
    input_shape: tuple[int, int, int],
    embedding_dim: int,
    name: str,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(embedding_dim)(x)
    return keras.Model(inputs, outputs, name=name)


class MultimodalCancelableModel(keras.Model):
    """Face + iris feature extraction, unimodal hashing, multimodal fusion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(name="multimodal_cancelable_model")
        self.config = config
        self.face_backbone = _build_backbone(
            config.face_input_shape,
            config.face_embedding_dim,
            "face_backbone",
        )
        self.iris_backbone = _build_backbone(
            config.iris_input_shape,
            config.iris_embedding_dim,
            "iris_backbone",
        )
        self.face_hash = SmoothBinaryLayer(
            bits=config.face_embedding_dim,
            alpha=config.smoothing_alpha,
            name="face_hash",
        )
        self.iris_hash = SmoothBinaryLayer(
            bits=config.iris_embedding_dim,
            alpha=config.smoothing_alpha,
            name="iris_hash",
        )
        self.face_project = layers.Dense(config.shared_embedding_dim, activation="relu")
        self.iris_project = layers.Dense(config.shared_embedding_dim, activation="relu")
        self.fusion = layers.Dense(config.shared_embedding_dim, activation="relu")
        self.fusion_norm = layers.LayerNormalization()
        self.fusion_dropout = layers.Dropout(0.1)
        self.shared_hash = SmoothBinaryLayer(
            bits=config.shared_hash_bits,
            alpha=config.smoothing_alpha,
            name="shared_hash",
        )
        self.face_classifier = AdaFaceHead(config.num_classes, name="face_logits")
        self.iris_classifier = AdaFaceHead(config.num_classes, name="iris_logits")
        self.shared_classifier = AdaFaceHead(config.num_classes, name="shared_logits")
        self._face_feature_bank = defaultdict(
            lambda: deque(maxlen=config.dsh_bank_per_class)
        )
        self._iris_feature_bank = defaultdict(
            lambda: deque(maxlen=config.dsh_bank_per_class)
        )
        self._shared_feature_bank = defaultdict(
            lambda: deque(maxlen=config.dsh_bank_per_class)
        )

    def _feature_bank_ready(self, feature_bank) -> bool:
        non_empty_classes = 0
        for class_id in feature_bank:
            class_count = len(feature_bank[class_id])
            if class_count == 0:
                continue
            non_empty_classes += 1
            if class_count < self.config.dsh_min_samples_per_class:
                return False
        return non_empty_classes >= 2

    def _update_feature_bank(
        self,
        feature_bank,
        features: tf.Tensor,
        labels: tf.Tensor,
    ) -> tf.Tensor:
        features_np = features.numpy()
        labels_np = labels.numpy()
        for feature, label in zip(features_np, labels_np):
            feature_bank[int(label)].append(feature)

        stacked = []
        for class_id in sorted(feature_bank):
            class_features = list(feature_bank[class_id])
            if class_features:
                stacked.extend(class_features)

        return tf.convert_to_tensor(stacked, dtype=features.dtype)

    def refresh_hash_projections(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        labels: tf.Tensor,
    ) -> bool:
        """Update hash projections using an identity-stratified feature bank."""
        face_inputs, iris_inputs = inputs
        face_raw = self.face_backbone(face_inputs, training=False)
        iris_raw = self.iris_backbone(iris_inputs, training=False)
        face_embedding = tf.math.l2_normalize(face_raw, axis=-1)
        iris_embedding = tf.math.l2_normalize(iris_raw, axis=-1)
        face_code = self.face_hash(face_embedding, training=True)
        iris_code = self.iris_hash(iris_embedding, training=True)
        fused_raw = self.fusion(
            self.face_project(face_code) + self.iris_project(iris_code),
            training=False,
        )
        fused_raw = self.fusion_norm(fused_raw, training=False)
        fused_embedding = tf.math.l2_normalize(fused_raw, axis=-1)
        _ = self.shared_hash(fused_embedding, training=True)
        banked_face = self._update_feature_bank(
            self._face_feature_bank,
            face_embedding,
            labels,
        )
        banked_iris = self._update_feature_bank(
            self._iris_feature_bank,
            iris_embedding,
            labels,
        )
        banked_shared = self._update_feature_bank(
            self._shared_feature_bank,
            fused_embedding,
            labels,
        )
        if not (
            self._feature_bank_ready(self._face_feature_bank)
            and self._feature_bank_ready(self._iris_feature_bank)
            and self._feature_bank_ready(self._shared_feature_bank)
        ):
            return False

        face_weights, face_bias = fit_dsh_projections(
            banked_face.numpy(),
            num_bits=self.config.face_embedding_dim,
            num_clusters=self.config.dsh_num_clusters,
            num_iters=self.config.dsh_kmeans_iters,
        )
        iris_weights, iris_bias = fit_dsh_projections(
            banked_iris.numpy(),
            num_bits=self.config.iris_embedding_dim,
            num_clusters=self.config.dsh_num_clusters,
            num_iters=self.config.dsh_kmeans_iters,
        )
        shared_weights, shared_bias = fit_dsh_projections(
            banked_shared.numpy(),
            num_bits=self.config.shared_hash_bits,
            num_clusters=self.config.dsh_num_clusters,
            num_iters=self.config.dsh_kmeans_iters,
        )
        self.face_hash.set_projections(
            face_weights,
            face_bias,
            mix=self.config.dsh_projection_mix,
        )
        self.iris_hash.set_projections(
            iris_weights,
            iris_bias,
            mix=self.config.dsh_projection_mix,
        )
        self.shared_hash.set_projections(
            shared_weights,
            shared_bias,
            mix=self.config.dsh_projection_mix,
        )
        return True

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        face_inputs, iris_inputs = inputs
        face_raw = self.face_backbone(face_inputs, training=training)
        iris_raw = self.iris_backbone(iris_inputs, training=training)
        face_embedding = tf.math.l2_normalize(face_raw, axis=-1)
        iris_embedding = tf.math.l2_normalize(iris_raw, axis=-1)

        face_code = self.face_hash(face_embedding, training=training)
        iris_code = self.iris_hash(iris_embedding, training=training)

        fused_raw = self.fusion(
            self.face_project(face_code) + self.iris_project(iris_code),
            training=training,
        )
        fused_raw = self.fusion_norm(fused_raw, training=training)
        fused_raw = self.fusion_dropout(fused_raw, training=training)
        fused = tf.math.l2_normalize(fused_raw, axis=-1)
        shared_code_soft = self.shared_hash(fused, training=True)
        if training:
            shared_code = shared_code_soft
        else:
            shared_code = tf.cast(shared_code_soft >= 0.5, tf.float32)
        face_logits, face_norms = self.face_classifier(face_raw)
        iris_logits, iris_norms = self.iris_classifier(iris_raw)
        shared_logits, shared_norms = self.shared_classifier(fused_raw)

        return {
            "face_embedding": face_embedding,
            "iris_embedding": iris_embedding,
            "face_code": face_code,
            "iris_code": iris_code,
            "fused_embedding": fused,
            "shared_code_soft": shared_code_soft,
            "shared_code": shared_code,
            "face_logits": face_logits,
            "face_norms": face_norms,
            "iris_logits": iris_logits,
            "iris_norms": iris_norms,
            "shared_logits": shared_logits,
            "shared_norms": shared_norms,
        }


def build_model(config: ModelConfig) -> MultimodalCancelableModel:
    return MultimodalCancelableModel(config)
