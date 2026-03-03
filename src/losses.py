from __future__ import annotations

import tensorflow as tf

from .config import ModelConfig


def _apply_adaface_margin(
    cosine_logits: tf.Tensor,
    norms: tf.Tensor,
    labels: tf.Tensor,
    margin: float,
    scale: float,
    h: float,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    norms = tf.stop_gradient(tf.squeeze(norms, axis=-1))
    mean = tf.reduce_mean(norms)
    std = tf.math.reduce_std(norms)
    std = tf.maximum(std, 1e-3)

    margin_scaler = h * (norms - mean) / std
    margin_scaler = tf.clip_by_value(margin_scaler, -1.0, 1.0)
    adaptive_margin = margin * (1.0 + margin_scaler)

    one_hot = tf.one_hot(labels, depth=tf.shape(cosine_logits)[-1], dtype=tf.float32)
    theta = tf.acos(cosine_logits)
    target_logits = tf.cos(theta + adaptive_margin[:, None] * one_hot)
    logits = cosine_logits * (1.0 - one_hot) + target_logits * one_hot
    return logits * scale


def classification_loss(
    cosine_logits: tf.Tensor,
    norms: tf.Tensor,
    labels: tf.Tensor,
    config: ModelConfig,
) -> tf.Tensor:
    logits = _apply_adaface_margin(
        cosine_logits,
        norms,
        labels,
        config.adaface_margin,
        config.adaface_scale,
        config.adaface_h,
    )
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)


def pairwise_hash_loss(codes: tf.Tensor, labels: tf.Tensor, margin: float, positive_target: float) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    codes = tf.cast(codes, tf.float32)

    diffs = tf.abs(codes[:, None, :] - codes[None, :, :])
    distances = tf.reduce_mean(diffs, axis=-1)

    same_identity = tf.equal(labels[:, None], labels[None, :])
    positive_mask = tf.logical_and(same_identity, ~tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    negative_mask = ~same_identity

    positive_term = tf.boolean_mask(tf.nn.relu(distances - positive_target), positive_mask)
    negative_term = tf.boolean_mask(tf.nn.relu(margin - distances), negative_mask)

    positive_loss = tf.cond(
        tf.size(positive_term) > 0,
        lambda: tf.reduce_mean(positive_term),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    negative_loss = tf.cond(
        tf.size(negative_term) > 0,
        lambda: tf.reduce_mean(negative_term),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    return positive_loss + negative_loss


def pairwise_embedding_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    positive_target: float,
    negative_target: float,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    embeddings = tf.cast(embeddings, tf.float32)

    similarities = tf.matmul(embeddings, embeddings, transpose_b=True)
    same_identity = tf.equal(labels[:, None], labels[None, :])
    positive_mask = tf.logical_and(same_identity, ~tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    negative_mask = ~same_identity

    positive_term = tf.boolean_mask(tf.nn.relu(positive_target - similarities), positive_mask)
    negative_term = tf.boolean_mask(tf.nn.relu(similarities - negative_target), negative_mask)

    positive_loss = tf.cond(
        tf.size(positive_term) > 0,
        lambda: tf.reduce_mean(positive_term),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    negative_loss = tf.cond(
        tf.size(negative_term) > 0,
        lambda: tf.reduce_mean(negative_term),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    return positive_loss + negative_loss


def batch_hard_triplet_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    margin: float,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.int32)
    embeddings = tf.cast(embeddings, tf.float32)

    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    distances = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)

    same_identity = tf.equal(labels[:, None], labels[None, :])
    positive_mask = tf.logical_and(same_identity, ~tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    negative_mask = ~same_identity

    negative_fill = tf.fill(tf.shape(distances), tf.constant(-1e9, dtype=distances.dtype))
    positive_fill = tf.fill(tf.shape(distances), tf.constant(1e9, dtype=distances.dtype))

    hardest_positive = tf.reduce_max(tf.where(positive_mask, distances, negative_fill), axis=1)
    hardest_negative = tf.reduce_min(tf.where(negative_mask, distances, positive_fill), axis=1)

    valid_anchor = tf.reduce_any(positive_mask, axis=1) & tf.reduce_any(negative_mask, axis=1)
    triplet_terms = tf.nn.relu(hardest_positive - hardest_negative + margin)
    valid_terms = tf.boolean_mask(triplet_terms, valid_anchor)

    return tf.cond(
        tf.size(valid_terms) > 0,
        lambda: tf.reduce_mean(valid_terms),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def compute_total_loss(
    outputs: dict[str, tf.Tensor],
    labels: tf.Tensor,
    config: ModelConfig,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    face_cls = classification_loss(outputs["face_logits"], outputs["face_norms"], labels, config)
    iris_cls = classification_loss(outputs["iris_logits"], outputs["iris_norms"], labels, config)
    cls_loss = (face_cls + iris_cls) / 2.0

    pair_loss = pairwise_hash_loss(
        outputs["shared_code"],
        labels,
        config.pair_margin,
        config.pair_positive_target,
    )
    embedding_loss = pairwise_embedding_loss(
        outputs["fused_embedding"],
        labels,
        config.embedding_positive_target,
        config.embedding_negative_target,
    )
    triplet_loss = batch_hard_triplet_loss(
        outputs["fused_embedding"],
        labels,
        config.triplet_margin,
    )

    total = (
        config.classification_weight * cls_loss
        + config.pairwise_weight * pair_loss
        + config.embedding_pairwise_weight * embedding_loss
        + config.triplet_weight * triplet_loss
    )
    metrics = {
        "total_loss": total,
        "classification_loss": cls_loss,
        "pairwise_loss": pair_loss,
        "embedding_pairwise_loss": embedding_loss,
        "triplet_loss": triplet_loss,
    }
    return total, metrics
