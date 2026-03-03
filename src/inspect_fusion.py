from __future__ import annotations

import tensorflow as tf

from .data import collect_evaluation_records
from .evaluate import _records_to_batch
from .model import build_model
from .presets import get_config
from .train import latest_checkpoint_path


def _pairwise_cosine_stats(embeddings: tf.Tensor, labels: list[int]) -> tuple[float, float]:
    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)

    enroll_idx: list[int] = []
    probe_idx: list[int] = []
    for label in sorted(grouped):
        indices = grouped[label]
        split_point = max(1, len(indices) // 2)
        enroll = indices[:split_point]
        probe = indices[split_point:]
        if not probe:
            continue
        enroll_idx.extend(enroll)
        probe_idx.extend(probe)

    enroll = tf.gather(embeddings, enroll_idx)
    probe = tf.gather(embeddings, probe_idx)
    enroll_labels = tf.constant([labels[idx] for idx in enroll_idx], dtype=tf.int32)
    probe_labels = tf.constant([labels[idx] for idx in probe_idx], dtype=tf.int32)
    similarities = tf.matmul(enroll, probe, transpose_b=True)
    same_identity = tf.equal(enroll_labels[:, None], probe_labels[None, :])
    genuine = tf.boolean_mask(similarities, same_identity)
    impostor = tf.boolean_mask(similarities, tf.logical_not(same_identity))
    return float(tf.reduce_mean(genuine).numpy()), float(tf.reduce_mean(impostor).numpy())


def _print_stage(name: str, tensor: tf.Tensor, labels: list[int]) -> None:
    feature_std = tf.math.reduce_std(tensor, axis=0)
    sample_norms = tf.norm(tensor, axis=-1)
    normalized = tf.math.l2_normalize(tensor, axis=-1)
    genuine_mean, impostor_mean = _pairwise_cosine_stats(normalized, labels)
    print(
        "{name} dim={dim} mean_norm={mean_norm:.4f} mean_feature_std={mean_std:.6f} "
        "genuine_cos={genuine:.4f} impostor_cos={impostor:.4f}".format(
            name=name,
            dim=int(tensor.shape[-1]),
            mean_norm=float(tf.reduce_mean(sample_norms).numpy()),
            mean_std=float(tf.reduce_mean(feature_std).numpy()),
            genuine=genuine_mean,
            impostor=impostor_mean,
        )
    )


def main() -> None:
    config, preset_name = get_config()
    model = build_model(config)
    records = collect_evaluation_records(config)
    if not records:
        print("status: no eval records")
        return

    inputs, labels_tensor, _ = _records_to_batch(records[:64], config)
    _ = model(inputs, training=False)
    ckpt_path = latest_checkpoint_path(config)
    if ckpt_path is None:
        print("status: no checkpoint")
        return
    model.load_weights(ckpt_path)

    face_inputs, iris_inputs = inputs
    face_raw = model.face_backbone(face_inputs, training=False)
    iris_raw = model.iris_backbone(iris_inputs, training=False)
    face_embedding = tf.math.l2_normalize(face_raw, axis=-1)
    iris_embedding = tf.math.l2_normalize(iris_raw, axis=-1)
    face_projected = model.face_project(face_embedding, training=False)
    iris_projected = model.iris_project(iris_embedding, training=False)
    fused_inputs = tf.concat([face_projected, iris_projected], axis=-1)
    fused_raw = model.fusion(fused_inputs, training=False)
    fused_after_dropout = model.fusion_dropout(fused_raw, training=False)

    labels = labels_tensor.numpy().tolist()

    print("preset:", preset_name)
    print("checkpoint_path:", ckpt_path)
    _print_stage("face_embedding", face_embedding, labels)
    _print_stage("iris_embedding", iris_embedding, labels)
    _print_stage("face_projected", face_projected, labels)
    _print_stage("iris_projected", iris_projected, labels)
    _print_stage("fused_inputs", fused_inputs, labels)
    _print_stage("fused_raw", fused_raw, labels)
    _print_stage("fused_after_dropout", fused_after_dropout, labels)


if __name__ == "__main__":
    main()
