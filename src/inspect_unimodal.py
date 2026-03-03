from __future__ import annotations

import tensorflow as tf

from .data import collect_evaluation_records
from .evaluate import _compute_far_frr_eer, _records_to_batch
from .model import build_model
from .presets import get_config
from .train import latest_checkpoint_path


def _split_indices(labels: list[int]) -> tuple[list[int], list[int]]:
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
    return enroll_idx, probe_idx


def _embedding_metrics(
    embeddings: tf.Tensor,
    labels: list[int],
    enroll_idx: list[int],
    probe_idx: list[int],
) -> tuple[float, float, dict[str, float]]:
    enroll = tf.gather(embeddings, enroll_idx)
    probe = tf.gather(embeddings, probe_idx)
    enroll_labels = tf.constant([labels[idx] for idx in enroll_idx], dtype=tf.int32)
    probe_labels = tf.constant([labels[idx] for idx in probe_idx], dtype=tf.int32)

    similarities = tf.matmul(enroll, probe, transpose_b=True)
    same_identity = tf.equal(enroll_labels[:, None], probe_labels[None, :])

    genuine = tf.boolean_mask(similarities, same_identity).numpy().tolist()
    impostor = tf.boolean_mask(similarities, tf.logical_not(same_identity)).numpy().tolist()
    metrics = _compute_far_frr_eer(genuine, impostor)
    return sum(genuine) / len(genuine), sum(impostor) / len(impostor), metrics


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

    outputs = model(inputs, training=False)
    labels = labels_tensor.numpy().tolist()
    enroll_idx, probe_idx = _split_indices(labels)

    print("preset:", preset_name)
    print("checkpoint_path:", ckpt_path)
    for name in ("face_embedding", "iris_embedding", "fused_embedding"):
        genuine_mean, impostor_mean, metrics = _embedding_metrics(
            outputs[name],
            labels,
            enroll_idx,
            probe_idx,
        )
        print(
            "{name} genuine_mean={genuine:.4f} impostor_mean={impostor:.4f} "
            "threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
                name=name,
                genuine=genuine_mean,
                impostor=impostor_mean,
                **metrics,
            )
        )


if __name__ == "__main__":
    main()
