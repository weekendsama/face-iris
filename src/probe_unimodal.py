from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import build_training_batch_iterator, build_training_dataset, collect_evaluation_records
from .losses import batch_hard_triplet_loss, classification_loss, pairwise_embedding_loss
from .model import AdaFaceHead, _build_backbone
from .presets import get_config


@dataclass(frozen=True)
class UnimodalSpec:
    modality: str
    modality_index: int
    input_shape: tuple[int, int, int]
    embedding_dim: int


def _spec_for(config, modality: str) -> UnimodalSpec:
    normalized = modality.strip().lower().replace("-", "_")
    if normalized == "face":
        return UnimodalSpec("face", 0, config.face_input_shape, config.face_embedding_dim)
    if normalized == "iris":
        return UnimodalSpec("iris", 1, config.iris_input_shape, config.iris_embedding_dim)
    raise ValueError("modality must be 'face' or 'iris'")


def unimodal_checkpoint_path(config, modality: str) -> str:
    spec = _spec_for(config, modality)
    return f"{config.checkpoint_dir}/{spec.modality}_probe"


def _build_unimodal_components(config, modality: str):
    spec = _spec_for(config, modality)
    base_backbone = _build_backbone(
        spec.input_shape,
        spec.embedding_dim,
        f"{spec.modality}_probe_backbone_base",
    )
    inputs = keras.Input(shape=spec.input_shape, name=f"{spec.modality}_probe_input")
    x = base_backbone(inputs)
    x = layers.BatchNormalization(name=f"{spec.modality}_probe_bn_1")(x)
    x = layers.ReLU(name=f"{spec.modality}_probe_relu")(x)
    x = layers.Dense(spec.embedding_dim, use_bias=False, name=f"{spec.modality}_probe_proj")(x)
    outputs = layers.BatchNormalization(name=f"{spec.modality}_probe_bn_2")(x)
    backbone = keras.Model(inputs, outputs, name=f"{spec.modality}_probe_backbone")
    head = AdaFaceHead(config.num_classes, name=f"{spec.modality}_probe_head")
    return spec, backbone, head


def _build_training_source(config, modality_index: int):
    train_batch_iterator = build_training_batch_iterator(config)
    train_dataset = None if train_batch_iterator is not None else build_training_dataset(config)

    if train_batch_iterator is not None:
        def _iterator():
            while True:
                inputs, labels = next(train_batch_iterator)
                yield inputs[modality_index], labels

        return _iterator()

    if train_dataset is not None:
        def _dataset_iterator():
            for inputs, labels in train_dataset.repeat():
                yield inputs[modality_index], labels

        return _dataset_iterator()

    raise RuntimeError("No training data available for unimodal probe.")


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


def _evaluate_embeddings(embeddings: tf.Tensor, labels: list[int]) -> tuple[float, float, dict[str, float]]:
    enroll_idx, probe_idx = _split_indices(labels)
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


def _compute_far_frr_eer(
    genuine_scores: list[float],
    impostor_scores: list[float],
) -> dict[str, float]:
    if not genuine_scores or not impostor_scores:
        nan = float("nan")
        return {"threshold": nan, "far": nan, "frr": nan, "eer": nan}

    thresholds = sorted(set(genuine_scores + impostor_scores))
    best = None
    for threshold in thresholds:
        far = sum(score >= threshold for score in impostor_scores) / len(impostor_scores)
        frr = sum(score < threshold for score in genuine_scores) / len(genuine_scores)
        gap = abs(far - frr)
        if best is None or gap < best["gap"]:
            best = {
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "eer": 0.5 * (far + frr),
                "gap": gap,
            }
    assert best is not None
    return {
        "threshold": best["threshold"],
        "far": best["far"],
        "frr": best["frr"],
        "eer": best["eer"],
    }


def _records_to_batch(records, config):
    face_batch = tf.stack(
        [
            tf.image.resize(
                tf.image.decode_image(
                    tf.io.read_file(str(item.face_path)),
                    channels=config.face_input_shape[-1],
                    expand_animations=False,
                ),
                config.face_input_shape[:2],
            )
            for item in records
        ],
        axis=0,
    )
    iris_batch = tf.stack(
        [
            tf.image.resize(
                tf.image.decode_image(
                    tf.io.read_file(str(item.iris_path)),
                    channels=config.iris_input_shape[-1],
                    expand_animations=False,
                ),
                config.iris_input_shape[:2],
            )
            for item in records
        ],
        axis=0,
    )
    face_batch = tf.cast(face_batch, tf.float32)
    iris_batch = tf.cast(iris_batch, tf.float32)
    labels = tf.constant([item.label for item in records], dtype=tf.int32)
    record_ids = [
        f"{item.label}|{item.face_path}|{item.iris_path}"
        for item in records
    ]
    return (face_batch, iris_batch), labels, record_ids


def train_unimodal_probe(config, modality: str) -> dict[str, object]:
    tf.keras.utils.set_random_seed(config.random_seed)
    spec, backbone, head = _build_unimodal_components(config, modality)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.unimodal_learning_rate)
    source = _build_training_source(config, spec.modality_index)

    final_loss = 0.0
    final_cls_loss = 0.0
    final_embedding_loss = 0.0
    final_triplet_loss = 0.0
    for _ in range(config.unimodal_train_steps):
        batch_inputs, labels = next(source)
        with tf.GradientTape() as tape:
            raw_embeddings = backbone(batch_inputs, training=True)
            embeddings = tf.math.l2_normalize(raw_embeddings, axis=-1)
            logits, norms = head(raw_embeddings)
            cls_loss = classification_loss(logits, norms, labels, config)
            embedding_loss = pairwise_embedding_loss(
                embeddings,
                labels,
                config.embedding_positive_target,
                config.embedding_negative_target,
            )
            triplet_loss = batch_hard_triplet_loss(
                embeddings,
                labels,
                config.triplet_margin,
            )
            loss = (
                config.unimodal_classification_weight * cls_loss
                + config.unimodal_embedding_pairwise_weight * embedding_loss
                + config.unimodal_triplet_weight * triplet_loss
            )
        variables = backbone.trainable_variables + head.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        final_loss = float(loss.numpy())
        final_cls_loss = float(cls_loss.numpy())
        final_embedding_loss = float(embedding_loss.numpy())
        final_triplet_loss = float(triplet_loss.numpy())

    save_path = unimodal_checkpoint_path(config, spec.modality)
    tf.io.gfile.makedirs(config.checkpoint_dir)
    checkpoint = tf.train.Checkpoint(backbone=backbone, head=head)
    checkpoint.write(save_path)
    return {
        "modality": spec.modality,
        "checkpoint_path": save_path,
        "train_final_loss": final_loss,
        "train_final_cls_loss": final_cls_loss,
        "train_final_embedding_loss": final_embedding_loss,
        "train_final_triplet_loss": final_triplet_loss,
    }


def evaluate_unimodal_probe(config, modality: str) -> dict[str, object]:
    spec, backbone, head = _build_unimodal_components(config, modality)
    records = collect_evaluation_records(config)
    if not records:
        raise RuntimeError("No evaluation data available for unimodal probe.")

    inputs, labels_tensor, _ = _records_to_batch(records[:64], config)
    eval_inputs = inputs[spec.modality_index]
    _ = backbone(eval_inputs, training=False)
    raw_embeddings = backbone(eval_inputs, training=False)
    _ = head(raw_embeddings)

    checkpoint_path = unimodal_checkpoint_path(config, spec.modality)
    index_path = f"{checkpoint_path}.index"
    if not tf.io.gfile.exists(index_path):
        raise RuntimeError(f"Unimodal checkpoint not found: {checkpoint_path}")

    checkpoint = tf.train.Checkpoint(backbone=backbone, head=head)
    checkpoint.restore(checkpoint_path).expect_partial()

    eval_raw = backbone(eval_inputs, training=False)
    eval_embeddings = tf.math.l2_normalize(eval_raw, axis=-1)
    labels = labels_tensor.numpy().tolist()
    genuine_mean, impostor_mean, metrics = _evaluate_embeddings(eval_embeddings, labels)
    return {
        "modality": spec.modality,
        "checkpoint_path": checkpoint_path,
        "genuine_mean": genuine_mean,
        "impostor_mean": impostor_mean,
        "metrics": metrics,
    }


def _print_evaluation(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    print("modality:", result["modality"])
    print("checkpoint_path:", result["checkpoint_path"])
    print(
        "embedding genuine_mean={genuine:.4f} impostor_mean={impostor:.4f} "
        "threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            genuine=result["genuine_mean"],
            impostor=result["impostor_mean"],
            **metrics,
        )
    )


def main() -> None:
    config, preset_name = get_config()
    print("preset:", preset_name)
    for modality in ("face", "iris"):
        train_result = train_unimodal_probe(config, modality)
        print("modality:", train_result["modality"])
        print("train_final_loss:", f"{train_result['train_final_loss']:.4f}")
        print("train_final_cls_loss:", f"{train_result['train_final_cls_loss']:.4f}")
        print("train_final_embedding_loss:", f"{train_result['train_final_embedding_loss']:.4f}")
        print("train_final_triplet_loss:", f"{train_result['train_final_triplet_loss']:.4f}")
        print("checkpoint_path:", train_result["checkpoint_path"])
        eval_result = evaluate_unimodal_probe(config, modality)
        _print_evaluation(eval_result)


if __name__ == "__main__":
    main()
