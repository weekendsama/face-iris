from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import build_training_batch_iterator, build_training_dataset, collect_evaluation_records
from .losses import batch_hard_triplet_loss, classification_loss, pairwise_embedding_loss
from .model import AdaFaceHead, _build_backbone
from .presets import get_config


UNIMODAL_EER_CACHE_VERSION = "v10"


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
    if spec.modality == "iris" and config.iris_unimodal_backbone == "mobilenet_v3_small":
        return f"{config.checkpoint_dir}/{spec.modality}_probe_mobilenet_v3_small"
    return f"{config.checkpoint_dir}/{spec.modality}_probe"


def versioned_unimodal_checkpoint_path(config, modality: str, step: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{unimodal_checkpoint_path(config, modality)}-{timestamp}-step{step:04d}"


def latest_unimodal_checkpoint_path(config, modality: str) -> str | None:
    base_path = unimodal_checkpoint_path(config, modality)
    pattern = f"{base_path}-*-step*.index"
    candidates = tf.io.gfile.glob(pattern)
    if not candidates:
        if tf.io.gfile.exists(f"{base_path}.index"):
            return base_path
        return None

    def _sort_key(path: str) -> tuple[str, int]:
        match = re.search(r"-(\d{8}-\d{6})-step(\d+)\.index$", path)
        if not match:
            return ("", -1)
        return (match.group(1), int(match.group(2)))

    latest_index = max(candidates, key=_sort_key)
    return latest_index[: -len(".index")]


def _list_unimodal_checkpoint_paths(config, modality: str) -> list[str]:
    base_path = unimodal_checkpoint_path(config, modality)
    pattern = f"{base_path}-*-step*.index"
    candidates = tf.io.gfile.glob(pattern)
    checkpoint_paths = sorted({path[: -len(".index")] for path in candidates})
    cached_paths = [path for path in checkpoint_paths if _read_unimodal_eer(path) is not None]
    if cached_paths:
        return cached_paths
    if tf.io.gfile.exists(f"{base_path}.index"):
        checkpoint_paths.append(base_path)
    return sorted(set(checkpoint_paths))


def _unimodal_metric_path(checkpoint_path: str) -> str:
    return f"{checkpoint_path}.eer.{UNIMODAL_EER_CACHE_VERSION}.txt"


def _unimodal_best_path(config, modality: str) -> str:
    return f"{unimodal_checkpoint_path(config, modality)}.best.{UNIMODAL_EER_CACHE_VERSION}.txt"


def _read_unimodal_best_checkpoint(config, modality: str) -> str | None:
    best_path = _unimodal_best_path(config, modality)
    if not tf.io.gfile.exists(best_path):
        return None
    checkpoint_path = tf.io.read_file(best_path).numpy().decode("utf-8").strip()
    if not checkpoint_path:
        return None
    if not tf.io.gfile.exists(f"{checkpoint_path}.index"):
        return None
    if _read_unimodal_eer(checkpoint_path) is None:
        return None
    return checkpoint_path


def _write_unimodal_best_checkpoint(config, modality: str, checkpoint_path: str) -> None:
    tf.io.write_file(_unimodal_best_path(config, modality), f"{checkpoint_path}\n")


def _read_unimodal_eer(checkpoint_path: str) -> float | None:
    metric_path = _unimodal_metric_path(checkpoint_path)
    if not tf.io.gfile.exists(metric_path):
        return None
    content = tf.io.read_file(metric_path).numpy().decode("utf-8").strip()
    try:
        return float(content)
    except ValueError:
        return None


def _write_unimodal_eer(checkpoint_path: str, eer: float) -> None:
    tf.io.write_file(_unimodal_metric_path(checkpoint_path), f"{eer:.10f}\n")


def _build_unimodal_components(config, modality: str):
    spec = _spec_for(config, modality)
    inputs = keras.Input(shape=spec.input_shape, name=f"{spec.modality}_probe_input")
    feature_extractor = None
    if spec.modality == "iris" and config.iris_unimodal_backbone == "mobilenet_v3_small":
        mobilenet_size = config.iris_mobilenet_input_size
        x = layers.Resizing(
            mobilenet_size,
            mobilenet_size,
            interpolation="bilinear",
            name=f"{spec.modality}_probe_resize",
        )(inputs)
        x = layers.Concatenate(name=f"{spec.modality}_probe_rgb")([x, x, x])
        base_backbone = keras.applications.MobileNetV3Small(
            input_shape=(mobilenet_size, mobilenet_size, 3),
            alpha=config.iris_mobilenet_alpha,
            include_top=False,
            weights=config.iris_mobilenet_weights,
            pooling="avg",
            include_preprocessing=True,
        )
        base_backbone.trainable = False
        feature_extractor = base_backbone
        x = base_backbone(x)
        x = layers.Dense(spec.embedding_dim, use_bias=False, name=f"{spec.modality}_probe_base_proj")(x)
    else:
        base_backbone = _build_backbone(
            spec.input_shape,
            spec.embedding_dim,
            f"{spec.modality}_probe_backbone_base",
        )
        feature_extractor = base_backbone
        x = base_backbone(inputs)
    x = layers.BatchNormalization(name=f"{spec.modality}_probe_bn_1")(x)
    x = layers.ReLU(name=f"{spec.modality}_probe_relu")(x)
    x = layers.Dense(spec.embedding_dim, use_bias=False, name=f"{spec.modality}_probe_proj")(x)
    outputs = layers.BatchNormalization(name=f"{spec.modality}_probe_bn_2")(x)
    backbone = keras.Model(inputs, outputs, name=f"{spec.modality}_probe_backbone")
    backbone.feature_extractor = feature_extractor
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


def _unimodal_train_steps(config, modality: str) -> int:
    if modality == "iris" and config.iris_unimodal_backbone == "mobilenet_v3_small":
        return config.iris_mobilenet_train_steps
    return config.unimodal_train_steps


def _unimodal_learning_rate(config, modality: str) -> float:
    if modality == "iris" and config.iris_unimodal_backbone == "mobilenet_v3_small":
        return config.iris_mobilenet_learning_rate
    return config.unimodal_learning_rate


def _unimodal_backbone_training_flag(config, modality: str) -> bool:
    if modality == "iris" and config.iris_unimodal_backbone == "mobilenet_v3_small":
        return False
    return True


def _is_mobilenet_unimodal(config, modality: str) -> bool:
    return modality == "iris" and config.iris_unimodal_backbone == "mobilenet_v3_small"


def _set_mobilenet_finetune_trainable(backbone: keras.Model, enabled: bool) -> None:
    feature_extractor = getattr(backbone, "feature_extractor", None)
    if feature_extractor is None:
        return
    if not enabled:
        feature_extractor.trainable = False
        return

    feature_extractor.trainable = True
    trainable_seen = 0
    for layer in reversed(feature_extractor.layers):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        if layer.weights:
            trainable_seen += 1
        layer.trainable = trainable_seen <= 20


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


def _evaluate_checkpoint_embeddings(
    config,
    modality: str,
    checkpoint_path: str,
    eval_inputs: tf.Tensor,
    labels: list[int],
) -> tuple[float, float, dict[str, float]] | None:
    _, backbone, head = _build_unimodal_components(config, modality)
    _ = backbone(eval_inputs, training=False)
    sample_raw = backbone(eval_inputs, training=False)
    _ = head(sample_raw)
    checkpoint = tf.train.Checkpoint(backbone=backbone, head=head)
    try:
        checkpoint.restore(checkpoint_path).expect_partial()
    except ValueError:
        return None
    eval_raw = backbone(eval_inputs, training=False)
    eval_embeddings = tf.math.l2_normalize(eval_raw, axis=-1)
    return _evaluate_embeddings(eval_embeddings, labels)


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
    train_steps = _unimodal_train_steps(config, spec.modality)
    mobilenet_mode = _is_mobilenet_unimodal(config, spec.modality)
    head_steps = min(config.iris_mobilenet_head_steps, train_steps) if mobilenet_mode else train_steps
    optimizer = tf.keras.optimizers.Adam(learning_rate=_unimodal_learning_rate(config, spec.modality))
    backbone_training = _unimodal_backbone_training_flag(config, spec.modality)
    source = _build_training_source(config, spec.modality_index)

    final_loss = 0.0
    final_cls_loss = 0.0
    final_embedding_loss = 0.0
    final_triplet_loss = 0.0
    _set_mobilenet_finetune_trainable(backbone, enabled=False)
    for _ in range(train_steps):
        if mobilenet_mode and _ == head_steps:
            _set_mobilenet_finetune_trainable(backbone, enabled=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.iris_mobilenet_finetune_learning_rate)
        batch_inputs, labels = next(source)
        with tf.GradientTape() as tape:
            raw_embeddings = backbone(batch_inputs, training=backbone_training)
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

    save_path = versioned_unimodal_checkpoint_path(
        config,
        spec.modality,
        train_steps,
    )
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
        "train_steps": train_steps,
    }


def evaluate_unimodal_probe(config, modality: str) -> dict[str, object]:
    spec = _spec_for(config, modality)
    records = collect_evaluation_records(config)
    if not records:
        raise RuntimeError("No evaluation data available for unimodal probe.")

    inputs, labels_tensor, _ = _records_to_batch(records[:64], config)
    eval_inputs = inputs[spec.modality_index]
    best_checkpoint_path = _read_unimodal_best_checkpoint(config, spec.modality)
    labels = labels_tensor.numpy().tolist()
    if best_checkpoint_path is not None:
        evaluation = _evaluate_checkpoint_embeddings(
            config,
            spec.modality,
            best_checkpoint_path,
            eval_inputs,
            labels,
        )
        if evaluation is not None:
            genuine_mean, impostor_mean, metrics = evaluation
            _write_unimodal_eer(best_checkpoint_path, metrics["eer"])
            return {
                "modality": spec.modality,
                "checkpoint_path": best_checkpoint_path,
                "genuine_mean": genuine_mean,
                "impostor_mean": impostor_mean,
                "metrics": metrics,
            }

    latest_checkpoint_path = latest_unimodal_checkpoint_path(config, spec.modality)
    checkpoint_paths = _list_unimodal_checkpoint_paths(config, spec.modality)
    if not checkpoint_paths and latest_checkpoint_path is None:
        raise RuntimeError(f"Unimodal checkpoint not found for modality: {spec.modality}")
    has_cached_metrics = any(_read_unimodal_eer(path) is not None for path in checkpoint_paths)
    best_result = None
    if latest_checkpoint_path is not None:
        evaluation = _evaluate_checkpoint_embeddings(
            config,
            spec.modality,
            latest_checkpoint_path,
            eval_inputs,
            labels,
        )
        if evaluation is not None:
            genuine_mean, impostor_mean, metrics = evaluation
            _write_unimodal_eer(latest_checkpoint_path, metrics["eer"])
            best_result = {
                "checkpoint_path": latest_checkpoint_path,
                "genuine_mean": genuine_mean,
                "impostor_mean": impostor_mean,
                "metrics": metrics,
            }
            if not has_cached_metrics:
                _write_unimodal_best_checkpoint(config, spec.modality, latest_checkpoint_path)
                return {
                    "modality": spec.modality,
                    "checkpoint_path": latest_checkpoint_path,
                    "genuine_mean": genuine_mean,
                    "impostor_mean": impostor_mean,
                    "metrics": metrics,
                }
    for checkpoint_path in checkpoint_paths:
        if best_result is not None and checkpoint_path == best_result["checkpoint_path"]:
            continue
        cached_eer = _read_unimodal_eer(checkpoint_path)
        if cached_eer is not None and best_result is not None and cached_eer >= best_result["metrics"]["eer"]:
            continue
        evaluation = _evaluate_checkpoint_embeddings(
            config,
            spec.modality,
            checkpoint_path,
            eval_inputs,
            labels,
        )
        if evaluation is None:
            continue
        genuine_mean, impostor_mean, metrics = evaluation
        _write_unimodal_eer(checkpoint_path, metrics["eer"])
        if best_result is None or metrics["eer"] < best_result["metrics"]["eer"]:
            best_result = {
                "checkpoint_path": checkpoint_path,
                "genuine_mean": genuine_mean,
                "impostor_mean": impostor_mean,
                "metrics": metrics,
            }

    if best_result is None:
        raise RuntimeError(f"No compatible unimodal checkpoints found for modality: {spec.modality}")
    _write_unimodal_best_checkpoint(config, spec.modality, best_result["checkpoint_path"])
    return {
        "modality": spec.modality,
        "checkpoint_path": best_result["checkpoint_path"],
        "genuine_mean": best_result["genuine_mean"],
        "impostor_mean": best_result["impostor_mean"],
        "metrics": best_result["metrics"],
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
