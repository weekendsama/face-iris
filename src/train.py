from __future__ import annotations

from datetime import datetime
import re

import tensorflow as tf

from .config import DEFAULT_CONFIG
from .data import build_training_batch_iterator, build_training_dataset
from .losses import compute_total_loss
from .model import build_model
from .presets import get_config
from .probe_unimodal import train_unimodal_probe


def make_dummy_batch(config, step: int) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    face = tf.random.uniform(
        (config.batch_size, *config.face_input_shape),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )
    iris = tf.random.uniform(
        (config.batch_size, *config.iris_input_shape),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )

    # Use a staged label schedule so DSH is skipped early, then enabled once
    # every seen class reaches the minimum sample count in the feature bank.
    if step == 0:
        label_pattern = tf.constant([0, 1, 0, 2], dtype=tf.int32)
    elif step == 1:
        label_pattern = tf.constant([1, 2, 1, 2], dtype=tf.int32)
    else:
        label_pattern = tf.constant([0, 1, 2, 0], dtype=tf.int32)
    repeats = (config.batch_size + int(label_pattern.shape[0]) - 1) // int(label_pattern.shape[0])
    labels = tf.tile(label_pattern, [repeats])[: config.batch_size]
    return (face, iris), labels


def train_step(model, optimizer, inputs, labels, config, step: int):
    dsh_refreshed = False
    if config.dsh_refresh_interval > 0 and step % config.dsh_refresh_interval == 0:
        dsh_refreshed = model.refresh_hash_projections(inputs, labels)

    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        total_loss, metrics = compute_total_loss(outputs, labels, config)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metrics["dsh_refreshed"] = dsh_refreshed
    return metrics


def checkpoint_path(config) -> str:
    return f"{config.checkpoint_dir}/{config.checkpoint_name}"


def versioned_checkpoint_path(config, step: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{checkpoint_path(config)}-{timestamp}-step{step:04d}"


def latest_checkpoint_path(config) -> str | None:
    pattern = f"{checkpoint_path(config)}-*-step*.index"
    candidates = tf.io.gfile.glob(pattern)
    if not candidates:
        legacy_path = checkpoint_path(config)
        if tf.io.gfile.exists(f"{legacy_path}.index"):
            return legacy_path
        return None

    def _sort_key(path: str) -> tuple[str, int]:
        match = re.search(r"-(\d{8}-\d{6})-step(\d+)\.index$", path)
        if not match:
            return ("", -1)
        return (match.group(1), int(match.group(2)))

    latest_index = max(candidates, key=_sort_key)
    return latest_index[: -len(".index")]


def main() -> None:
    config, preset_name = get_config()
    if config.training_mode in {"face_only", "iris_only"}:
        modality = "face" if config.training_mode == "face_only" else "iris"
        result = train_unimodal_probe(config, modality)
        print("preset:", preset_name)
        print("training_mode:", config.training_mode)
        print("train_final_loss:", f"{result['train_final_loss']:.4f}")
        print("train_final_cls_loss:", f"{result['train_final_cls_loss']:.4f}")
        print("train_final_embedding_loss:", f"{result['train_final_embedding_loss']:.4f}")
        print("train_final_triplet_loss:", f"{result['train_final_triplet_loss']:.4f}")
        print("checkpoint_saved:", result["checkpoint_path"])
        return

    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_model(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    train_batch_iterator = build_training_batch_iterator(config)
    train_dataset = None if train_batch_iterator is not None else build_training_dataset(config)

    if train_batch_iterator is not None:
        step_source = train_batch_iterator
    elif train_dataset is None:
        step_source = (make_dummy_batch(config, step) for step in range(config.train_steps))
    else:
        step_source = iter(train_dataset.repeat())

    print("preset:", preset_name)
    print("pairing_protocol:", config.pairing_mode)
    for step in range(config.train_steps):
        inputs, labels = next(step_source)
        metrics = train_step(model, optimizer, inputs, labels, config, step)
        print(
            "step={step} lr={lr:.6f} labels={labels} total={total:.4f} cls={cls:.4f} pair={pair:.4f} dsh={dsh}".format(
                step=step + 1,
                lr=float(optimizer.learning_rate.numpy()),
                labels=labels.numpy().tolist(),
                total=float(metrics["total_loss"].numpy()),
                cls=float(metrics["classification_loss"].numpy()),
                pair=float(metrics["pairwise_loss"].numpy()),
                dsh="refreshed" if metrics["dsh_refreshed"] else "skipped",
            )
        )

    tf.io.gfile.makedirs(config.checkpoint_dir)
    save_path = versioned_checkpoint_path(config, config.train_steps)
    model.save_weights(save_path)
    print("checkpoint_saved:", save_path)


if __name__ == "__main__":
    main()
