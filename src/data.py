from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import tensorflow as tf

from .config import ModelConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SampleRecord:
    face_path: Path
    iris_path: Path
    label: int


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_images_recursive(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_multimodal_samples(dataset_root: str | Path) -> list[SampleRecord]:
    """
    Expected layout:
    data/
      train/
        person_001/
          face/
          iris/
        person_002/
          face/
          iris/
    """
    root = Path(dataset_root)
    train_root = root / "train"
    if not train_root.exists():
        return []

    records: list[SampleRecord] = []
    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    for label, class_dir in enumerate(class_dirs):
        face_images = _list_images(class_dir / "face")
        iris_images = _list_images(class_dir / "iris")
        pair_count = min(len(face_images), len(iris_images))
        for idx in range(pair_count):
            records.append(
                SampleRecord(
                    face_path=face_images[idx],
                    iris_path=iris_images[idx],
                    label=label,
                )
            )
    return records


def _discover_lfw_identities(lfw_root: str | Path) -> list[tuple[str, list[Path]]]:
    root = Path(lfw_root)
    base = root / "lfw_funneled" if (root / "lfw_funneled").exists() else root
    if not base.exists():
        return []

    identities: list[tuple[str, list[Path]]] = []
    for class_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        images = _list_images(class_dir)
        if images:
            identities.append((class_dir.name, images))
    return identities


def _discover_casia_identities(casia_root: str | Path) -> list[tuple[str, list[Path]]]:
    root = Path(casia_root)
    if not root.exists():
        return []

    identities: list[tuple[str, list[Path]]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = _list_images_recursive(class_dir)
        if images:
            identities.append((class_dir.name, images))
    return identities


def discover_public_multimodal_samples(
    lfw_root: str | Path,
    casia_root: str | Path,
) -> list[SampleRecord]:
    """
    Build pseudo-paired samples by aligning LFW and CASIA identities by sorted index.
    This is the primary public-dataset manual-pairing protocol for this project.
    """
    face_identities = _discover_lfw_identities(lfw_root)
    iris_identities = _discover_casia_identities(casia_root)
    pairable_identities = min(len(face_identities), len(iris_identities))

    records: list[SampleRecord] = []
    for label in range(pairable_identities):
        _, face_images = face_identities[label]
        _, iris_images = iris_identities[label]
        pair_count = min(len(face_images), len(iris_images))
        for idx in range(pair_count):
            records.append(
                SampleRecord(
                    face_path=face_images[idx],
                    iris_path=iris_images[idx],
                    label=label,
                )
            )
    return records


def summarize_dataset_structure(dataset_root: str | Path) -> dict[str, object]:
    root = Path(dataset_root)
    train_root = root / "train"
    summary: dict[str, object] = {
        "dataset_root": str(root),
        "train_root_exists": train_root.exists(),
        "classes": [],
        "total_pairs": 0,
    }
    if not train_root.exists():
        return summary

    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    classes: list[dict[str, object]] = []
    total_pairs = 0
    for class_dir in class_dirs:
        face_images = _list_images(class_dir / "face")
        iris_images = _list_images(class_dir / "iris")
        pair_count = min(len(face_images), len(iris_images))
        total_pairs += pair_count
        classes.append(
            {
                "name": class_dir.name,
                "face_count": len(face_images),
                "iris_count": len(iris_images),
                "pair_count": pair_count,
                "face_dir_exists": (class_dir / "face").exists(),
                "iris_dir_exists": (class_dir / "iris").exists(),
            }
        )

    summary["classes"] = classes
    summary["total_pairs"] = total_pairs
    return summary


def summarize_public_dataset_structure(
    lfw_root: str | Path,
    casia_root: str | Path,
) -> dict[str, object]:
    face_identities = _discover_lfw_identities(lfw_root)
    iris_identities = _discover_casia_identities(casia_root)
    pairable_identities = min(len(face_identities), len(iris_identities))

    classes: list[dict[str, object]] = []
    total_pairs = 0
    for idx in range(pairable_identities):
        face_name, face_images = face_identities[idx]
        iris_name, iris_images = iris_identities[idx]
        pair_count = min(len(face_images), len(iris_images))
        total_pairs += pair_count
        classes.append(
            {
                "label": idx,
                "face_identity": face_name,
                "iris_identity": iris_name,
                "face_count": len(face_images),
                "iris_count": len(iris_images),
                "pair_count": pair_count,
            }
        )

    return {
        "lfw_root": str(lfw_root),
        "casia_root": str(casia_root),
        "lfw_identities": len(face_identities),
        "casia_identities": len(iris_identities),
        "pairable_identities": pairable_identities,
        "total_pairs": total_pairs,
        "classes": classes,
        "pairing_mode": "pseudo_indexed",
    }


def _load_image(path: tf.Tensor, image_shape: tuple[int, int, int]) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    channels = image_shape[-1]
    image = tf.image.decode_image(
        image_bytes,
        channels=channels,
        expand_animations=False,
    )
    image = tf.image.resize(image, image_shape[:2])
    return tf.cast(image, tf.float32)


def _split_records_by_identity(
    records: list[SampleRecord],
    train_identity_fraction: float,
    split: str,
) -> list[SampleRecord]:
    grouped: dict[int, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    labels = sorted(grouped)
    split_index = max(1, min(len(labels) - 1, int(len(labels) * train_identity_fraction)))
    if split == "train":
        selected_labels = set(labels[:split_index])
    elif split == "eval":
        selected_labels = set(labels[split_index:])
    else:
        selected_labels = set(labels)

    output: list[SampleRecord] = []
    for label in labels:
        if label in selected_labels:
            output.extend(grouped[label])
    return output


def _resolve_training_records(config: ModelConfig, split: str = "train") -> list[SampleRecord]:
    if config.pairing_mode == "public_manual_indexed" and config.use_public_dataset_roots:
        records = discover_public_multimodal_samples(
            config.lfw_root,
            config.casia_iris_root,
        )
        return _split_records_by_identity(records, config.train_identity_fraction, split)
    records = discover_multimodal_samples(config.dataset_root)
    return _split_records_by_identity(records, config.train_identity_fraction, split)


def collect_evaluation_records(config: ModelConfig) -> list[SampleRecord]:
    def _select(records: list[SampleRecord]) -> list[SampleRecord]:
        grouped: dict[int, list[SampleRecord]] = {}
        for record in records:
            grouped.setdefault(record.label, []).append(record)

        selected: list[SampleRecord] = []
        target_samples = max(2, config.eval_samples_per_identity)
        while target_samples >= 2:
            eligible_labels = [
                label
                for label in sorted(grouped)
                if len(grouped[label]) >= target_samples
            ]
            if len(eligible_labels) >= 2:
                for label in eligible_labels[: config.eval_identities]:
                    selected.extend(grouped[label][:target_samples])
                return selected
            target_samples -= 1
        return selected

    eval_records = _select(_resolve_training_records(config, split="eval"))
    eval_identity_count = len({record.label for record in eval_records})
    minimum_identity_target = min(config.eval_identities, 4)
    if eval_identity_count >= minimum_identity_target:
        return eval_records
    return _select(_resolve_training_records(config, split="all"))


def build_training_dataset(config: ModelConfig) -> tf.data.Dataset | None:
    records = _resolve_training_records(config, split="train")
    if not records:
        return None

    face_paths = [str(item.face_path) for item in records]
    iris_paths = [str(item.iris_path) for item in records]
    labels = [item.label for item in records]

    dataset = tf.data.Dataset.from_tensor_slices((face_paths, iris_paths, labels))
    dataset = dataset.shuffle(
        buffer_size=max(len(records), config.batch_size * 2),
        seed=config.data_seed,
        reshuffle_each_iteration=True,
    )

    def _map_fn(face_path: tf.Tensor, iris_path: tf.Tensor, label: tf.Tensor):
        face = _load_image(face_path, config.face_input_shape)
        iris = _load_image(iris_path, config.iris_input_shape)
        return (face, iris), tf.cast(label, tf.int32)

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_training_batch_iterator(config: ModelConfig):
    records = _resolve_training_records(config, split="train")
    if not records:
        return None

    grouped: dict[int, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    label_order = sorted(grouped)
    if not label_order:
        return None

    rng = random.Random(config.data_seed)
    positions = {label: 0 for label in label_order}

    def _next_record(label: int) -> SampleRecord:
        items = grouped[label]
        if positions[label] == 0:
            rng.shuffle(items)
        record = items[positions[label]]
        positions[label] = (positions[label] + 1) % len(items)
        return record

    def _iterator():
        identities_per_batch = max(
            1,
            min(config.sampler_identities_per_batch, config.batch_size, len(label_order)),
        )
        samples_per_identity = max(1, config.batch_size // identities_per_batch)
        eligible_labels = [
            label for label in label_order if len(grouped[label]) >= samples_per_identity
        ]
        sampling_pool = eligible_labels if len(eligible_labels) >= identities_per_batch else label_order
        label_queue = sampling_pool[:]
        rng.shuffle(label_queue)
        queue_index = 0

        while True:
            if queue_index + identities_per_batch > len(label_queue):
                rng.shuffle(label_queue)
                queue_index = 0
            chosen_labels = label_queue[queue_index : queue_index + identities_per_batch]
            queue_index += identities_per_batch
            batch_records: list[SampleRecord] = []

            for label in chosen_labels:
                for _ in range(samples_per_identity):
                    batch_records.append(_next_record(label))

            while len(batch_records) < config.batch_size:
                batch_records.append(_next_record(chosen_labels[len(batch_records) % len(chosen_labels)]))

            batch_records = batch_records[: config.batch_size]
            rng.shuffle(batch_records)

            face_batch = tf.stack(
                [_load_image(tf.constant(str(item.face_path)), config.face_input_shape) for item in batch_records],
                axis=0,
            )
            iris_batch = tf.stack(
                [_load_image(tf.constant(str(item.iris_path)), config.iris_input_shape) for item in batch_records],
                axis=0,
            )
            labels = tf.constant([item.label for item in batch_records], dtype=tf.int32)
            yield (face_batch, iris_batch), labels

    return _iterator()


def build_record_batch_iterator(config: ModelConfig, split: str = "eval"):
    records = _resolve_training_records(config, split=split)
    if not records:
        return None

    grouped: dict[int, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    label_order = sorted(grouped)
    if not label_order:
        return None

    rng = random.Random(config.data_seed)
    positions = {label: 0 for label in label_order}

    def _next_record(label: int) -> SampleRecord:
        items = grouped[label]
        if positions[label] == 0:
            rng.shuffle(items)
        record = items[positions[label]]
        positions[label] = (positions[label] + 1) % len(items)
        return record

    def _iterator():
        identities_per_batch = max(
            1,
            min(config.sampler_identities_per_batch, config.batch_size, len(label_order)),
        )
        samples_per_identity = max(1, config.batch_size // identities_per_batch)
        eligible_labels = [
            label for label in label_order if len(grouped[label]) >= samples_per_identity
        ]
        sampling_pool = eligible_labels if len(eligible_labels) >= identities_per_batch else label_order
        label_queue = sampling_pool[:]
        rng.shuffle(label_queue)
        queue_index = 0

        while True:
            if queue_index + identities_per_batch > len(label_queue):
                rng.shuffle(label_queue)
                queue_index = 0
            chosen_labels = label_queue[queue_index : queue_index + identities_per_batch]
            queue_index += identities_per_batch
            batch_records: list[SampleRecord] = []

            for label in chosen_labels:
                for _ in range(samples_per_identity):
                    batch_records.append(_next_record(label))

            while len(batch_records) < config.batch_size:
                batch_records.append(_next_record(chosen_labels[len(batch_records) % len(chosen_labels)]))

            batch_records = batch_records[: config.batch_size]
            rng.shuffle(batch_records)
            yield batch_records

    return _iterator()
