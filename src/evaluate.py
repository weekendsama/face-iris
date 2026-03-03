from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf

from .config import DEFAULT_CONFIG
from .data import build_record_batch_iterator, build_training_batch_iterator, build_training_dataset, collect_evaluation_records
from .protection import (
    blockwise_hamming_similarity,
    generate_user_key,
    hamming_similarity,
    protect_template,
)
from .train import make_dummy_batch, train_step
from .train import latest_checkpoint_path
from .model import build_model
from .presets import get_config
from .probe_unimodal import evaluate_unimodal_probe


def _build_eval_source(config):
    train_batch_iterator = build_training_batch_iterator(config)
    train_dataset = None if train_batch_iterator is not None else build_training_dataset(config)

    if train_batch_iterator is not None:
        return train_batch_iterator, "grouped_dataset"
    if train_dataset is not None:
        return iter(train_dataset.repeat()), "tfdata_dataset"
    return (make_dummy_batch(config, step) for step in range(config.eval_batches)), "dummy"


def _compute_pair_metrics(
    templates: tf.Tensor,
    labels: tf.Tensor,
    block_size: int,
) -> dict[str, object]:
    num_samples = int(labels.shape[0])
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    genuine_block_scores: list[float] = []
    impostor_block_scores: list[float] = []

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            global_score = float(hamming_similarity(templates[i], templates[j]).numpy())
            block_score = float(
                blockwise_hamming_similarity(
                    templates[i],
                    templates[j],
                    block_size=block_size,
                ).numpy()
            )
            if int(labels[i]) == int(labels[j]):
                genuine_scores.append(global_score)
                genuine_block_scores.append(block_score)
            else:
                impostor_scores.append(global_score)
                impostor_block_scores.append(block_score)

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "genuine_block_scores": genuine_block_scores,
        "impostor_block_scores": impostor_block_scores,
    }


def _compute_verification_metrics(
    enroll_codes: list[tf.Tensor],
    enroll_labels: list[int],
    probe_codes: list[tf.Tensor],
    probe_labels: list[int],
    block_size: int,
    num_bits: int,
) -> dict[str, object]:
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    genuine_block_scores: list[float] = []
    impostor_block_scores: list[float] = []

    protected_enroll_templates: list[tf.Tensor] = []
    for i in range(len(enroll_codes)):
        perm, mask = generate_user_key(num_bits, seed=enroll_labels[i] + 1000)
        protected_enroll_templates.append(protect_template(enroll_codes[i], perm, mask))

    for i in range(len(protected_enroll_templates)):
        claim_label = enroll_labels[i]
        perm, mask = generate_user_key(num_bits, seed=claim_label + 1000)
        for j in range(len(probe_codes)):
            protected_probe = protect_template(probe_codes[j], perm, mask)
            global_score = float(hamming_similarity(protected_enroll_templates[i], protected_probe).numpy())
            block_score = float(
                blockwise_hamming_similarity(
                    protected_enroll_templates[i],
                    protected_probe,
                    block_size=block_size,
                ).numpy()
            )
            if enroll_labels[i] == probe_labels[j]:
                genuine_scores.append(global_score)
                genuine_block_scores.append(block_score)
            else:
                impostor_scores.append(global_score)
                impostor_block_scores.append(block_score)

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "genuine_block_scores": genuine_block_scores,
        "impostor_block_scores": impostor_block_scores,
    }


def _compute_code_verification_metrics(
    enroll_codes: list[tf.Tensor],
    enroll_labels: list[int],
    probe_codes: list[tf.Tensor],
    probe_labels: list[int],
    block_size: int,
) -> dict[str, object]:
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    genuine_block_scores: list[float] = []
    impostor_block_scores: list[float] = []

    for i in range(len(enroll_codes)):
        for j in range(len(probe_codes)):
            global_score = float(hamming_similarity(enroll_codes[i], probe_codes[j]).numpy())
            block_score = float(
                blockwise_hamming_similarity(
                    enroll_codes[i],
                    probe_codes[j],
                    block_size=block_size,
                ).numpy()
            )
            if enroll_labels[i] == probe_labels[j]:
                genuine_scores.append(global_score)
                genuine_block_scores.append(block_score)
            else:
                impostor_scores.append(global_score)
                impostor_block_scores.append(block_score)

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "genuine_block_scores": genuine_block_scores,
        "impostor_block_scores": impostor_block_scores,
    }


def _soft_similarity(code_a: tf.Tensor, code_b: tf.Tensor) -> float:
    return float((1.0 - tf.reduce_mean(tf.abs(code_a - code_b))).numpy())


def _compute_soft_verification_metrics(
    enroll_codes: list[tf.Tensor],
    enroll_labels: list[int],
    probe_codes: list[tf.Tensor],
    probe_labels: list[int],
) -> dict[str, object]:
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for i in range(len(enroll_codes)):
        for j in range(len(probe_codes)):
            score = _soft_similarity(enroll_codes[i], probe_codes[j])
            if enroll_labels[i] == probe_labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
    }


def _compute_embedding_verification_metrics(
    enroll_embeddings: list[tf.Tensor],
    enroll_labels: list[int],
    probe_embeddings: list[tf.Tensor],
    probe_labels: list[int],
) -> dict[str, object]:
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for i in range(len(enroll_embeddings)):
        for j in range(len(probe_embeddings)):
            score = float(tf.tensordot(enroll_embeddings[i], probe_embeddings[j], axes=1).numpy())
            if enroll_labels[i] == probe_labels[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
    }


def _compute_far_frr_eer(
    genuine_scores: list[float],
    impostor_scores: list[float],
) -> dict[str, float]:
    if not genuine_scores or not impostor_scores:
        return {
            "threshold": math.nan,
            "far": math.nan,
            "frr": math.nan,
            "eer": math.nan,
        }

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


def _compute_curve_points(
    genuine_scores: list[float],
    impostor_scores: list[float],
) -> list[dict[str, float]]:
    if not genuine_scores or not impostor_scores:
        return []

    thresholds = sorted(set(genuine_scores + impostor_scores))
    points = []
    for threshold in thresholds:
        far = sum(score >= threshold for score in impostor_scores) / len(impostor_scores)
        frr = sum(score < threshold for score in genuine_scores) / len(genuine_scores)
        points.append(
            {
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "tar": 1.0 - frr,
                "trr": 1.0 - far,
            }
        )
    return points


def _mean_score(scores: list[float]) -> float:
    if not scores:
        return math.nan
    return sum(scores) / len(scores)


def _export_curve_csv(path: str, points: list[dict[str, float]]) -> None:
    lines = ["threshold,far,frr,tar,trr"]
    for point in points:
        lines.append(
            "{threshold:.6f},{far:.6f},{frr:.6f},{tar:.6f},{trr:.6f}".format(
                **point,
            )
        )
    tf.io.gfile.makedirs(str(Path(path).parent))
    tf.io.write_file(path, "\n".join(lines) + "\n")


def _apply_bit_flip_noise(template: tf.Tensor, flip_prob: float, seed: int) -> tf.Tensor:
    if flip_prob <= 0.0:
        return template
    seed_tensor = tf.constant([seed, seed ^ 0x13579B], dtype=tf.int32)
    flips = tf.random.stateless_uniform(
        shape=tf.shape(template),
        seed=seed_tensor,
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    ) < flip_prob
    flips = tf.cast(flips, tf.int32)
    return tf.bitwise.bitwise_xor(template, flips)


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


def main() -> None:
    config, preset_name = get_config()
    if config.training_mode in {"face_only", "iris_only"}:
        modality = "face" if config.training_mode == "face_only" else "iris"
        result = evaluate_unimodal_probe(config, modality)
        metrics = result["metrics"]
        print("preset:", preset_name)
        print("training_mode:", config.training_mode)
        print("checkpoint_path:", result["checkpoint_path"])
        print(
            "embedding genuine_mean={genuine:.4f} impostor_mean={impostor:.4f} "
            "threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
                genuine=result["genuine_mean"],
                impostor=result["impostor_mean"],
                **metrics,
            )
        )
        return

    tf.keras.utils.set_random_seed(config.random_seed)
    model = build_model(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    step_source, source_name = _build_eval_source(config)
    record_source = build_record_batch_iterator(config)

    if record_source is not None:
        records = next(record_source)
        first_inputs, first_labels, first_record_ids = _records_to_batch(records, config)
    else:
        first_inputs, first_labels = next(step_source)
        first_record_ids = [
            f"fallback|warmup|{idx}|{int(first_labels[idx].numpy())}"
            for idx in range(int(first_labels.shape[0]))
        ]

    _ = model(first_inputs, training=False)
    ckpt_path = latest_checkpoint_path(config)
    warmup_source = step_source
    warmup_losses = []
    checkpoint_loaded = False
    if ckpt_path is not None:
        model.load_weights(ckpt_path)
        checkpoint_loaded = True
    else:
        metrics = train_step(model, optimizer, first_inputs, first_labels, config, 0)
        warmup_losses.append(float(metrics["total_loss"].numpy()))
        for step in range(1, config.eval_warmup_steps):
            inputs, labels = next(warmup_source)
            metrics = train_step(model, optimizer, inputs, labels, config, step)
            warmup_losses.append(float(metrics["total_loss"].numpy()))

    protected_templates = []
    raw_codes = []
    fused_embeddings = []
    soft_templates = []
    all_labels = []
    all_record_ids = []
    eval_records = collect_evaluation_records(config)
    if eval_records:
        for start in range(0, len(eval_records), config.batch_size):
            records = eval_records[start : start + config.batch_size]
            inputs, labels, record_ids = _records_to_batch(records, config)
            outputs = model(inputs, training=False)
            shared_code_soft = outputs["shared_code_soft"]
            shared_codes = outputs["shared_code"]
            for idx in range(int(labels.shape[0])):
                label = int(labels[idx].numpy())
                perm, mask = generate_user_key(config.shared_hash_bits, seed=label + 1000)
                protected = protect_template(shared_codes[idx], perm, mask)
                protected_templates.append(protected)
                raw_codes.append(shared_codes[idx])
                fused_embeddings.append(outputs["fused_embedding"][idx])
                soft_templates.append(shared_code_soft[idx])
                all_labels.append(label)
                all_record_ids.append(record_ids[idx])
        source_name = f"{source_name}+identity_pairs"
    else:
        for step in range(config.eval_batches):
            if step == 0:
                inputs, labels, record_ids = first_inputs, first_labels, first_record_ids
            elif record_source is not None:
                records = next(record_source)
                inputs, labels, record_ids = _records_to_batch(records, config)
            else:
                inputs, labels = next(step_source)
                record_ids = [f"fallback|{step}|{idx}|{int(labels[idx].numpy())}" for idx in range(int(labels.shape[0]))]
            outputs = model(inputs, training=False)
            shared_code_soft = outputs["shared_code_soft"]
            shared_codes = outputs["shared_code"]

            for idx in range(int(labels.shape[0])):
                label = int(labels[idx].numpy())
                perm, mask = generate_user_key(config.shared_hash_bits, seed=label + 1000)
                protected_templates.append(protect_template(shared_codes[idx], perm, mask))
                raw_codes.append(shared_codes[idx])
                fused_embeddings.append(outputs["fused_embedding"][idx])
                soft_templates.append(shared_code_soft[idx])
                all_labels.append(label)
                all_record_ids.append(record_ids[idx])

    templates = tf.stack(protected_templates, axis=0)
    code_tensor = tf.stack(raw_codes, axis=0)
    fused_embedding_tensor = tf.stack(fused_embeddings, axis=0)
    soft_code_tensor = tf.stack(soft_templates, axis=0)
    labels = tf.constant(all_labels, dtype=tf.int32)
    pair_metrics = _compute_pair_metrics(templates, labels, config.block_size)

    grouped_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(all_labels):
        grouped_indices.setdefault(label, []).append(idx)

    enroll_templates = []
    enroll_codes = []
    enroll_embeddings = []
    enroll_labels = []
    probe_templates = []
    probe_codes = []
    probe_embeddings = []
    probe_labels = []
    enroll_soft_codes = []
    probe_soft_codes = []
    for label in sorted(grouped_indices):
        indices = grouped_indices[label]
        split_point = max(1, len(indices) // 2)
        enroll_idx = indices[:split_point]
        probe_idx = indices[split_point:]
        if not probe_idx:
            continue
        for idx in enroll_idx:
            enroll_templates.append(templates[idx])
            enroll_codes.append(code_tensor[idx])
            enroll_embeddings.append(fused_embedding_tensor[idx])
            enroll_soft_codes.append(soft_code_tensor[idx])
            enroll_labels.append(label)
        for idx in probe_idx:
            noisy_probe_code = _apply_bit_flip_noise(
                tf.cast(code_tensor[idx], tf.int32),
                config.eval_probe_bit_flip_prob,
                seed=label * 10000 + idx,
            )
            probe_templates.append(
                _apply_bit_flip_noise(
                    templates[idx],
                    config.eval_probe_bit_flip_prob,
                    seed=label * 10000 + idx,
                )
            )
            probe_codes.append(tf.cast(noisy_probe_code, tf.float32))
            probe_embeddings.append(fused_embedding_tensor[idx])
            probe_soft_codes.append(soft_code_tensor[idx])
            probe_labels.append(label)

    verification_metrics = _compute_verification_metrics(
        enroll_codes,
        enroll_labels,
        probe_codes,
        probe_labels,
        config.block_size,
        config.shared_hash_bits,
    )
    raw_code_metrics = _compute_code_verification_metrics(
        enroll_codes,
        enroll_labels,
        probe_codes,
        probe_labels,
        config.block_size,
    )
    filtered_genuine_scores = verification_metrics["genuine_scores"]
    filtered_impostor_scores = verification_metrics["impostor_scores"]
    filtered_genuine_block_scores = verification_metrics["genuine_block_scores"]
    filtered_impostor_block_scores = verification_metrics["impostor_block_scores"]
    raw_genuine_scores = raw_code_metrics["genuine_scores"]
    raw_impostor_scores = raw_code_metrics["impostor_scores"]
    raw_genuine_block_scores = raw_code_metrics["genuine_block_scores"]
    raw_impostor_block_scores = raw_code_metrics["impostor_block_scores"]
    soft_verification_metrics = _compute_soft_verification_metrics(
        enroll_soft_codes,
        enroll_labels,
        probe_soft_codes,
        probe_labels,
    )
    embedding_verification_metrics = _compute_embedding_verification_metrics(
        enroll_embeddings,
        enroll_labels,
        probe_embeddings,
        probe_labels,
    )
    soft_genuine_scores = soft_verification_metrics["genuine_scores"]
    soft_impostor_scores = soft_verification_metrics["impostor_scores"]
    embedding_genuine_scores = embedding_verification_metrics["genuine_scores"]
    embedding_impostor_scores = embedding_verification_metrics["impostor_scores"]

    raw_global_metrics = _compute_far_frr_eer(
        raw_genuine_scores,
        raw_impostor_scores,
    )
    raw_block_metrics = _compute_far_frr_eer(
        raw_genuine_block_scores,
        raw_impostor_block_scores,
    )
    global_metrics = _compute_far_frr_eer(
        filtered_genuine_scores,
        filtered_impostor_scores,
    )
    block_metrics = _compute_far_frr_eer(
        filtered_genuine_block_scores,
        filtered_impostor_block_scores,
    )
    soft_metrics = _compute_far_frr_eer(
        soft_genuine_scores,
        soft_impostor_scores,
    )
    embedding_metrics = _compute_far_frr_eer(
        embedding_genuine_scores,
        embedding_impostor_scores,
    )
    raw_global_curve = _compute_curve_points(
        raw_genuine_scores,
        raw_impostor_scores,
    )
    raw_block_curve = _compute_curve_points(
        raw_genuine_block_scores,
        raw_impostor_block_scores,
    )
    global_curve = _compute_curve_points(
        filtered_genuine_scores,
        filtered_impostor_scores,
    )
    block_curve = _compute_curve_points(
        filtered_genuine_block_scores,
        filtered_impostor_block_scores,
    )
    soft_curve = _compute_curve_points(
        soft_genuine_scores,
        soft_impostor_scores,
    )

    run_stem = Path(ckpt_path).name if ckpt_path is not None else "warmup_only"
    raw_global_curve_path = str(Path(config.eval_report_dir) / f"{run_stem}_raw_global.csv")
    raw_block_curve_path = str(Path(config.eval_report_dir) / f"{run_stem}_raw_block.csv")
    global_curve_path = str(Path(config.eval_report_dir) / f"{run_stem}_global.csv")
    block_curve_path = str(Path(config.eval_report_dir) / f"{run_stem}_block.csv")
    soft_curve_path = str(Path(config.eval_report_dir) / f"{run_stem}_soft.csv")
    _export_curve_csv(raw_global_curve_path, raw_global_curve)
    _export_curve_csv(raw_block_curve_path, raw_block_curve)
    _export_curve_csv(global_curve_path, global_curve)
    _export_curve_csv(block_curve_path, block_curve)
    _export_curve_csv(soft_curve_path, soft_curve)

    print("preset:", preset_name)
    print("pairing_protocol:", config.pairing_mode)
    print("source:", source_name)
    print("checkpoint_loaded:", "yes" if checkpoint_loaded else "no")
    print("checkpoint_path:", ckpt_path if ckpt_path is not None else "none")
    print("probe_bit_flip_prob:", f"{config.eval_probe_bit_flip_prob:.4f}")
    if not checkpoint_loaded:
        print("warmup_steps:", config.eval_warmup_steps)
    if warmup_losses:
        print("warmup_last_loss:", f"{warmup_losses[-1]:.4f}")
    print("samples:", int(labels.shape[0]))
    print("raw_genuine_pairs:", len(pair_metrics["genuine_scores"]))
    print("raw_impostor_pairs:", len(pair_metrics["impostor_scores"]))
    print("verification_enroll:", len(enroll_templates))
    print("verification_probe:", len(probe_templates))
    print("verification_genuine_pairs:", len(filtered_genuine_scores))
    print("verification_impostor_pairs:", len(filtered_impostor_scores))
    print(
        "raw_binary_mean genuine={genuine:.4f} impostor={impostor:.4f}".format(
            genuine=_mean_score(raw_genuine_scores),
            impostor=_mean_score(raw_impostor_scores),
        )
    )
    print(
        "raw_binary_global threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            **raw_global_metrics,
        )
    )
    print(
        "raw_binary_block threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            **raw_block_metrics,
        )
    )
    print(
        "strict_protected_mean genuine={genuine:.4f} impostor={impostor:.4f}".format(
            genuine=_mean_score(filtered_genuine_scores),
            impostor=_mean_score(filtered_impostor_scores),
        )
    )
    print(
        "global threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            **global_metrics,
        )
    )
    print(
        "block threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            **block_metrics,
        )
    )
    print(
        "fused_embedding_mean genuine={genuine:.4f} impostor={impostor:.4f}".format(
            genuine=_mean_score(embedding_genuine_scores),
            impostor=_mean_score(embedding_impostor_scores),
        )
    )
    print(
        "fused_embedding_cosine threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            **embedding_metrics,
        )
    )
    print(
        "soft_code_mean genuine={genuine:.4f} impostor={impostor:.4f}".format(
            genuine=_mean_score(soft_genuine_scores),
            impostor=_mean_score(soft_impostor_scores),
        )
    )
    print(
        "soft threshold={threshold:.4f} FAR={far:.4f} FRR={frr:.4f} EER={eer:.4f}".format(
            **soft_metrics,
        )
    )
    print("raw_global_curve_csv:", raw_global_curve_path)
    print("raw_block_curve_csv:", raw_block_curve_path)
    print("global_curve_csv:", global_curve_path)
    print("block_curve_csv:", block_curve_path)
    print("soft_curve_csv:", soft_curve_path)


if __name__ == "__main__":
    main()
