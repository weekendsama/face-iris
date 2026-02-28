from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 123
    data_seed: int = 42
    dataset_root: str = "data"
    lfw_root: str = r"D:\classwork\lfw"
    casia_iris_root: str = r"E:\BaiduNetdiskDownload\CASIA-Iris-Thousand"
    pairing_mode: str = "public_manual_indexed"
    use_public_dataset_roots: bool = True
    checkpoint_dir: str = "checkpoints/public_manual"
    checkpoint_name: str = "multimodal_cancelable_public_manual"
    eval_report_dir: str = "eval_reports/public_manual"
    train_steps: int = 16
    eval_warmup_steps: int = 4
    eval_batches: int = 8
    eval_identities: int = 16
    eval_samples_per_identity: int = 4
    eval_probe_bit_flip_prob: float = 0.08
    train_identity_fraction: float = 0.8
    face_input_shape: tuple[int, int, int] = (224, 224, 3)
    iris_input_shape: tuple[int, int, int] = (128, 128, 1)
    face_embedding_dim: int = 512
    iris_embedding_dim: int = 128
    shared_embedding_dim: int = 256
    shared_hash_bits: int = 256
    smoothing_alpha: float = 2.0
    block_size: int = 32
    num_classes: int = 1000
    learning_rate: float = 3e-4
    batch_size: int = 8
    pair_margin: float = 0.20
    pair_positive_target: float = 0.20
    classification_weight: float = 1.0
    pairwise_weight: float = 0.05
    embedding_positive_target: float = 0.80
    embedding_negative_target: float = 0.15
    embedding_pairwise_weight: float = 0.40
    adaface_margin: float = 0.4
    adaface_scale: float = 32.0
    adaface_h: float = 0.333
    dsh_num_clusters: int = 4
    dsh_kmeans_iters: int = 3
    dsh_refresh_interval: int = 3
    dsh_cache_batches: int = 4
    dsh_bank_per_class: int = 3
    dsh_min_samples_per_class: int = 2
    dsh_projection_mix: float = 0.05
    sampler_identities_per_batch: int = 4


DEFAULT_CONFIG = ModelConfig()
