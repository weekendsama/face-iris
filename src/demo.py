from __future__ import annotations

import tensorflow as tf

from .config import DEFAULT_CONFIG
from .model import build_model
from .protection import (
    blockwise_hamming_similarity,
    generate_user_key,
    hamming_similarity,
    protect_template,
)


def main() -> None:
    config = DEFAULT_CONFIG
    model = build_model(config)

    face_batch = tf.random.uniform((2, *config.face_input_shape), maxval=255, dtype=tf.float32)
    iris_batch = tf.random.uniform((2, *config.iris_input_shape), maxval=255, dtype=tf.float32)

    outputs = model((face_batch, iris_batch), training=False)
    shared_code = outputs["shared_code"]

    perm, mask = generate_user_key(config.shared_hash_bits, seed=7)
    protected_a = protect_template(shared_code[0], perm, mask)
    protected_b = protect_template(shared_code[1], perm, mask)

    global_score = hamming_similarity(protected_a, protected_b)
    block_score = blockwise_hamming_similarity(
        protected_a,
        protected_b,
        block_size=config.block_size,
    )

    print("Shared code shape:", shared_code.shape)
    print("Protected template bits:", protected_a.shape[-1])
    print("Global similarity:", float(global_score.numpy()))
    print("Block similarity:", float(block_score.numpy()))


if __name__ == "__main__":
    main()
