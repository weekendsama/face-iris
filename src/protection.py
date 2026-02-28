from __future__ import annotations

import tensorflow as tf


def hard_binarize(code: tf.Tensor) -> tf.Tensor:
    return tf.cast(code >= 0.5, tf.int32)


def generate_user_key(num_bits: int, seed: int = 42) -> tuple[tf.Tensor, tf.Tensor]:
    seed_tensor = tf.constant([seed, seed ^ 0xABCDEF], dtype=tf.int32)
    shuffle_noise = tf.random.stateless_uniform(
        shape=(num_bits,),
        seed=seed_tensor,
        dtype=tf.float32,
    )
    perm = tf.argsort(shuffle_noise)
    mask = tf.random.stateless_uniform(
        shape=(num_bits,),
        seed=seed_tensor + 1,
        minval=0,
        maxval=2,
        dtype=tf.int32,
    )
    return perm, mask


def protect_template(code: tf.Tensor, perm: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    code = hard_binarize(code)
    permuted = tf.gather(code, perm, axis=-1)
    return tf.bitwise.bitwise_xor(permuted, mask)


def hamming_distance(template_a: tf.Tensor, template_b: tf.Tensor) -> tf.Tensor:
    diff = tf.not_equal(template_a, template_b)
    return tf.reduce_mean(tf.cast(diff, tf.float32), axis=-1)


def hamming_similarity(template_a: tf.Tensor, template_b: tf.Tensor) -> tf.Tensor:
    return 1.0 - hamming_distance(template_a, template_b)


def blockwise_hamming_similarity(
    template_a: tf.Tensor,
    template_b: tf.Tensor,
    block_size: int,
) -> tf.Tensor:
    num_bits = tf.shape(template_a)[-1]
    usable_bits = (num_bits // block_size) * block_size
    a = template_a[..., :usable_bits]
    b = template_b[..., :usable_bits]
    new_shape = tf.concat([tf.shape(a)[:-1], [usable_bits // block_size, block_size]], axis=0)
    a_blocks = tf.reshape(a, new_shape)
    b_blocks = tf.reshape(b, new_shape)
    block_scores = 1.0 - tf.reduce_mean(
        tf.cast(tf.not_equal(a_blocks, b_blocks), tf.float32),
        axis=-1,
    )
    return tf.reduce_mean(block_scores, axis=-1)
