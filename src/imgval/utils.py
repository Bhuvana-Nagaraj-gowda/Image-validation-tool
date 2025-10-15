from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import tensorflow as tf
from PIL import Image
import numpy as np

@dataclass
class LoadOptions:
    resize: Tuple[int, int] = (640, 640)
    grayscale: bool = False

def load_image(path: str, opts: LoadOptions) -> tf.Tensor:
    # Load with PIL to support many formats and preserve EXIF orientation
    img = Image.open(path).convert("RGB")
    if opts.grayscale:
        img = img.convert("L").convert("RGB")
    img = img.resize(opts.resize, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return tf.convert_to_tensor(arr)

def cosine_similarity(a: tf.Tensor, b: tf.Tensor, axis=-1, eps: float = 1e-8) -> tf.Tensor:
    a_norm = tf.nn.l2_normalize(a, axis=axis, epsilon=eps)
    b_norm = tf.nn.l2_normalize(b, axis=axis, epsilon=eps)
    return tf.reduce_sum(a_norm * b_norm, axis=axis)
