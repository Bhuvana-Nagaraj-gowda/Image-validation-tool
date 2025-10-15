from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
from .utils import LoadOptions, load_image, cosine_similarity

@dataclass
class Thresholds:
    ssim: float = 0.95
    psnr: float = 35.0
    embedding_cosine: float = 0.95

def _mobilenet_encoder():
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
    return base

def compute_metrics(path_a: str, path_b: str, opts: LoadOptions, thresholds: Thresholds):
    img_a = load_image(path_a, opts)
    img_b = load_image(path_b, opts)

    if img_a.shape != img_b.shape:
        size_mismatch = True
    else:
        size_mismatch = False

    # TensorFlow expects batched tensors in some keras models
    a = tf.expand_dims(img_a, 0)
    b = tf.expand_dims(img_b, 0)

    # SSIM/PSNR (range [0,1])
    ssim = tf.image.ssim(img_a, img_b, max_val=1.0)
    psnr = tf.image.psnr(img_a, img_b, max_val=1.0)

    # Embedding cosine similarity using MobileNetV2
    encoder = _mobilenet_encoder()
    a_pre = tf.keras.applications.mobilenet_v2.preprocess_input(a * 255.0)
    b_pre = tf.keras.applications.mobilenet_v2.preprocess_input(b * 255.0)
    emb_a = encoder(a_pre, training=False)
    emb_b = encoder(b_pre, training=False)
    emb_cos = cosine_similarity(emb_a, emb_b, axis=-1)

    # Absolute pixel difference and mask
    abs_diff = tf.abs(img_a - img_b)  # [H,W,3]
    mean_abs = tf.reduce_mean(abs_diff)
    max_abs = tf.reduce_max(abs_diff)
    # Simple threshold from Otsu-like heuristic on mean per-pixel delta
    gray_diff = tf.reduce_mean(abs_diff, axis=-1)  # [H,W]
    thresh = tfp_otsu_threshold(gray_diff)  # scalar
    diff_mask = gray_diff > thresh

    # Histogram chi-square distance (per channel, numpy)
    hist_chi2 = _histogram_chi2(img_a, img_b)

    # Build result dict
    metrics = {
        "ssim": float(ssim.numpy() if hasattr(ssim, "numpy") else ssim),
        "psnr": float(psnr.numpy() if hasattr(psnr, "numpy") else psnr),
        "embedding_cosine": float(emb_cos.numpy().squeeze() if hasattr(emb_cos, "numpy") else emb_cos),
        "mean_absolute_diff": float(mean_abs.numpy() if hasattr(mean_abs, "numpy") else mean_abs),
        "max_absolute_diff": float(max_abs.numpy() if hasattr(max_abs, "numpy") else max_abs),
        "hist_chi2_rgb": hist_chi2,
        "size_mismatch": bool(size_mismatch),
        "thresholds": thresholds.__dict__,
    }

    # Determine pass/fail
    failures = []
    if metrics["ssim"] < thresholds.ssim:
        failures.append(f"SSIM<{thresholds.ssim}")
    if metrics["psnr"] < thresholds.psnr:
        failures.append(f"PSNR<{thresholds.psnr}")
    if metrics["embedding_cosine"] < thresholds.embedding_cosine:
        failures.append(f"EMB<{thresholds.embedding_cosine}")

    metrics["status"] = "FAIL" if failures else "PASS"
    metrics["failure_reasons"] = failures

    # Return also the raw arrays for visualization
    return metrics, abs_diff.numpy(), diff_mask.numpy()

def _histogram_chi2(img_a: tf.Tensor, img_b: tf.Tensor, bins: int = 32):
    import numpy as np
    a = img_a.numpy().reshape(-1, 3)
    b = img_b.numpy().reshape(-1, 3)
    chi2 = []
    for c in range(3):
        ha, _ = np.histogram(a[:, c], bins=bins, range=(0, 1), density=True)
        hb, _ = np.histogram(b[:, c], bins=bins, range=(0, 1), density=True)
        # add small epsilon to avoid division by zero
        eps = 1e-8
        chi = 0.5 * np.sum(((ha - hb) ** 2) / (ha + hb + eps))
        chi2.append(float(chi))
    return chi2

def tfp_otsu_threshold(gray: tf.Tensor) -> tf.Tensor:
    """Compute an Otsu-like threshold using TensorFlow ops only."""
    # gray in [0,1], shape [H,W]
    hist = tf.histogram_fixed_width(gray, value_range=[0.0, 1.0], nbins=256)
    hist = tf.cast(hist, tf.float32)
    p = hist / tf.reduce_sum(hist)
    omega = tf.math.cumsum(p)
    mu = tf.math.cumsum(p * tf.range(256, dtype=tf.float32))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-8)
    # index of max between-class variance
    idx = tf.argmax(sigma_b2)
    return tf.cast(idx, tf.float32) / 255.0
