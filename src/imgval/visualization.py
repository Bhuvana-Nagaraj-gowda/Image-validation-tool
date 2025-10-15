from __future__ import annotations
import numpy as np
from PIL import Image, ImageEnhance

def save_abs_diff(abs_diff: np.ndarray, path: str):
    # abs_diff in [0,1], shape [H,W,3]
    arr = (np.clip(abs_diff, 0, 1) * 255).astype("uint8")
    Image.fromarray(arr).save(path)

def save_heatmap_overlay(img_path_a: str, diff_mask: np.ndarray, out_path: str, alpha: float = 0.5):
    base = Image.open(img_path_a).convert("RGB")
    mask = (diff_mask * 255).astype("uint8")
    # Create red overlay where mask is active
    red = Image.new("RGB", base.size, (255, 0, 0))
    mask_img = Image.fromarray(mask).resize(base.size, Image.NEAREST)
    overlay = Image.composite(red, base, mask_img)
    # Blend with original
    blended = Image.blend(base, overlay, alpha=alpha)
    ImageEnhance.Sharpness(blended).enhance(1.2).save(out_path)
