from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import tensorflow as tf
from .metrics import compute_metrics, Thresholds
from .utils import LoadOptions
from .visualization import save_abs_diff, save_heatmap_overlay
from .report import build_html_report

def run_compare(
    image_a: str,
    image_b: str,
    resize_width: int = 640,
    resize_height: int = 640,
    grayscale: bool = False,
    json_out: Optional[str] = None,
    html_out: Optional[str] = None,
    out_diff: Optional[str] = None,
    out_heatmap: Optional[str] = None,
    ssim_threshold: float = 0.95,
    psnr_threshold: float = 35.0,
    emb_threshold: float = 0.95,
    fail_on_size_mismatch: bool = False,
):
    opts = LoadOptions(resize=(resize_width, resize_height), grayscale=grayscale)
    th = Thresholds(ssim=ssim_threshold, psnr=psnr_threshold, embedding_cosine=emb_threshold)

    metrics, abs_diff, diff_mask = compute_metrics(image_a, image_b, opts, th)

    # Save images
    out_diff = out_diff or "abs_diff.png"
    out_heatmap = out_heatmap or "heatmap_overlay.png"
    save_abs_diff(abs_diff, out_diff)
    save_heatmap_overlay(image_a, diff_mask, out_heatmap)

    # Optionally override status on size mismatch
    if fail_on_size_mismatch and metrics.get("size_mismatch", False):
        metrics["status"] = "FAIL"
        metrics.setdefault("failure_reasons", []).append("SIZE_MISMATCH")

    # JSON
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(metrics, f, indent=2)

    # HTML
    if html_out:
        Path(html_out).parent.mkdir(parents=True, exist_ok=True)
        html = build_html_report(image_a, image_b, out_diff, out_heatmap, metrics)
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(html)

    return metrics, out_diff, out_heatmap
