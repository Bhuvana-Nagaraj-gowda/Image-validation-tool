# Image Validation Tool (TensorFlow)

A complete, production‑ready starter to compare two images and report issues using TensorFlow.
It calculates structural and perceptual metrics (SSIM, PSNR, MobileNetV2 embedding similarity),
produces visual diffs, and outputs JSON and HTML reports. A Streamlit UI is also included.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
imgval compare examples/sample_a.jpg examples/sample_b.jpg --html report.html --json report.json
# Launch UI
streamlit run app/streamlit_app.py
```

## CLI

```bash
imgval compare <image_a> <image_b> [options]

Options:
  --resize-width INT           Width to resize for metrics (default: 640)
  --resize-height INT          Height to resize for metrics (default: 640)
  --grayscale / --no-grayscale Use grayscale for SSIM/PSNR (default: False)
  --json PATH                  Where to write JSON report
  --html PATH                  Where to write HTML report
  --out-diff PATH              Save raw absolute-difference image (PNG)
  --out-heatmap PATH           Save heatmap overlay image (PNG)
  --ssim-threshold FLOAT       If SSIM < threshold, mark as FAIL (default: 0.95)
  --psnr-threshold FLOAT       If PSNR < threshold, mark as FAIL (default: 35.0)
  --emb-threshold FLOAT        If Embedding cosine < threshold, mark as FAIL (default: 0.95)
  --fail-on-size-mismatch      Fail if dimensions differ (default: False)
  --verbose                    Print more logs
```

## Metrics

- **SSIM (TensorFlow)**: `tf.image.ssim` on either RGB or luminance.
- **PSNR (TensorFlow)**: `tf.image.psnr`.
- **Embedding Cosine Similarity**: MobileNetV2 (ImageNet, avg pooled) cosine similarity.
- **Pixel Difference Map**: absolute |A−B| and thresholded mask.
- **Histogram Drift**: per‑channel χ² distance (informational).

## Reports

- **JSON**: machine‑readable metrics & pass/fail.
- **HTML**: side‑by‑side images with overlays and metric table.

## UI

Run `streamlit run app/streamlit_app.py` for a simple web UI to drag‑and‑drop two images and export a report.

## Tests

`pytest`-style smoke test for metric functions (no heavy downloads).

---

**Note**: The first run of embedding similarity will download MobileNetV2 weights.
