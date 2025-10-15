import streamlit as st
from pathlib import Path
import tempfile
from imgval.compare import run_compare

st.set_page_config(page_title="Image Validation", layout="wide")
st.title(" Image Validation Tool")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Image A", type=["png","jpg","jpeg","bmp","webp"])
with col2:
    file_b = st.file_uploader("Upload Image B", type=["png","jpg","jpeg","bmp","webp"])

with st.expander("Options"):
    resize_w = st.number_input("Resize width", value=640, min_value=64, step=64)
    resize_h = st.number_input("Resize height", value=640, min_value=64, step=64)
    grayscale = st.checkbox("Grayscale for SSIM/PSNR", value=False)
    ssim_thr = st.number_input("SSIM threshold", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
    psnr_thr = st.number_input("PSNR threshold", value=35.0, min_value=0.0, max_value=100.0, step=0.5)
    emb_thr = st.number_input("Embedding cosine threshold", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
    fail_size = st.checkbox("Fail on size mismatch", value=False)

run = st.button("Run comparison", type="primary", disabled=not (file_a and file_b))

if run:
    with tempfile.TemporaryDirectory() as tmpd:
        pa = Path(tmpd) / file_a.name
        pb = Path(tmpd) / file_b.name
        pa.write_bytes(file_a.read())
        pb.write_bytes(file_b.read())
        metrics, diff_path, heat_path = run_compare(
            str(pa), str(pb),
            resize_width=resize_w, resize_height=resize_h, grayscale=grayscale,
            ssim_threshold=ssim_thr, psnr_threshold=psnr_thr, emb_threshold=emb_thr,
            fail_on_size_mismatch=fail_size,
            json_out=str(Path(tmpd) / "report.json"),
            html_out=str(Path(tmpd) / "report.html"),
        )
        st.success(f"Status: {metrics['status']}")
        st.json(metrics)
        st.image([str(pa), str(pb)], caption=["Image A","Image B"])
        st.image([diff_path, heat_path], caption=["Absolute Difference", "Heatmap Overlay"])

        # Download buttons
        st.download_button("Download JSON", Path(tmpd, "report.json").read_bytes(), file_name="report.json")
        st.download_button("Download HTML", Path(tmpd, "report.html").read_bytes(), file_name="report.html")
