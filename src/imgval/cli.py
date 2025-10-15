from __future__ import annotations
import typer
from rich import print
from .compare import run_compare

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

@app.command()
def compare(
    image_a: str = typer.Argument(..., help="Path to first image"),
    image_b: str = typer.Argument(..., help="Path to second image"),
    resize_width: int = typer.Option(640, help="Resize width for metrics"),
    resize_height: int = typer.Option(640, help="Resize height for metrics"),
    grayscale: bool = typer.Option(False, help="Use grayscale for SSIM/PSNR"),
    json: str = typer.Option(None, help="Path to write JSON report"),
    html: str = typer.Option(None, help="Path to write HTML report"),
    out_diff: str = typer.Option(None, help="Save absolute difference PNG here"),
    out_heatmap: str = typer.Option(None, help="Save heatmap overlay PNG here"),
    ssim_threshold: float = typer.Option(0.95, help="Minimum SSIM to pass"),
    psnr_threshold: float = typer.Option(35.0, help="Minimum PSNR to pass"),
    emb_threshold: float = typer.Option(0.95, help="Minimum embedding cosine to pass"),
    fail_on_size_mismatch: bool = typer.Option(False, help="Fail if dimensions differ"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    if verbose:
        print("[bold cyan]Image Validation[/] starting...")
    metrics, diff_path, heat_path = run_compare(
        image_a=image_a,
        image_b=image_b,
        resize_width=resize_width,
        resize_height=resize_height,
        grayscale=grayscale,
        json_out=json,
        html_out=html,
        out_diff=out_diff,
        out_heatmap=out_heatmap,
        ssim_threshold=ssim_threshold,
        psnr_threshold=psnr_threshold,
        emb_threshold=emb_threshold,
        fail_on_size_mismatch=fail_on_size_mismatch,
    )
    print(f"[bold]Status:[/] {metrics['status']}  |  SSIM={metrics['ssim']:.4f}  PSNR={metrics['psnr']:.2f}  EMB={metrics['embedding_cosine']:.4f}")
    if metrics.get("failure_reasons"):
        print(f"[red]Reasons:[/] {', '.join(metrics['failure_reasons'])}")
    if html:
        print(f"HTML report: {html}")
    if json:
        print(f"JSON report: {json}")
    if diff_path:
        print(f"Diff image: {diff_path}")
    if heat_path:
        print(f"Heatmap: {heat_path}")
