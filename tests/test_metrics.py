from imgval.utils import LoadOptions, load_image
from imgval.metrics import compute_metrics, Thresholds

def test_smoke_import():
    assert Thresholds().ssim > 0.0
