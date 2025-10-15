from __future__ import annotations
import base64
from dataclasses import dataclass
from typing import Dict, Any
from jinja2 import Template

HTML_TMPL = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Image Validation Report</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#111}
h1{font-size:1.6rem;margin:0 0 8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}
.card{border:1px solid #eee;border-radius:12px;padding:12px;box-shadow:0 2px 6px rgba(0,0,0,.04)}
.badge{display:inline-block;padding:.25rem .5rem;border-radius:999px;font-size:.8rem}
.pass{background:#e8f5e9;color:#256029}
.fail{background:#ffebee;color:#b71c1c}
table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #eee;padding:8px;text-align:left}
img{max-width:100%;height:auto;border-radius:8px}
.caption{font-size:.85rem;color:#666}
</style>
</head>
<body>
<h1>Image Validation Report</h1>
<div>Status:
<span class="badge {{ 'pass' if status == 'PASS' else 'fail' }}">{{ status }}</span>
</div>
<p class="caption">Failure reasons: {{ failure_reasons or ['None'] | join(', ') }}</p>

<div class="grid">
  <div class="card">
    <h3>Image A</h3>
    <img src="data:image/png;base64,{{ img_a_b64 }}" />
  </div>
  <div class="card">
    <h3>Image B</h3>
    <img src="data:image/png;base64,{{ img_b_b64 }}" />
  </div>
</div>

<div class="grid">
  <div class="card">
    <h3>Abs Difference</h3>
    <img src="data:image/png;base64,{{ diff_b64 }}" />
  </div>
  <div class="card">
    <h3>Heatmap Overlay</h3>
    <img src="data:image/png;base64,{{ heat_b64 }}" />
  </div>
</div>

<div class="card">
  <h3>Metrics</h3>
  <table>
    <tbody>
      {% for k, v in metrics.items() %}
      {% if k not in ['thresholds','status','failure_reasons','size_mismatch'] %}
      <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
      {% endif %}
      {% endfor %}
      <tr><th>Size Mismatch</th><td>{{ size_mismatch }}</td></tr>
      <tr><th>Thresholds</th><td>{{ thresholds }}</td></tr>
    </tbody>
  </table>
</div>
</body>
</html>
""")

@dataclass
class ReportInputs:
    img_a_b64: str
    img_b_b64: str
    diff_b64: str
    heat_b64: str
    metrics: Dict[str, Any]

def _encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def build_html_report(path_a: str, path_b: str, diff_path: str, heat_path: str, metrics: Dict[str, Any]) -> str:
    html = HTML_TMPL.render(
        status = metrics.get("status","PASS"),
        failure_reasons = metrics.get("failure_reasons", []),
        metrics = metrics,
        size_mismatch = metrics.get("size_mismatch", False),
        thresholds = metrics.get("thresholds", {}),
        img_a_b64 = _encode_file(path_a),
        img_b_b64 = _encode_file(path_b),
        diff_b64 = _encode_file(diff_path),
        heat_b64 = _encode_file(heat_path),
    )
    return html
