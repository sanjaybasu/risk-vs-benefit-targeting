"""Run the registered proportionality diagnostic (scripts/28, imported unchanged) on wave-2 trials.

Usage: uv run python scripts/35_w2_diagnostic.py w2_a w2_b ...  -> results/proportionality_diagnostic_w2.json
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diag", ROOT / "scripts" / "28_proportionality_diagnostic.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

path = ROOT / "results" / "proportionality_diagnostic_w2.json"
out = json.loads(path.read_text()) if path.exists() else {}
for t in sys.argv[1:]:
    b = diag.prep(t, "context")
    X, _ = diag.rankers.design_matrix(b)
    out[t] = diag.diagnose(X, b["W"].to_numpy(), b["outcome"].to_numpy(), b["e"].to_numpy(),
                           b["cluster_id"].to_numpy(), b["better"].iat[0])
    print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[t].items()}, flush=True)
    path.write_text(json.dumps(out, indent=1))
