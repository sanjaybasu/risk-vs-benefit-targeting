"""Upload every scoring output, log, and manifest to folder scoring/ of OSF project rs7pm (update 3).

Files are flattened to <stage>__<name>; a changed file uploads as a new OSF version, never a new
name, so no run can be dropped. Optional extra files (for example results/holdout_forecasts.json)
are given as arguments.

Usage: uv run python scripts/22_osf_scoring_upload.py [extra files ...]
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("osf", ROOT / "scripts" / "15_osf_update.py")
osf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osf)


def main() -> None:
    s = osf.session()
    up = osf.folder(s, "scoring").split("?")[0]
    sc = ROOT / "scoring"
    files = [(p, p.relative_to(sc).as_posix().replace("/", "__")) for p in sorted(sc.rglob("*"))
             if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".out")]
    files += [(Path(f).resolve(), Path(f).name) for f in sys.argv[1:]]
    lr = s.get(f"{osf.API}/nodes/{osf.NODE}/files/osfstorage/?page%5Bsize%5D=100").json()["data"]
    fid = next(f["id"] for f in lr if f["attributes"]["name"] == "scoring")
    listing, nxt = {}, f"{osf.API}/nodes/{osf.NODE}/files/osfstorage/{fid}/?page%5Bsize%5D=100"
    while nxt:
        page = s.get(nxt).json()
        for f in page["data"]:
            listing[f["attributes"]["name"]] = (f["links"]["upload"],
                                                f["attributes"].get("extra", {}).get("hashes", {}).get("sha256"))
        nxt = page["links"].get("next")
    n = 0
    for path, name in files:
        data = path.read_bytes()
        if name in listing:
            link, remote = listing[name]
            if remote == hashlib.sha256(data).hexdigest():
                continue
            url = f"{link.split('?')[0]}?kind=file"
        else:
            url = f"{up}?kind=file&name={name}"
        for attempt in range(6):  # OSF returns intermittent 403/5xx on bursts of writes
            r = s.put(url, data=data, timeout=600)
            if r.status_code in (200, 201):
                break
            time.sleep(3 * 2 ** attempt)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"{name}: {r.status_code} {r.text[:200]}")
        n += 1
    print(f"uploaded {n} of {len(files)} files to OSF scoring/")


if __name__ == "__main__":
    main()
