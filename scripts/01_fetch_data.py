"""Download every public dataset into data/raw/<dataset>/ and record provenance.

Each download lands in its own folder. For each file the script writes FETCH.json with the
URL, the UTC fetch time, the byte size, and the SHA-256. It never unpacks pickle files; the
RouterBench pickles are converted inside an isolated Modal container
(scripts/modal_convert_routerbench.py).

Usage:  uv run python scripts/01_fetch_data.py [dataset ...]
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOWNLOADS = Path.home() / "Downloads"
UA = {"User-Agent": "system-three-benchmark/0.1 (research data fetch)"}

HF = "https://huggingface.co/datasets"
IST = "https://datashare.ed.ac.uk/server/api/core/bitstreams"
SOURCES: dict[str, list[tuple[str, str]]] = {
    "llmrouterbench": [
        ("bench-release.tar.gz", f"{HF}/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz"),
        ("LICENSE", f"{HF}/NPULH/LLMRouterBench/raw/main/LICENSE"),
    ],
    "lmarena": [("train.csv", f"{HF}/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv")],
    "criteo": [
        (
            "criteo-research-uplift-v2.1.csv.gz",
            f"{HF}/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
        )
    ],
    "hillstrom": [
        (
            "hillstrom.csv.gz",
            "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
        )
    ],
    "nsw": [
        ("nswre74_treated.txt", "https://users.nber.org/~rdehejia/data/nswre74_treated.txt"),
        ("nswre74_control.txt", "https://users.nber.org/~rdehejia/data/nswre74_control.txt"),
        ("nsw_treated.txt", "https://users.nber.org/~rdehejia/data/nsw_treated.txt"),
        ("nsw_control.txt", "https://users.nber.org/~rdehejia/data/nsw_control.txt"),
    ],
    "ihdp": [
        ("ihdp_npci_1-100.train.npz", "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"),
        ("ihdp_npci_1-100.test.npz", "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"),
    ],
    "upworthy": [
        ("exploratory.csv", "https://osf.io/download/3vqmp/"),
        ("confirmatory.csv", "https://osf.io/download/vy8mj/"),
        ("holdout.csv", "https://osf.io/download/ynf3k/"),
    ],
    "illinois": [],  # git clone, below
    "ist": [
        ("IST_corrected.csv", f"{IST}/eca52b56-6d1b-47b3-814a-0b7089f8c870/content"),
        ("IST_variables.csv", f"{IST}/069dc665-f3c8-4c95-8ec0-bf6736e0c8ac/content"),
        ("license.txt", f"{IST}/cd1cab69-c585-46e4-9eee-4db8e6cc9f13/content"),
    ],
    "crash2": [("crash2.rda", "https://hbiostat.org/data/repo/crash2.rda")],
    "swebench": [],  # sparse git clone, below
    "oregon": [],  # local copy of the click-through download, below
    "camden": [],  # local copy of the Dataverse download, below
    "voter": [],  # local copy of the Yale Dataverse download (doi:10.60600/YU/CGMWNW), below
}
GIT = {
    "illinois": ("https://github.com/reifjulian/illinois-wellness-data", None),
    "swebench": (
        "https://github.com/SWE-bench/experiments",
        ["evaluation/verified/*/results/results.json", "evaluation/verified/*/metadata.yaml"],
    ),
}
LOCAL = {
    "oregon": [DOWNLOADS / "Oregon" / "oregon_puf.zip",
               DOWNLOADS / "Oregon" / "By downloading these materials, I agree to the following.txt"],
    "camden": [DOWNLOADS / "dataverse_files.zip"],
    "voter": [DOWNLOADS / "doi-10.60600-yu-cgmwnw" / f for f in (
        "GerberGreenLarimer_APSR_2008_social_pressure.dta",
        "GerberGreenLarimer_APSR_2008.R",
        "GerberGreenLarimer_APSR_2008_r_output.txt",
        "GerberGreenLarimer_APSR_2008.log",
        "MANIFEST.TXT",
    )],
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(ds: str, name: str, url: str, path: Path, log: list[dict]) -> None:
    log.append({
        "dataset": ds, "file": name, "url": url,
        "fetched_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "bytes": path.stat().st_size, "sha256": sha256(path),
    })


def fetch(ds: str) -> list[dict]:
    out = RAW / ds
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    for name, url in SOURCES.get(ds, []):
        dest = out / name
        if not dest.exists():
            with requests.get(url, headers=UA, stream=True, timeout=120) as r:
                r.raise_for_status()
                ctype = r.headers.get("content-type", "")
                if "text/html" in ctype and not name.endswith((".html", ".htm")):
                    raise RuntimeError(f"{ds}/{name}: server returned HTML, not data ({url})")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
        record(ds, name, url, dest, log)
    if ds in GIT:
        url, patterns = GIT[ds]
        repo = out / "repo"
        if not repo.exists():
            if patterns:
                subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none",
                                "--no-checkout", url, str(repo)], check=True)
                subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone",
                                *patterns], check=True)
                subprocess.run(["git", "-C", str(repo), "checkout", "-q"], check=True)
            else:
                subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(repo)], check=True)
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        log.append({"dataset": ds, "file": "repo", "url": url, "commit": head,
                    "fetched_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")})
    for src in LOCAL.get(ds, []):
        if not src.exists():
            raise FileNotFoundError(f"{ds}: expected local download at {src}")
        dest = out / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        record(ds, src.name, f"local copy of {src}", dest, log)
    (out / "FETCH.json").write_text(json.dumps(log, indent=2))
    return log


def main(argv: list[str]) -> int:
    names = argv or list(SOURCES)
    failed = []
    for ds in names:
        try:
            log = fetch(ds)
            mb = sum(x.get("bytes", 0) for x in log) / 1e6
            print(f"ok    {ds:16s} {len(log)} items, {mb:,.1f} MB", flush=True)
        except Exception as e:  # report and continue; rule 8 is applied by the operator
            failed.append(ds)
            print(f"FAIL  {ds:16s} {e}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
