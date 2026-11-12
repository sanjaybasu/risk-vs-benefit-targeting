"""Stage (default) or submit (--submit) the Paper 1 pre-registration on OSF.

Stage: create a private OSF project, upload the filled pre-registration and the evidence files,
and create a draft registration with the "Secondary Data Preregistration" template filled from the
pre-registration's sections. Nothing becomes public.
Submit: submit the staged draft as an immediate public registration. Run only after both
custodians confirm (DECISIONS.md, 2026-11-12).

Token: ~/.config/osf/token (never printed). State: results/osf_state.json.

Usage:
  uv run python scripts/10_osf_post.py            # stage
  uv run python scripts/10_osf_post.py --submit   # make public
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT.parents[1] / "notebooks" / "system-three" / "preregistration"
STATE = ROOT / "results" / "osf_state.json"
API = "https://api.osf.io/v2"
FILES = "https://files.osf.io/v1/resources"
SCHEMA = "64775783798e08000a70407e"  # Secondary Data Preregistration v3
SUBJECTS = ["584240d954be81056ceca973",  # Statistics and Probability
            "584240da54be81056cecacac",  # Health Services Research
            "584240da54be81056cecac48",  # Social and Behavioral Sciences
            "584240da54be81056cecaaaa"]  # Medicine and Health Sciences
TITLE = "When does targeting by risk fail? A pre-registered test of a scale forecast across randomized trials"


def session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + (Path.home() / ".config" / "osf" / "token").read_text().strip()
    s.headers["Content-Type"] = "application/vnd.api+json"
    return s


def sections(md: str) -> dict[str, str]:
    """Split the pre-registration into its numbered sections."""
    parts = re.split(r"^## ", md, flags=re.M)
    out = {}
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        out[head.strip()] = body.strip()
    return out


def sub(text: str, start: str, stop: str | None = None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(stop, i + len(start)) if stop else -1
    return text[i: j if j > 0 else None].strip()


def responses(md: str) -> dict:
    s = sections(md)
    q, h, d = s["1. Question"], s["2. Hypotheses"], s["3. Data"]
    seal, rank, est = s["4. Splits and the seal"], s["5. Rankings"], s["6. Estimands and analysis"]
    lim = s["7. What this study does not claim"]
    return {
        "72-2": q,
        "72-4": h,
        "72-7": d,
        "72-9": "The dataset is publicly available",
        "72-14": "Every dataset is public: direct download, click-through agreement, or a free "
                 "account with no proposal review. URLs, licenses, and SHA-256 hashes are in DATA.md "
                 "of the code repository (attached as code archive).",
        "72-16": "See Table D and DATA.md (DOIs and URLs per dataset).",
        "72-18": "2026-11-12",
        "72-20": "Each dataset is the public file of a completed randomized trial or randomized "
                 "experiment; collection procedures are in each source publication (Table D).",
        "72-33": "Outcome per trial (binary; Table D). Pre-assignment covariates only, whitelisted per "
                 "trial in the loaders (src/system_three/loaders). " + sub(est, "*Primary estimand", "*Primary moderator"),
        "72-37": "The randomization unit of each trial (individual or household; Table D). Inference is "
                 "clustered at that unit.",
        "72-44": "Each trial's primary result is published (Table D). The cross-trial analysis is new.",
        "72-46": sub(seal, "### 4a. Prior data contact") + "\n\n" + seal.split("### 4a.")[0].strip(),
        "72-49": rank + "\n\n" + sub(est, "*Primary moderator", "*Power and influence"),
        "72-53": sub(est, "*Power and influence", "*Forward prediction"),
        "72-55": sub(est, "*Multiplicity") ,
        "72-59": sub(est, "*Sensitivity analyses for H1", "*Power and influence"),
        "72-61": sub(est, "*Secondary analyses", "*Multiplicity") + "\n\n" + lim,
    }


def existing(s: requests.Session, node: str) -> set[str]:
    r = s.get(f"{API}/nodes/{node}/files/osfstorage/?page%5Bsize%5D=100")
    r.raise_for_status()
    return {f["attributes"]["name"] for f in r.json()["data"]}


def upload(s: requests.Session, node: str, path: Path, name: str | None = None, tries: int = 5) -> None:
    import time

    url = f"{FILES}/{node}/providers/osfstorage/?kind=file&name={name or path.name}"
    for k in range(tries):
        try:
            r = s.put(url, data=path.read_bytes(), headers={"Content-Type": "application/octet-stream"},
                      timeout=600)
            if r.status_code in (200, 201):
                return
            if r.status_code == 409:  # already uploaded by an earlier, interrupted run
                return
            err = f"{r.status_code} {r.text[:300]}"
        except requests.exceptions.RequestException as e:
            err = str(e)
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"upload {path.name}: {err}")


def stage() -> dict:
    s = session()
    md = (NB / "PREREGISTRATION_v1_filled.md").read_text()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "node" not in st:
        r = s.post(f"{API}/nodes/", json={"data": {"type": "nodes", "attributes": {
            "title": TITLE, "category": "project", "public": False,
            "description": "Pre-registration, seal manifests, forecasts, and code for Paper 1 of the "
                           "System Three program (Waymark)."}}})
        r.raise_for_status()
        st["node"] = r.json()["data"]["id"]
        STATE.write_text(json.dumps(st, indent=1))  # record the project before any upload
    node = st["node"]
    archive = ROOT / "results" / "code_archive.tar.gz"
    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar.gz", "-o", str(archive), "HEAD"],
                   check=True)
    files = [NB / "PREREGISTRATION_v1_filled.md", NB / "redteam_v0_revision_memo.md",
             ROOT / "results" / "seal_manifest.json", ROOT / "predictions" / "manifest.json",
             ROOT / "results" / "scale_forecast.json", ROOT / "results" / "h1_power_simulation.json",
             ROOT / "results" / "holdouts.json", ROOT / "results" / "dev_baselines.json",
             ROOT / "results" / "dataset_table.json", archive]
    files += sorted((ROOT / "predictions").glob("*.parquet"))
    have = existing(s, node)
    for f in files:
        name = "predictions_manifest.json" if f.parent.name == "predictions" and f.suffix == ".json" else None
        if (name or f.name) in have:
            continue
        upload(s, node, f, name)
    if "draft" not in st:
        r = s.post(f"{API}/draft_registrations/", json={"data": {
            "type": "draft_registrations",
            "attributes": {"title": TITLE, "registration_responses": responses(md)},
            "relationships": {
                "branched_from": {"data": {"type": "nodes", "id": node}},
                "registration_schema": {"data": {"type": "registration-schemas", "id": SCHEMA}}}}})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"draft: {r.status_code} {r.text[:500]}")
        st["draft"] = r.json()["data"]["id"]
    else:
        r = s.patch(f"{API}/draft_registrations/{st['draft']}/", json={"data": {
            "type": "draft_registrations", "id": st["draft"],
            "attributes": {"registration_responses": responses(md)}}})
        r.raise_for_status()
    # OSF requires at least one subject to register (OSF core taxonomy ids).
    subj = [{"type": "subjects", "id": i} for i in SUBJECTS]
    r = s.put(f"{API}/draft_registrations/{st['draft']}/relationships/subjects/", json={"data": subj})
    if r.status_code not in (200, 201, 204):
        raise RuntimeError(f"subjects: {r.status_code} {r.text[:300]}")
    st["staged_files"] = [f.name for f in files]
    STATE.write_text(json.dumps(st, indent=1))
    return st


def submit() -> dict:
    s = session()
    st = json.loads(STATE.read_text())
    r = s.post(f"{API}/registrations/", json={"data": {
        "type": "registrations",
        "attributes": {"draft_registration": st["draft"], "registration_choice": "immediate"}}})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"submit: {r.status_code} {r.text[:500]}")
    st["registration"] = r.json()["data"]["id"]
    STATE.write_text(json.dumps(st, indent=1))
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    print(json.dumps(submit() if a.submit else stage(), indent=1))
