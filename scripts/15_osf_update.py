"""Post registration update 1 to the OSF project linked to registration r2vsu.

upload  (default): upload REGISTRATION_UPDATE_1.md, the new manifests, and the new prediction sets
        into folder update_1/ of project rs7pm. OSF time-stamps every file version, so the files
        are on record before any unsealing even while the project is private.
--publish: once the registration is approved (public), make project rs7pm public too.

Usage: uv run python scripts/15_osf_update.py [--publish]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT.parents[1] / "notebooks" / "system-three" / "preregistration"
API = "https://api.osf.io/v2"
FILES = "https://files.osf.io/v1/resources"
NODE, REG = "rs7pm", "r2vsu"


def session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + (Path.home() / ".config" / "osf" / "token").read_text().strip()
    return s


def folder(s: requests.Session, name: str) -> str:
    r = s.get(f"{API}/nodes/{NODE}/files/osfstorage/?page%5Bsize%5D=100")
    r.raise_for_status()
    for f in r.json()["data"]:
        if f["attributes"]["kind"] == "folder" and f["attributes"]["name"] == name:
            return f["links"]["upload"]
    r = s.put(f"{FILES}/{NODE}/providers/osfstorage/?kind=folder&name={name}")
    r.raise_for_status()
    return r.json()["data"]["links"]["upload"]


def upload_all() -> None:
    s = session()
    up = folder(s, "update_1").split("?")[0]  # the folder link may already carry ?kind=...
    files = [NB / "REGISTRATION_UPDATE_1.md", NB / "REGISTRATION_UPDATE_2.md", NB / "REGISTRATION_UPDATE_3.md"]
    files += [ROOT / "results" / f for f in ("loeo_manifest.json", "scale_forecast_splitctx.json",
                                             "causalpfn_manifest.json", "dopfn_manifest.json",
                                             "llm_manifest.json", "tierA_manifest.json",
                                             "rose_manifest.json")]
    for d in ("predictions_loeo", "predictions_splitctx", "predictions_causalpfn",
              "predictions_dopfn", "predictions_llm", "predictions_tierA", "predictions_rose"):
        files += [(f, f"{d}__{f.name}") for f in sorted((ROOT / d).glob("*.parquet"))]
    listing = {}
    lr = s.get(f"{API}/nodes/{NODE}/files/osfstorage/?page%5Bsize%5D=100").json()["data"]
    fid = next(f["id"] for f in lr if f["attributes"]["name"] == "update_1")
    nxt = f"{API}/nodes/{NODE}/files/osfstorage/{fid}/?page%5Bsize%5D=100"
    while nxt:
        page = s.get(nxt).json()
        for f in page["data"]:
            listing[f["attributes"]["name"]] = (f["links"]["upload"], f["attributes"].get("extra", {})
                                                 .get("hashes", {}).get("sha256"))
        nxt = page["links"].get("next")
    import hashlib

    for item in files:
        path, name = (item, item.name) if isinstance(item, Path) else item
        if not path.exists():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        if name in listing:
            link, remote = listing[name]
            if remote == hashlib.sha256(data).hexdigest():
                continue  # unchanged
            rr = s.put(f"{link.split('?')[0]}?kind=file", data=data, timeout=600)  # new version
        else:
            rr = s.put(f"{up}?kind=file&name={name}", data=data, timeout=600)
        if rr.status_code not in (200, 201):
            raise RuntimeError(f"{name}: {rr.status_code} {rr.text[:200]}")
        print("uploaded", name, rr.status_code)


def publish() -> None:
    s = session()
    reg = s.get(f"{API}/registrations/{REG}/").json()["data"]["attributes"]
    if not reg.get("public"):
        raise SystemExit("registration not yet approved/public; publish later")
    r = s.patch(f"{API}/nodes/{NODE}/", json={"data": {"type": "nodes", "id": NODE,
                                                       "attributes": {"public": True}}},
                headers={"Content-Type": "application/vnd.api+json"})
    r.raise_for_status()
    print("project public:", r.json()["data"]["attributes"]["public"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    publish() if ap.parse_args().publish else upload_all()
