"""Convert the RouterBench pickle files to parquet inside an isolated Modal CPU container.

RouterBench ships only as pandas pickles. Unpickling runs arbitrary code, so it never happens on
a workstation. This container downloads the 0-shot and 5-shot pickles from Hugging Face,
unpickles them, and writes parquet plus a schema summary to the Modal volume
`system-three-data`. Cost: CPU only, a few minutes; no GPU, no LLM API.

Run:   modal run scripts/modal_convert_routerbench.py
Fetch: modal volume get system-three-data routerbench data/raw/routerbench --force
"""

import modal

app = modal.App("system-three-routerbench-convert")
vol = modal.Volume.from_name("system-three-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pandas==2.2.3", "pyarrow==18.1.0", "numpy<2.2", "huggingface_hub==0.26.5"
)


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=32768, timeout=1800)
def convert() -> dict:
    import hashlib
    import json
    from pathlib import Path

    import pandas as pd
    from huggingface_hub import hf_hub_download

    out = Path("/vol/routerbench")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ["routerbench_0shot.pkl", "routerbench_5shot.pkl"]:
        p = hf_hub_download("withmartian/routerbench", name, repo_type="dataset")
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        df = pd.read_pickle(p)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v if v is None or isinstance(v, str) else json.dumps(v, default=str))
        stem = name.removesuffix(".pkl")
        df.to_parquet(out / f"{stem}.parquet", index=False)
        summary[stem] = {
            "source_sha256": sha,
            "rows": int(len(df)),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "head": df.head(3).astype(str).to_dict(orient="records"),
        }
    (out / "SCHEMA.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(convert.remote(), indent=2)[:6000])
