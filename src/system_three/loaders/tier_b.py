"""Tier B loaders: Hillstrom email, Criteo uplift, NSW (Dehejia-Wahba), IHDP."""

import numpy as np
import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

# ---------------------------------------------------------------- Hillstrom
HILL_ARMS = {"No E-Mail": "no_email", "Mens E-Mail": "mens_email", "Womens E-Mail": "womens_email"}
HILLSTROM = DatasetInfo(
    name="hillstrom", tier="B", outcome_name="visit_2wk", better="higher", reference="no_email",
    actions=tuple(HILL_ARMS.values()), license="not stated (public challenge dataset)",
    commercial_use=False,
    citation="Hillstrom K. The MineThatData E-Mail Analytics and Data Mining Challenge. 2008.",
)
HILLSTROM_PUBLISHED = {"source": "challenge data summary: visit 18.28% mens vs 10.62% none",
                       "arm": "mens_email", "metric": "diff", "value": 0.0766, "tol": 0.0005}


def load_hillstrom() -> pd.DataFrame:
    h = pd.read_csv(RAW / "hillstrom" / "hillstrom.csv.gz")
    x = pd.concat([h[["recency", "history", "mens", "womens", "newbie"]],
                   pd.get_dummies(h[["zip_code", "channel"]], dtype=float)], axis=1)
    action = h["segment"].map(HILL_ARMS)
    df = frame("hillstrom", x, action, h["visit"], HILLSTROM, propensity=1 / 3)
    df.attrs["arm_probs"] = {a: 1 / 3 for a in HILL_ARMS.values()}
    return df


# ---------------------------------------------------------------- Criteo
CRITEO = DatasetInfo(
    name="criteo", tier="B", outcome_name="visit", better="higher", reference="no_ad",
    actions=("no_ad", "ad"), license="CC-BY-NC-SA-4.0", commercial_use=False,
    citation="Diemert E, Betlei A, Renaudin C, Amini MR. A large scale benchmark for uplift "
    "modeling. AdKDD 2018. Data v2.1.",
)
CRITEO_P = 0.85  # fixed treatment ratio of the randomized design
CRITEO_PUBLISHED = {"source": "design treatment share 0.85 (dataset card)", "arm": "ad",
                    "metric": "treated_share", "value": 0.85, "tol": 0.001}


def load_criteo() -> pd.DataFrame:
    c = pd.read_csv(RAW / "criteo" / "criteo-research-uplift-v2.1.csv.gz",
                    dtype={f"f{i}": "float32" for i in range(12)})
    action = c["treatment"].map({1: "ad", 0: "no_ad"})
    prop = np.where(c["treatment"] == 1, CRITEO_P, 1 - CRITEO_P)
    return frame("criteo", c[[f"f{i}" for i in range(12)]], action, c["visit"], CRITEO,
                 propensity=pd.Series(prop))


# ---------------------------------------------------------------- NSW (Dehejia-Wahba subsample)
NSW = DatasetInfo(
    name="nsw", tier="B", outcome_name="earnings_1978_usd", better="higher", reference="control",
    actions=("control", "job_training"), license="not stated (author-hosted)", commercial_use=False,
    citation="LaLonde RJ. Am Econ Rev 1986;76:604-620. Dehejia RH, Wahba S. J Am Stat Assoc "
    "1999;94:1053-1062.",
)
NSW_PUBLISHED = {"source": "Dehejia-Wahba experimental benchmark $1,794", "arm": "job_training",
                 "metric": "diff", "value": 1794.0, "tol": 2.0}
NSW_COLS = ["treat", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "re78"]


def load_nsw() -> pd.DataFrame:
    parts = [pd.read_csv(RAW / "nsw" / f, sep=r"\s+", header=None, names=NSW_COLS)
             for f in ("nswre74_treated.txt", "nswre74_control.txt")]
    d = pd.concat(parts, ignore_index=True)
    p = d["treat"].mean()  # design share in the Dehejia-Wahba subsample (185 of 445)
    action = d["treat"].map({1: "job_training", 0: "control"})
    prop = np.where(d["treat"] == 1, p, 1 - p)
    return frame("nsw", d[NSW_COLS[1:-1]], action, d["re78"], NSW, propensity=pd.Series(prop))


# ---------------------------------------------------------------- IHDP (semi-synthetic, NPCI setting A)
IHDP = DatasetInfo(
    name="ihdp", tier="B", outcome_name="simulated_score", better="higher", reference="control",
    actions=("control", "home_visits"), license="not stated (author-hosted)", commercial_use=False,
    citation="Hill JL. J Comput Graph Stat 2011;20:217-240. Files: Johansson, Shalit, Sontag, "
    "ICML 2016 (fredjo.com).",
)
IHDP_PUBLISHED = {"source": "npz 'ate' field equals mean(mu1 - mu0), replication 1",
                  "arm": "home_visits", "metric": "ihdp_ate", "value": None, "tol": 0.05}


def load_ihdp(n_reps: int = 100) -> pd.DataFrame:
    """Rows: replication x unit. Truth is exact (y_ref = mu0, y_act = mu1, noise-free means).

    Treatment in IHDP is not randomized (a non-random subset of treated units was removed), so
    propensity is unknown (NaN); scoring uses the exact truth. Native split column: train or test.
    """
    out = []
    for part in ("train", "test"):
        z = np.load(RAW / "ihdp" / f"ihdp_npci_1-100.{part}.npz")
        for r in range(n_reps):
            x = pd.DataFrame(z["x"][:, :, r], columns=[f"v{i + 1}" for i in range(25)])
            t = z["t"][:, r].astype(int)
            uid = pd.Series([f"r{r + 1}_{part}_{i}" for i in range(len(t))])
            df = frame("ihdp", x, pd.Series(np.where(t == 1, "home_visits", "control")),
                       pd.Series(z["yf"][:, r]), IHDP, propensity=np.nan, row_id=uid,
                       cluster=uid, stratum=pd.Series([f"rep{r + 1}"] * len(t)),
                       y_ref=pd.Series(z["mu0"][:, r]), y_act=pd.Series(z["mu1"][:, r]))
            df["native_split"] = part
            out.append(df)
    return pd.concat(out, ignore_index=True)
