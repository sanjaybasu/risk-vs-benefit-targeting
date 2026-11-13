"""Wave-2 loader: Turkey overdraft unshrouding experiment (Yapi Kredi, September 15, 2012 message).

Follows the posted specification notebooks/system-three/preregistration/wave2_specs/turkey_overdraft.md.
Reference arm: September 15 message with no overdraft mention (treatment 7-12 bill-pay discount only,
or 19-24 debit-card discount only). Action arm: overdraft interest-discount message (treatment 13-18,
25-30, 31-36; `message_fa`). Treatment 1-6 (availability reminder, no discount) is dropped.

Outcome `fausebal` replicates data8_maintables.do: months 2012m9-2012m12, missing `avgbalance` set to 0,
mean by account, 1 if the mean is below zero.

The loader reads a compact per-account parquet built once by build_turkey_parquet() from the deposited
.dta members (campaign_and_postcampaign_and_dailydata.dta and orthogonality.dta) extracted from Data.zip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

DIR = RAW / "wave2_turkey_overdraft"
EXTRACTED = DIR / "extracted"
PARQUET = EXTRACTED / "turkey_analysis.parquet"

INFO = DatasetInfo(
    name="w2_turkey_overdraft", tier="C", outcome_name="overdraft_used_sep_dec2012", better="lower",
    reference="no_overdraft_mention", actions=("no_overdraft_mention", "interest_discount"),
    license="CC0 1.0", commercial_use=True,
    citation="Alan S, Cemalcilar M, Karlan D, Zinman J. Unshrouding: evidence from bank overdrafts in "
    "Turkey. J Finance 2018;73(2). doi:10.1111/jofi.12593",
)
# Spec: implied discount vs no-mention contrast = Table 2 row 3 + row 4 = 0.0089 - 0.0124 = -0.0035
# (NBER WP 20956 Table 2 col 1; no SE reported for the sum). The tolerance is about 2 SE of a raw
# difference in means at these arm sizes (SE ~ 0.0031), fixed before the outcome was tabulated by arm.
PUBLISHED = {"source": "NBER WP 20956 Table 2 col 1, row 3 + row 4 (spec)", "arm": "interest_discount",
             "metric": "diff", "value": 0.0089 - 0.0124, "tol": 0.006}

P_ACTION = 0.6  # design probability among the two arms (spec: 53,953 / 89,957 = 0.5998, by design 0.6)

REF_T = list(range(7, 13)) + list(range(19, 25))
ACT_T = list(range(13, 19)) + list(range(25, 37))

# Pre-assignment covariates allowed by the spec (stratification, demographics, Sep 2011-Aug 2012 behavior).
X_COLS = ["female", "married", "marital_unknown", "istanbul", "ankara", "izmir", "bursa", "falimit", "limit",
          "acctbalance", "acctbaltreat", "assets", "deposits", "paymentmean", "debt", "transactions",
          "pastuse", "htefa", "minbal", "autobillpay", "creditcard", "credit", "phase1treat"]

PANEL_COLS = ["id", "date", "avgbalance", "faamount", "a_deposit", "treatment", "cinsiyet", "medenihal",
              "city", "istanbul", "ankara", "izmir", "bursa", "falimit", "limit", "acctbalance",
              "acctbaltreat", "phase1treat"]
ORTH_COLS = ["id", "treatment", "transactions", "assets", "deposits", "paymentmean", "debt", "pastuse",
             "autobillpay", "creditcard", "credit"]


def build_turkey_parquet() -> pd.DataFrame:
    """Build the one-row-per-account file from the extracted .dta members (run once)."""
    d = pd.read_stata(EXTRACTED / "campaign_and_postcampaign_and_dailydata.dta",
                      convert_categoricals=False, columns=PANEL_COLS)
    date = pd.to_datetime(d["date"])

    # Outcome, data8_maintables.do: group==1 = 2012m9-2012m12; missing avgbalance -> 0; mean by id; < 0.
    win = d[(date >= "2012-09-01") & (date <= "2012-12-01")]
    bal = win["avgbalance"].fillna(0.0).groupby(win["id"]).mean()
    out = pd.DataFrame({"avgbalance_sepdec": bal, "fausebal": (bal < 0).astype(int)})

    # htefa: any faamount > 0 in 2012m2-2012m8 (egen total(used), by(id); replace htefa = 1 if > 0).
    used = (d["faamount"] > 0) & (date >= "2012-02-01") & (date <= "2012-08-01")
    out["htefa"] = used.groupby(d["id"]).any().astype(int)
    # minbal: minimum a_deposit over months before 2012m9.
    pre = d[date < "2012-09-01"]
    out["minbal"] = pre.groupby("id")["a_deposit"].min()

    # Time-invariant assignment and stratification fields (constant within id in the deposited panel).
    base = d.groupby("id")[["treatment", "cinsiyet", "medenihal", "city", "istanbul", "ankara", "izmir",
                            "bursa", "falimit", "limit", "acctbalance", "acctbaltreat", "phase1treat"]].first()
    base = base.rename(columns={"cinsiyet": "female"})
    base["married"] = (base["medenihal"] == 1).astype(int)
    base["marital_unknown"] = (base["medenihal"] == 3).astype(int)
    base["acctbalance"] = base["acctbalance"].where(~base["acctbalance"].isin([-9999, -8888]))

    # Sep 2011-Aug 2012 baseline behavior, constructed in data7_orthogonality.do.
    o = pd.read_stata(EXTRACTED / "orthogonality.dta", convert_categoricals=False, columns=ORTH_COLS)
    o = o.set_index("id").rename(columns={"treatment": "treatment_orth"})

    t = base.join(out, how="left").join(o, how="left").reset_index()
    if not (t["treatment"] == t["treatment_orth"]).all():
        raise ValueError("treatment differs between panel and orthogonality file")
    t = t.drop(columns="treatment_orth")
    t.to_parquet(PARQUET, index=False)
    return t


def load_turkey() -> pd.DataFrame:
    if not PARQUET.exists():
        build_turkey_parquet()
    d = pd.read_parquet(PARQUET)
    d = d[d["treatment"].isin(REF_T + ACT_T)].reset_index(drop=True)
    act = np.where(d["treatment"].isin(ACT_T), "interest_discount", "no_overdraft_mention")
    # strat = group(female medenihal falimit city acctbaltreat), data8_maintables.do
    strat = (d[["female", "medenihal", "falimit", "city", "acctbaltreat"]].astype(int).astype(str)
             .agg("|".join, axis=1))
    df = frame("w2_turkey_overdraft", d[X_COLS], pd.Series(act), d["fausebal"], INFO,
               propensity=pd.Series(np.where(act == "interest_discount", P_ACTION, 1 - P_ACTION)),
               cluster=d["id"], stratum=strat, row_id=d["id"])
    return df
