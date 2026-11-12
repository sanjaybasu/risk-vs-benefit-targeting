"""Tier C loaders: Illinois Workplace Wellness, Oregon Health Insurance Experiment (ED sample)."""

import io
import zipfile

import numpy as np
import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

# ---------------------------------------------------------------- Illinois
ILLINOIS = DatasetInfo(
    name="illinois", tier="C", outcome_name="spend_month_y1_usd", better="lower",
    reference="control", actions=("control", "wellness_program"),
    license="CC0-1.0 plus README terms of use: no re-identification of subjects",
    commercial_use=True,
    citation="Jones D, Molitor D, Reif J. What do workplace wellness programs do? Evidence from the "
    "Illinois Workplace Wellness Study. Q J Econ 2019;134:1747-1791. Data doi:10.5281/zenodo.4064860.",
)
# QJE 2019 replication, Appendix Table 4a, column 2 (no controls): ITT on total spending
# (dollars/month), Aug 2016 to Jul 2017, 10.8 (SE 48.5), N = 3,239; the regression is weighted
# by months of coverage (aw = covg_0816_0717), stored here as aux_weight.
ILLINOIS_PUBLISHED = {"source": "QJE 2019 replication Appendix Table 4a, col 2",
                      "arm": "wellness_program", "metric": "diff_aux_weight", "value": 10.8,
                      "tol": 0.06}


def load_illinois() -> pd.DataFrame:
    d = pd.read_csv(RAW / "illinois" / "repo" / "data" / "csv" / "claims.csv")
    d = d[d["spend_0816_0717"].notna()].reset_index(drop=True)
    pre = [c for c in d.columns if c.endswith(("_0715_0716", "_1015_0716"))]
    x = d[["male", "age50", "age37_49", "white"] + pre]
    # Randomization was stratified; design probability is the within-stratum treated share.
    p_s = d.groupby("Strata_claims")["treat"].transform("mean")
    prop = np.where(d["treat"] == 1, p_s, 1 - p_s)
    action = d["treat"].map({1: "wellness_program", 0: "control"})
    df = frame("illinois", x, action, d["spend_0816_0717"], ILLINOIS,
               propensity=pd.Series(prop), stratum=d["Strata_claims"])
    df["aux_weight"] = d["covg_0816_0717"].to_numpy()
    return df


# ---------------------------------------------------------------- Oregon (ED sample, Portland)
OREGON = DatasetInfo(
    name="oregon", tier="C", outcome_name="any_ed_visit", better="lower",
    reference="not_selected", actions=("not_selected", "lottery_selected"),
    license="NBER click-through terms: no commercial use, no re-identification, cite",
    commercial_use=False,
    citation="Taubman SL, Allen HL, Wright BJ, Baicker K, Finkelstein AN. Medicaid increases "
    "emergency-department use: evidence from Oregon's Health Insurance Experiment. Science "
    "2014;343:263-268.",
)
# Intent-to-treat of lottery selection on any ED visit; value not yet verified against the
# source table, so the dataset table reports it as unverified.
OREGON_PUBLISHED = {"source": "Taubman et al. 2014 Table 1 ITT, any ED visit [UNVERIFIED]",
                    "arm": "lottery_selected", "metric": "diff_strata", "value": None, "tol": None}


def _dta(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    f = [m for m in z.namelist() if m.endswith(name) and "MACOSX" not in m][0]
    return pd.read_stata(io.BytesIO(z.read(f)), convert_categoricals=False)


def load_oregon() -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "oregon" / "oregon_puf.zip")
    desc = _dta(z, "oregonhie_descriptive_vars.dta")
    ed = _dta(z, "oregonhie_ed_vars.dta")
    d = ed[ed["sample_ed"] == 1].merge(desc, on="person_id", how="inner", validate="1:1")
    d = d[d["any_visit_ed"].notna()].reset_index(drop=True)
    pre = [c for c in ed.columns if "_pre_" in c or c.endswith("_pre_ed")]
    lists = ["numhh_list", "birthyear_list", "english_list", "female_list", "self_list",
             "first_day_list", "pobox_list", "have_phone_list"]
    x = d[pre + lists].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    # Selection probability depends on the number of household members on the list.
    p_s = d.groupby("numhh_list")["treatment"].transform("mean")
    prop = np.where(d["treatment"] == 1, p_s, 1 - p_s)
    action = d["treatment"].map({1: "lottery_selected", 0: "not_selected"})
    return frame("oregon", x, action, d["any_visit_ed"], OREGON, propensity=pd.Series(prop),
                 cluster=d["household_id"], stratum=d["numhh_list"], row_id=d["person_id"])
