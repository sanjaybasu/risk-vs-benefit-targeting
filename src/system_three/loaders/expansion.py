"""Confirmatory-set expansion (2026-11-12): GUSTO-I and the San Francisco official-email trial."""

import numpy as np
import pandas as pd
import pyreadr

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

# ---------------------------------------------------------------- GUSTO-I
GUSTO = DatasetInfo(
    name="gusto", tier="C", outcome_name="death_30d", better="lower", reference="streptokinase",
    actions=("streptokinase", "accelerated_tpa"),
    license="GPL (CRAN predtools 0.0.3; data source listed as 'Internet'); counsel to confirm",
    commercial_use=False,
    citation="GUSTO Investigators. An international randomized trial comparing four thrombolytic "
    "strategies for acute myocardial infarction. N Engl J Med 1993;329:673-682. "
    "doi:10.1056/NEJM199309023291001",
)
# Abstract: 14% relative reduction in 30-day mortality, accelerated t-PA vs both streptokinase arms.
GUSTO_PUBLISHED = {"source": "NEJM 1993 abstract, 14% relative reduction", "arm": "accelerated_tpa",
                   "metric": "relative_change", "value": -0.14, "tol": 0.01}
# Randomized 1:1:1:1 to four arms; the two streptokinase arms are pooled in this file, so within
# the tPA-vs-SK comparison P(tPA) = 1/3. The SK+tPA combination arm is excluded.
GUSTO_X = ["sho", "hig", "dia", "hyp", "hrt", "sex", "Killip", "age", "ste", "pulse", "sysbp", "ant",
           "miloc", "height", "weight", "pmi", "htn", "smk", "pan", "fam", "prevcvd", "prevcabg",
           "regl", "grpl", "grps"]  # ttr (time to treatment) excluded: measured after allocation


def load_gusto() -> pd.DataFrame:
    d = pyreadr.read_r(RAW / "gusto" / "predtools" / "data" / "gusto.rda")["gusto"]
    d = d[d["tx"].isin(["tPA", "SK"])].reset_index(drop=True)
    x = d[GUSTO_X].copy()
    for c in x.columns:
        if x[c].dtype == object or str(x[c].dtype) == "category":
            x[c] = x[c].astype("category").cat.codes.replace(-1, np.nan)
    action = d["tx"].map({"tPA": "accelerated_tpa", "SK": "streptokinase"})
    prop = np.where(action == "accelerated_tpa", 1 / 3, 2 / 3)
    return frame("gusto", x, action, d["day30"], GUSTO, propensity=pd.Series(prop))


# ---------------------------------------------------------------- StepUp gym megastudy
STEPUP = DatasetInfo(
    name="stepup", tier="B", outcome_name="any_gym_visit_wk1_4", better="higher",
    reference="placebo_control", actions=("placebo_control", "behavioral_program"),
    license="no license on the OSF node; deidentified data shared with 24 Hour Fitness permission "
    "via the view-only link in the paper's data statement", commercial_use=False,
    citation="Milkman KL, Gromet D, Ho H, et al. Megastudies improve the impact of applied "
    "behavioural science. Nature 2021;600:478-483. doi:10.1038/s41586-021-04128-4",
)
STEPUP_PUBLISHED = {"source": "Nature 2021: 61,293 participants analyzed", "arm": "behavioral_program",
                    "metric": "n_units", "value": 61293, "tol": 0}


def load_stepup() -> pd.DataFrame:
    """One row per participant. Pre-specified action: any of the 53 programs vs Placebo Control
    (pooled; no arm selection on outcomes). Outcome: any visit during intervention weeks 1-4.
    Covariates: demographics and gym visits in weeks -52 to 0. Design propensity: the
    within-cohort share assigned to a program (randomization was within cohort)."""
    import zipfile

    z = zipfile.ZipFile(RAW / "stepup" / "Data_April21.zip")
    cols = ["participant_id", "week", "visits", "any_visit", "cohort", "age", "signup_date_local",
            "customer_state", "gender", "exp_condition", "new_member"]
    w = pd.read_csv(z.open("Data/StepUp Data/pptdata.csv"), usecols=cols)
    pre = w[(w["week"] >= -52) & (w["week"] <= 0)]
    during = w[(w["week"] >= 1) & (w["week"] <= 4)]
    g = pre.groupby("participant_id")
    feats = pd.DataFrame({
        "pre_weeks_obs": g["week"].size(),
        "pre_any_visit_rate": g["any_visit"].mean(),
        "pre_visits_per_week": g["visits"].mean(),
        "pre_any_visit_last4": pre[pre["week"] >= -3].groupby("participant_id")["any_visit"].mean(),
        "pre_any_visit_last12": pre[pre["week"] >= -11].groupby("participant_id")["any_visit"].mean(),
    })
    y = during.groupby("participant_id")["any_visit"].max().rename("y")
    base = w.drop_duplicates("participant_id").set_index("participant_id")
    d = base.join(feats).join(y, how="inner").reset_index()
    signup = pd.to_datetime(d["signup_date_local"], errors="coerce")
    x = feats.reindex(d["participant_id"]).reset_index(drop=True)
    x["age"] = d["age"]
    x["male"] = (d["gender"] == "M").astype(float)
    x["new_member"] = d["new_member"]
    x["signup_year"] = signup.dt.year + signup.dt.dayofyear / 365.25
    top_states = d["customer_state"].value_counts().index[:8]
    for s in top_states:
        x[f"state_{s}"] = (d["customer_state"] == s).astype(float)
    treated = d["exp_condition"] != "Placebo Control"
    p_c = treated.groupby(d["cohort"]).transform("mean")
    prop = np.where(treated, p_c, 1 - p_c)
    action = pd.Series(np.where(treated, "behavioral_program", "placebo_control"))
    return frame("stepup", x, action, d["y"], STEPUP, propensity=pd.Series(prop),
                 stratum=d["cohort"], row_id=d["participant_id"])


# ---------------------------------------------------------------- San Francisco official email, June 2010
SF_ARMS = {0: "control", 1: "registrar_email", 2: "advocacy_email"}
SF_ARM_PROBS = {a: 1 / 3 for a in SF_ARMS.values()}
SF_EMAIL = DatasetInfo(
    name="sf_email", tier="B", outcome_name="voted_jun2010", better="higher", reference="control",
    actions=tuple(SF_ARMS.values()), license="CC0-1.0", commercial_use=True,
    citation="Malhotra N, Michelson MR, Valenzuela AA. Emails from official sources can increase "
    "turnout. Q J Polit Sci 2012;7:321-332. Data: Yale Dataverse doi:10.60600/YU/SCUYUK.",
)
SF_PUBLISHED = {"source": "QJPS 2012, June 2010 registrar email ITT [value UNVERIFIED]",
                "arm": "registrar_email", "metric": "diff", "value": None, "tol": None}


def load_sf_email() -> pd.DataFrame:
    post = pd.read_stata(RAW / "sf_email" / "June10post", convert_categoricals=False)
    trt = pd.read_stata(RAW / "sf_email" / "june10_treatments", convert_categoricals=False)
    d = trt[trt["treatment"].notna()].merge(post, on="voterid", how="inner", validate="1:1")
    d = d.reset_index(drop=True)
    x = pd.DataFrame({
        "age": pd.to_numeric(d["voterage"], errors="coerce"),
        "reg_year": pd.to_numeric(d["registrationdate"].astype(str).str[-4:], errors="coerce"),
        "perm_vbm": (d["avstatus"] == "PermVBM").astype(float),
        "dem": (d["partyname"] == "Democratic").astype(float),
        "rep": (d["partyname"] == "Republican").astype(float),
        "dts": (d["partyname"] == "Non-Decline To State").astype(float),
        "us_born": d["birthplace"].astype(str).str.len().between(2, 2).astype(float),
    })
    for k in range(1, 6):  # pre-election vote history from the treatment file
        x[f"vhist{k}"] = d[f"vhist{k}"].astype(str).str.strip().ne("").astype(float)
    y = (d["electiondate_5"].astype(str).str.startswith("2010-06-08")).astype(float)
    action = d["treatment"].astype(int).map(SF_ARMS)
    df = frame("sf_email", x, action, y, SF_EMAIL, propensity=action.map(SF_ARM_PROBS),
               cluster=d["caseid_first"], row_id=d["voterid"])
    df.attrs["arm_probs"] = SF_ARM_PROBS
    return df
