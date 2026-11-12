"""Confirmatory-set expansion (2026-11-12): GUSTO-I and the San Francisco official-email trial."""

import numpy as np
import pandas as pd
import pyreadr

from system_three.loaders._base import RAW, _is_text, frame
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
        if _is_text(x[c]) or str(x[c].dtype) == "category":
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


# ---------------------------------------------------------------- ELAIA-1 and ELAIA-2 (AKI alerts)
def _elaia_info(k: int) -> DatasetInfo:
    cite = {1: "Wilson FP, Martin M, Yamamoto Y, et al. Electronic health record alerts for acute "
               "kidney injury: multicenter, randomized clinical trial. BMJ 2021;372:m4786. "
               "Data: Dryad doi:10.5061/dryad.4f4qrfj95",
            2: "Wilson FP, et al. A randomized clinical trial assessing the effect of automated "
               "medication-targeted alerts on acute kidney injury outcomes. Nat Commun 2023;14:2826. "
               "Data: Dryad doi:10.5061/dryad.kh189327p"}[k]
    return DatasetInfo(
        name=f"elaia{k}", tier="C", outcome_name="aki_progression_dialysis_or_death_14d",
        better="lower", reference="usual_care", actions=("usual_care", "aki_alert"),
        license="CC0-1.0 (Dryad)", commercial_use=True, citation=cite)


ELAIA1, ELAIA2 = _elaia_info(1), _elaia_info(2)
# BMJ 2021 abstract (PMID 33461986): composite 21.3% alert vs 20.9% usual care.
ELAIA1_PUBLISHED = {"source": "BMJ 2021 abstract: 21.3% vs 20.9%", "arm": "aki_alert",
                    "metric": "diff", "value": 0.004, "tol": 0.0015}
# Nat Commun 2023: composite 23.1% alert vs 25.3% usual care (RR 0.92, 0.83-1.01).
ELAIA2_PUBLISHED = {"source": "Nat Commun 2023 abstract: 23.1% vs 25.3%", "arm": "aki_alert",
                    "metric": "diff", "value": -0.022, "tol": 0.0015}
# Pre-randomization fields only (whitelist).
ELAIA1_X = ["hospital", "age", "sex", "race", "ethnicity", "admit_medical", "emergency_room", "icu",
            "ward", "chf", "copd", "diabetes", "hypertension", "liver_disease", "malignancy", "ckd_icd",
            "elx_score", "sofa", "baseline_creat", "initial_egfr", "creat_at_rand", "bun_at_rand",
            "anion_gap_at_rand", "bicarbonate_at_rand", "chloride_at_rand", "hemoglobin_at_rand",
            "plateletcount_at_rand", "potassium_at_rand", "sodium_at_rand", "wbcc_at_rand",
            "systolic_at_rand", "diastolic_at_rand", "pulse_at_rand", "resp_at_rand", "aki_to_rand",
            "time_to_rand", "acearbreninpre24", "aminopre24", "nsaidpre24", "prior_acearbrenin72",
            "prior_contraststudy72", "prior_ctsurgery7", "prior_nsaid72", "prior_ppi72", "bnp_flag"]
ELAIA2_X = ["hospital", "age", "sex", "icu_at_rand", "ward_at_rand", "er_at_rand", "num_med",
            "on_acearb", "on_nsaid", "on_ppi", "time_to_rand", "aki_to_rand", "aki_stage_at_rand",
            "admit_medical", "prior_cs72", "ckd_pmhx", "chf_elx_pmhx", "pulm_disease_elx_pmhx",
            "diabetes_elx_pmhx", "htn_elx_pmhx", "malignancy_elx_pmhx", "liverdisease_elx_pmhx",
            "elx_score_pmhx", "creat_at_rand", "mincreat48_prior", "mincreat7_prior",
            "admit_creatinine", "baseline_creat", "aniongap_at_rand", "bicarbonate_at_rand",
            "bun_at_rand", "chloride_at_rand", "egfr_at_admit", "hemoglobin_at_rand",
            "plateletcount_at_rand", "potassium_at_rand", "sodium_at_rand", "wbcc_at_rand",
            "systolic_at_rand", "diastolic_at_rand", "pulse_at_rand", "resp_at_rand", "spo2_at_rand",
            "temp_at_rand", "sofa_at_rand"]


def _load_elaia(k: int) -> pd.DataFrame:
    if k == 1:
        d = pd.read_csv(RAW / "elaia1" / "ELAIA-1_deidentified_data_10-6-2020.csv", low_memory=False)
        cols, info = ELAIA1_X, ELAIA1
    else:
        d = pd.read_csv(RAW / "elaia2" / "deidentified_elaia2_data_missing_NaN3.csv", low_memory=False)
        cols, info = ELAIA2_X, ELAIA2
    d = d[d["composite_outcome"].notna()].reset_index(drop=True)
    x = d[[c for c in cols if c in d.columns]].copy()
    for c in x.columns:
        if _is_text(x[c]):
            x[c] = x[c].astype("category").cat.codes.replace(-1, np.nan)
    # Individual 1:1 randomization, stratified by hospital (stratum).
    action = d["alert"].map({1: "aki_alert", 0: "usual_care"})
    return frame(info.name, x, action, d["composite_outcome"], info, propensity=0.5,
                 stratum=d["hospital"], row_id=d["id"])


def load_elaia1() -> pd.DataFrame:
    return _load_elaia(1)


def load_elaia2() -> pd.DataFrame:
    return _load_elaia(2)


# ---------------------------------------------------------------- Synbiotic neonatal sepsis (Odisha)
SYNBIOTIC = DatasetInfo(
    name="synbiotic", tier="C", outcome_name="sepsis_or_death_60d", better="lower",
    reference="placebo", actions=("placebo", "synbiotic"), license="CC0-1.0 (Dryad)",
    commercial_use=True,
    citation="Panigrahi P, Parida S, Nanda NC, et al. A randomized synbiotic trial to prevent sepsis "
    "among infants in rural India. Nature 2017;548:407-412. Data: Dryad doi:10.5061/dryad.275d4",
)
# Nature 2017: sepsis or death 9.0% placebo vs 5.4% synbiotic (RR 0.60).
SYNBIOTIC_PUBLISHED = {"source": "Nature 2017: RR 0.60 (0.48-0.74)", "arm": "synbiotic",
                       "metric": "relative_change", "value": -0.40, "tol": 0.02}
SYN_X = ["Sex_of_Infant", "BWT", "GA", "MOMAGE", "MOMPAR", "Twins", "DELIV", "DELIV_MODE",
         "Delivered_by", "Weight_Enroll", "POPBLOCK12"]


def load_synbiotic() -> pd.DataFrame:
    d = pd.read_csv(RAW / "synbiotic" / "Panigrahi+et+al.SEPSSIF.CSV", low_memory=False,
                    encoding="latin-1")
    d = d[d["SEPSISDEATH"].notna() & d["TX"].notna()].reset_index(drop=True)
    x = d[[c for c in SYN_X if c in d.columns]].copy()
    for c in x.columns:
        if _is_text(x[c]):
            x[c] = x[c].astype("category").cat.codes.replace(-1, np.nan)
    t = pd.to_numeric(d["TX"], errors="coerce")
    action = pd.Series(np.where(t == 1, "synbiotic", "placebo"))
    # Two Study_IDs appear twice with identical records; the published N (4,556) counts all rows.
    # Rows keep their own id; Study_ID is the cluster so duplicates share a split.
    return frame("synbiotic", x, action, pd.to_numeric(d["SEPSISDEATH"]), SYNBIOTIC,
                 propensity=0.5, stratum=d["POPBLOCK12"], cluster=d["Study_ID"],
                 row_id=d["Study_ID"] + "_" + d.index.astype(str))


# ---------------------------------------------------------------- Karlan-List matching grants
KARLAN = DatasetInfo(
    name="karlan_list", tier="B", outcome_name="gave", better="higher", reference="no_match",
    actions=("no_match", "matching_grant"), license="CC0-1.0 (IPA Dataverse, guestbook)",
    commercial_use=True,
    citation="Karlan D, List JA. Does price matter in charitable giving? Evidence from a large-scale "
    "natural field experiment. Am Econ Rev 2007;97:1774-1793. Data: doi:10.7910/DVN/27853",
)
KARLAN_PUBLISHED = {"source": "AER 2007 Table 2 response rate difference [value UNVERIFIED]",
                    "arm": "matching_grant", "metric": "diff", "value": None, "tol": None}
# Pre-treatment fields. stateresponse* are excluded: they aggregate this experiment's outcomes.
KARLAN_X = ["hpa", "ltmedmra", "freq", "years", "year5", "mrm2", "dormant", "female", "couple",
            "nonlit", "cases", "statecnt", "perbush", "close25", "red0", "blue0", "redcty", "bluecty",
            "pwhite", "pblack", "page18_39", "ave_hh_sz", "median_hhincome", "powner",
            "psch_atlstba", "pop_propurban"]


def load_karlan_list() -> pd.DataFrame:
    d = pd.read_stata(RAW / "karlan_list" / "AER merged.dta", convert_categoricals=False)
    p = 2 / 3  # design: two-thirds of letters assigned to a matching-grant arm (F66)
    action = d["treatment"].map({1: "matching_grant", 0: "no_match"})
    prop = np.where(d["treatment"] == 1, p, 1 - p)
    return frame("karlan_list", d[KARLAN_X], action, d["gave"], KARLAN, propensity=pd.Series(prop))


# ---------------------------------------------------------------- capped expansion (DECISIONS 2026-11-12)
def _stratum_prop(treated: pd.Series, stratum: pd.Series) -> np.ndarray:
    p = treated.groupby(stratum).transform("mean")
    return np.where(treated, p, 1 - p)


SIXCITY = DatasetInfo(
    name="sixcity", tier="B", outcome_name="voted_2001", better="higher", reference="control",
    actions=("control", "canvass"), license="CC0-1.0", commercial_use=True,
    citation="Green DP, Gerber AS, Nickerson DW. Getting out the vote in local elections: results "
    "from six door-to-door canvassing experiments. J Polit 2003;65:1083-1096. "
    "Data: Yale Dataverse doi:10.60600/YU/B9TZNZ",
)
SIXCITY_PUBLISHED = {"source": "J Polit 2003 pooled ITT [value UNVERIFIED]", "arm": "canvass",
                     "metric": "diff_strata", "value": None, "tol": None}


def load_sixcity() -> pd.DataFrame:
    """Individuals within households; assignment within city x turf (stratum), so the design
    probability is the within-stratum treated share; clusters are households (city + family)."""
    s = pd.read_stata(RAW / "sixcity" / "data.tab", convert_categoricals=False)
    s = s[s["voted01"].notna()].reset_index(drop=True)
    x = pd.DataFrame({
        "age": s["age"], "voted00": s["voted00"], "voted99": s["voted99"], "famsize": s["famsize"],
        "primary": pd.to_numeric(s["primary"], errors="coerce"),
        "female": (s["sex"] == "F").astype(float), "male": (s["sex"] == "M").astype(float),
        "dem": (s["party"] == "D").astype(float), "rep": (s["party"] == "R").astype(float),
        "black": (s["race"] == "B").astype(float), "white": (s["race"] == "W").astype(float),
    })
    for c in sorted(s["city"].unique()):
        x[f"city_{c}"] = (s["city"] == c).astype(float)
    stratum = s["city"].astype(str) + "_" + s["turf"].astype(str)
    # Strata assigned entirely to one arm carry no randomized contrast; they are dropped.
    share = (s["treatmen"] == 1).groupby(stratum).transform("mean")
    keep = (share > 0) & (share < 1)
    s, x, stratum = s[keep].reset_index(drop=True), x[keep].reset_index(drop=True), stratum[keep].reset_index(drop=True)
    treated = s["treatmen"] == 1
    action = pd.Series(np.where(treated, "canvass", "control"))
    return frame("sixcity", x, action, s["voted01"], SIXCITY,
                 propensity=pd.Series(_stratum_prop(treated, stratum)), stratum=stratum,
                 cluster=s["city"].astype(str) + "_" + s["family"].astype(str))


NEWHAVEN = DatasetInfo(
    name="newhaven", tier="B", outcome_name="voted_1998", better="higher", reference="control",
    actions=("control", "canvass"), license="CC0-1.0", commercial_use=True,
    citation="Gerber AS, Green DP. The effects of canvassing, telephone calls, and direct mail on "
    "voter turnout: a field experiment. Am Polit Sci Rev 2000;94:653-663; correction APSR "
    "2005;99:301-313. Data: Yale Dataverse doi:10.60600/YU/ODL44F",
)
NEWHAVEN_PUBLISHED = {"source": "APSR 2000/2005 canvass ITT [value UNVERIFIED]", "arm": "canvass",
                      "metric": "diff_strata", "value": None, "tol": None}


def load_newhaven() -> pd.DataFrame:
    """Factorial (canvass x mail x phone), households randomized. Contrast: in-person canvass vs
    none, marginal over the independently randomized mail and phone factors. Stratum: persons
    per household; clusters: household (id1)."""
    n = pd.read_stata(RAW / "newhaven" / "NHrep_individual.tab", convert_categoricals=False)
    n = n[n["v98"].notna()].reset_index(drop=True)
    x = n[["persons", "age", "agemiss", "majorpty", "v96_1", "v96_0", "ward"]].copy()
    treated = n["persngrp"] == 1
    action = pd.Series(np.where(treated, "canvass", "control"))
    return frame("newhaven", x, action, n["v98"], NEWHAVEN,
                 propensity=pd.Series(_stratum_prop(treated, n["persons"])),
                 stratum=n["persons"], cluster=n["id1"])


LENTA = DatasetInfo(
    name="lenta", tier="B", outcome_name="visit_after_sms", better="higher", reference="control",
    actions=("control", "sms"), license="not stated (BigTarget hackathon data via scikit-uplift)",
    commercial_use=False,
    citation="Lenta BigTarget uplift dataset (2020), distributed by scikit-uplift "
    "(sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz).",
)
LENTA_PUBLISHED = {"source": "no published estimate", "arm": "sms", "metric": "treated_share",
                   "value": None, "tol": None}


def load_lenta() -> pd.DataFrame:
    d = pd.read_csv(RAW / "lenta" / "lenta_dataset.csv.gz", low_memory=False)
    x = d.drop(columns=["group", "response_att", "response_sms", "response_viber"])
    x["gender"] = x["gender"].map({"Ж": 1.0, "М": 0.0}) if "gender" in x else np.nan
    treated = d["group"] == "test"
    p = 0.75  # observed assignment share (515,892 of 687,029); no published design share
    action = pd.Series(np.where(treated, "sms", "control"))
    return frame("lenta", x, action, d["response_att"], LENTA,
                 propensity=pd.Series(np.where(treated, p, 1 - p)))


X5 = DatasetInfo(
    name="x5", tier="B", outcome_name="purchase_after_sms", better="higher", reference="control",
    actions=("control", "sms"), license="not stated (X5 RetailHero 2019 competition data via "
    "scikit-uplift)", commercial_use=False,
    citation="X5 Retail Hero uplift modelling competition (2019), distributed by scikit-uplift.",
)
X5_PUBLISHED = {"source": "no published estimate", "arm": "sms", "metric": "treated_share",
                "value": None, "tol": None}


def load_x5() -> pd.DataFrame:
    """Clients randomized 1:1 to an SMS. Covariates: client file plus aggregates of the purchase
    history, which the competition documents as recorded before the communication."""
    u = pd.read_csv(RAW / "x5" / "uplift_train.csv.gz")
    c = pd.read_csv(RAW / "x5" / "clients.csv.gz")
    agg_path = RAW / "x5" / "purchase_features.parquet"
    if not agg_path.exists():
        cols = ["client_id", "transaction_id", "transaction_datetime", "purchase_sum",
                "regular_points_received", "express_points_spent", "regular_points_spent"]
        it = pd.read_csv(RAW / "x5" / "purchases.csv.gz", usecols=cols, chunksize=2_000_000)
        parts = []
        for ch in it:
            ch = ch.drop_duplicates(["client_id", "transaction_id"])
            parts.append(ch.groupby("client_id").agg(
                n_tx=("transaction_id", "nunique"), spend=("purchase_sum", "sum"),
                points_recv=("regular_points_received", "sum"),
                points_spent=("regular_points_spent", "sum"),
                express_spent=("express_points_spent", "sum"),
                last_tx=("transaction_datetime", "max"), first_tx=("transaction_datetime", "min")))
        a = pd.concat(parts).groupby(level=0).agg(
            n_tx=("n_tx", "sum"), spend=("spend", "sum"), points_recv=("points_recv", "sum"),
            points_spent=("points_spent", "sum"), express_spent=("express_spent", "sum"),
            last_tx=("last_tx", "max"), first_tx=("first_tx", "min"))
        for col in ("last_tx", "first_tx"):
            a[col] = pd.to_datetime(a[col]).astype("int64") / 8.64e13
        a.to_parquet(agg_path)
    a = pd.read_parquet(agg_path)
    d = u.merge(c, on="client_id", how="left").merge(a, left_on="client_id", right_index=True, how="left")
    x = pd.DataFrame({
        "age": d["age"], "gender_f": (d["gender"] == "F").astype(float),
        "gender_m": (d["gender"] == "M").astype(float),
        "first_issue": pd.to_datetime(d["first_issue_date"]).astype("int64") / 8.64e13,
        "first_redeem": pd.to_datetime(d["first_redeem_date"], errors="coerce").astype("int64") / 8.64e13,
        "n_tx": d["n_tx"], "spend": d["spend"], "points_recv": d["points_recv"],
        "points_spent": d["points_spent"], "express_spent": d["express_spent"],
        "last_tx": d["last_tx"], "first_tx": d["first_tx"],
    })
    x.loc[d["first_redeem_date"].isna(), "first_redeem"] = np.nan
    treated = d["treatment_flg"] == 1
    action = pd.Series(np.where(treated, "sms", "control"))
    return frame("x5", x, action, d["target"], X5, propensity=0.5, row_id=d["client_id"])


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
