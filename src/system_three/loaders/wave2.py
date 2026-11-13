"""Wave-2 loaders (registration PREREGISTRATION_WAVE2.md; per-trial specifications in OSF wave2/).

Each loader follows its posted specification (notebooks/system-three/preregistration/wave2_specs/).
Dataset names carry the prefix w2_, which selects the wave-2 split salt in splits.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo


def _dta(path, **kw):
    return pd.read_stata(path, convert_categoricals=False, **kw)


# ---------------------------------------------------------------- Michigan 2008 robocall
ROBOCALL = DatasetInfo(
    name="w2_robocall2008", tier="C", outcome_name="voted_aug2008_primary", better="higher",
    reference="no_call", actions=("no_call", "gotv_robocall"), license="CC0 1.0", commercial_use=True,
    citation="Gerber AS, Green DP, Kaplan EH, Kern HL. Baseline, placebo, and treatment: efficient "
    "estimation for three-group experiments. Polit Anal 2010;18:297-315. doi:10.1093/pan/mpq008",
)
# ITT from Table 4 counts: 1,508/8,448 vs 51,766/304,948.
ROBOCALL_PUBLISHED = {"source": "Polit Anal 2010 Table 4 counts", "arm": "gotv_robocall", "metric": "diff",
                      "value": 1508 / 8448 - 51766 / 304948, "tol": 0.002}


def load_robocall() -> pd.DataFrame:
    d = _dta(RAW / "wave2_robocall2008" / "robocall.orig")
    d = d[(d["hhcount"] == 1) & d["p2008_fi"].notna() & d["treatmen"].astype(str).isin(["0", "1"])].reset_index(drop=True)
    act = np.where(d["treatmen"].astype(str) == "1", "gotv_robocall", "no_call")
    p = float(np.mean(act == "gotv_robocall"))
    x = d[["g2002", "p2002", "p2004", "av", "pp2008"]].apply(pd.to_numeric, errors="coerce")
    return frame("w2_robocall2008", x, pd.Series(act), d["p2008_fi"], ROBOCALL,
                 propensity=pd.Series(np.where(act == "gotv_robocall", p, 1 - p)), row_id=d["id"])


# ---------------------------------------------------------------- San Mateo registrar email, June 2010
SANMATEO = DatasetInfo(
    name="w2_sanmateo_email", tier="C", outcome_name="voted_jun2010", better="higher", reference="no_email",
    actions=("no_email", "registrar_email"), license="CC0 1.0", commercial_use=True,
    citation="Malhotra N, Michelson MR, Valenzuela AA. Emails from official sources can increase turnout. "
    "Q J Polit Sci 2012;7:321-332. doi:10.1561/100.00011073",
)
SANMATEO_PUBLISHED = {"source": "authors' meta-analysis sheet, study B (June 2010)", "arm": "registrar_email",
                      "metric": "diff", "value": 0.0052462, "tol": 0.003}
US = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
         "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY US PR GU VI AS".split())


def load_sanmateo() -> pd.DataFrame:
    t = _dta(RAW / "wave2_sanmateo_email" / "june10_treatments.orig")
    p = _dta(RAW / "wave2_sanmateo_email" / "june10_post.orig")
    t = t[t["treatment"].isin([0, 1])]
    d = t.merge(p, on="voterid", how="left").reset_index(drop=True)
    y = (d["electiondate_5"].astype(str) == "2010-06-08 00:00:00").astype(float)
    reg = pd.to_datetime(d["registrationdate"], format="%m/%d/%Y", errors="coerce")
    bp = d["birthplace"].astype(str).str.strip()
    x = pd.DataFrame({"age": pd.to_numeric(d["voterage"], errors="coerce"),
                      "yrsreg": 2010 - reg.dt.year,
                      "forborn": np.where(bp.isin(["", "nan"]), np.nan, (~bp.isin(US)).astype(float))})
    for k in range(1, 6):
        x[f"vhist{k}"] = pd.to_numeric(d[f"vhist{k}"], errors="coerce")
    x["hh_size"] = d.groupby("caseid_first")["voterid"].transform("size")
    act = np.where(d["treatment"] == 1, "registrar_email", "no_email")
    pr = float(np.mean(act == "registrar_email"))
    return frame("w2_sanmateo_email", x, pd.Series(act), y, SANMATEO,
                 propensity=pd.Series(np.where(act == "registrar_email", pr, 1 - pr)),
                 cluster=d["caseid_first"], row_id=d["voterid"])


# ---------------------------------------------------------------- Wisconsin 2012 recall ballot secrecy
WISCONSIN = DatasetInfo(
    name="w2_wisconsin_recall", tier="C", outcome_name="voted_jun2012_recall", better="higher",
    reference="no_letter", actions=("no_letter", "ballot_secrecy_letter"), license="CC0 1.0", commercial_use=True,
    citation="Gerber AS, Huber GA, Biggers DR, Hendry DJ. Ballot secrecy concerns and voter mobilization. "
    "Am Politics Res 2014;42:896-923. doi:10.1177/1532673X14524269",
)
WISCONSIN_PUBLISHED = {"source": "APR 2014 Table 1 col 1", "arm": "ballot_secrecy_letter", "metric": "diff",
                       "value": 0.011, "tol": 0.006}
WI_X = ["yearssincereg", "age_corrected", "d_miss_age", "female", "d_miss_gender", "hhinc_corrected",
        "d_miss_hhinc", "asian", "black", "latino", "mideastern", "nativeamer", "d_miss_race", "milwaukee"]


def load_wisconsin() -> pd.DataFrame:
    d = _dta(RAW / "wave2_wisconsin_recall" / "wi_ballot_secrecy.orig")
    d = d[d["post08reg"] == 0].reset_index(drop=True)
    act = np.where(d["treat"] == 1, "ballot_secrecy_letter", "no_letter")
    return frame("w2_wisconsin_recall", d[WI_X], pd.Series(act), d["vs_12_update"], WISCONSIN, propensity=0.5)


# ---------------------------------------------------------------- Freedom from Hunger, round 1
FFH = DatasetInfo(
    name="w2_ffh_charity", tier="C", outcome_name="gave_after_mailer", better="higher", reference="story_insert",
    actions=("story_insert", "research_insert"), license="CC0 1.0", commercial_use=True,
    citation="Karlan D, Wood DH. The effect of effectiveness: donor response to aid effectiveness in a "
    "direct mail fundraising experiment. J Behav Exp Econ 2017;66:1-8. doi:10.1016/j.socec.2016.05.005",
)
FFH_PUBLISHED = {"source": "round-1 raw means 0.159 vs 0.158 (spec)", "arm": "research_insert", "metric": "diff",
                 "value": 0.001, "tol": 0.01}
FFH_X = ["amount_pre", "amount_lastpre", "amount_maxpre", "HS_prior_gift1", "HS_prior_gift2", "HS_prior_gift3",
         "HS_prior_gift4", "HS_prior_gift5", "HS_prior_gift6", "HS_prior_year", "HS_r1_bigask",
         "H_number_yearbefore", "H_ngifts", "H_nyears", "H_frequency", "H_recent_giving", "H_frequent",
         "H_infrequent", "H_littleask", "H_bigask", "H_medinc", "H_medinc_mdum", "H_Avg_years_ed",
         "H_Avg_years_ed_mdum"]


def load_ffh() -> pd.DataFrame:
    d = _dta(RAW / "wave2_ffh_charity" / "ffh_analysis.orig")
    d = d[(d["HS_round"] == 1) & d["Treatment_f1"].isin([1, 2])].reset_index(drop=True)
    x = d[FFH_X].apply(pd.to_numeric, errors="coerce")
    mailer = pd.Timestamp("2007-06-15")
    last = pd.to_datetime(d["date_lastpre"], errors="coerce")
    x["months_since_last_gift"] = (mailer - last).dt.days / 30.44
    y = ((d["amount_after"] > 0) & d["amount_after"].notna()).astype(float)
    act = np.where(d["Treatment_f1"] == 1, "research_insert", "story_insert")
    return frame("w2_ffh_charity", x, pd.Series(act), y, FFH, propensity=0.5,
                 stratum=d["r1_strat"].astype(str), row_id=d["p_id"])


# ---------------------------------------------------------------- South Africa loan offers: one example loan
SALOAN = DatasetInfo(
    name="w2_sa_loan", tier="C", outcome_name="took_up_loan", better="higher", reference="four_example_loans",
    actions=("four_example_loans", "one_example_loan"), license="CC0 1.0", commercial_use=True,
    citation="Bertrand M, Karlan D, Mullainathan S, Shafir E, Zinman J. What's advertising content worth? "
    "Evidence from a consumer credit marketing field experiment. Q J Econ 2010;125:263-305. "
    "doi:10.1162/qjec.2010.125.1.263",
)
SALOAN_PUBLISHED = {"source": "WP Table 1: N = 53,194 (wave > 1)", "arm": "one_example_loan", "metric": "n_units",
                    "value": 53194, "tol": 0}
P_ONE = {"LOW": 0.15, "MEDIUM": 0.15, "HIGH": 0.52}


def load_sa_loan() -> pd.DataFrame:
    d = _dta(RAW / "wave2_sa_loan" / "adcontentworth.orig")
    d = d[(d["wave"] > 1) & d["tookup"].notna() & d["oneln_trt"].notna() & d["risk"].notna()].reset_index(drop=True)
    x = pd.DataFrame({"risk_low": (d["risk"] == "LOW").astype(float), "risk_med": (d["risk"] == "MEDIUM").astype(float)})
    for c in ("female", "race", "nspeakeligible", "wave", "waved3", "deadshort_elig", "deadlong_elig"):
        x[c] = pd.to_numeric(d[c], errors="coerce")  # the posted specification's covariates only
    act = np.where(d["oneln_trt"] == 1, "one_example_loan", "four_example_loans")
    p1 = d["risk"].map(P_ONE).to_numpy()
    return frame("w2_sa_loan", x, pd.Series(act), d["tookup"], SALOAN,
                 propensity=pd.Series(np.where(act == "one_example_loan", p1, 1 - p1)), stratum=d["risk"])


# ---------------------------------------------------------------- edX MOOC megastudy: long-term plans
MOOC = DatasetInfo(
    name="w2_mooc_megastudy", tier="C", outcome_name="certificate", better="higher", reference="control",
    actions=("control", "plan"), license="CC0 1.0 with additional terms (Dataverse)", commercial_use=False,
    citation="Kizilcec RF, Reich J, Yeomans M, et al. Scaling up behavioral science interventions in online "
    "education. Proc Natl Acad Sci U S A 2020;117:14900-14905. doi:10.1073/pnas.1921417117",
)
MOOC_PUBLISHED = {"source": "focal sample, plan vs control (spec)", "arm": "plan", "metric": "diff",
                  "value": 0.0019, "tol": 0.01}


def load_mooc() -> pd.DataFrame:
    d = pd.read_csv(RAW / "wave2_mooc_megastudy" / "wave12.orig")
    d = d[(d["focalSample"] == 1) & d["cond"].isin(["control", "plan"])].reset_index(drop=True)
    x = pd.DataFrame({"intent_assess": d["intent_assess"], "hours": d["hours"].clip(upper=40),
                      "crs_finish": d["crs_finish"].clip(upper=10), "educ": d["educ"], "is_fluent": d["is_fluent"],
                      "HDI4": d["HDI4"].astype("category").cat.codes.replace(-1, np.nan),
                      "days_from_start": d["days.from.start"]})
    act = d["cond"]
    return frame("w2_mooc_megastudy", x, act, d["cert_basic"], MOOC, propensity=0.5,
                 stratum=d["courseID"].astype(str) + "|" + d["strata"].astype(str))
