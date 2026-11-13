"""Wave-2 loaders for the three U.S. labor-market trials (registration PREREGISTRATION_WAVE2.md).

Each loader follows its posted specification in notebooks/system-three/preregistration/wave2_specs/
(wa_reemployment_bonus.md, jsa_florida.md, jtpa.md). Column bindings and every departure are
recorded in wave2_specs/LOADER_NOTES_labor.md. Raw files are read from data/raw/wave2_<key>/extracted/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo


def _dta(path, **kw):
    return pd.read_stata(path, convert_categoricals=False, **kw)


def _num(s: pd.Series, missing=()) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.mask(v.isin(list(missing)))


def _dummies(s: pd.Series, prefix: str, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(s.astype(str), prefix=prefix, drop_first=drop_first, dtype=float)


# ---------------------------------------------------------------- Washington Reemployment Bonus, 1988
WA_BONUS = DatasetInfo(
    name="w2_wa_bonus", tier="C", outcome_name="exhausted_ui_benefit_year", better="lower",
    reference="control", actions=("control", "bonus_6x_long"), license="Upjohn ERDC public use",
    commercial_use=False,
    citation="O'Leary CJ, Spiegelman RG, Kline KJ. Do bonus offers shorten unemployment insurance spells? "
    "Results from the Washington experiment. J Policy Anal Manage 1995;14:245-269.",
)
# WP 93-22 Table 8: T6 minus control, proportion exhausting benefits, unadjusted.
WA_BONUS_PUBLISHED = {"source": "Upjohn WP 93-22 Table 8 (T6 vs control, unadjusted)", "arm": "bonus_6x_long",
                      "metric": "diff", "value": -0.023, "tol": 0.01}
WA_WAGE_MISSING = 9.9e7  # 99999999.99 sentinel in readWREB.SAS, stored as 1e8 in float32


def _wa_quarter_index(yyq: pd.Series) -> pd.Series:
    """881 -> 1988Q1 as an integer quarter count."""
    yy, q = yyq // 10, yyq % 10
    return (1900 + yy) * 4 + (q - 1)


def load_wa_bonus() -> pd.DataFrame:
    d = _dta(RAW / "wave2_wa_reemployment_bonus" / "extracted" / "data" / "wreb.dta")
    d = d[(d["asc"] == 1) & d["ppi"].astype(str).isin(["0", "6"])].reset_index(drop=True)

    # Outcome: no exhaustion flag in the file; spec fallback = benefit-year compensation >= maximum benefits payable.
    y = (d["uitotal"] >= d["mbp"]).astype(float)

    x = pd.DataFrame({
        "wba": d["wba1"],                      # WBA at initial-claim processing (wba2 is at benefit-year end)
        "entitle_wks": d["entitle"],
        "base_wages": d["bwages"],
        "male": d["sex"],
        "age": d["age"],
        "educ_yrs": d["educate"],
        "black": (d["ethnic"] == "2").astype(float),
        "hispanic": (d["ethnic"] == "3").astype(float),
        "other_nonwhite": d["ethnic"].isin(["4", "5", "6"]).astype(float),
        "manufacturing": np.where(d["bsic"].isin(["", "X"]), np.nan, d["bsic"].isin(["2", "3"]).astype(float)),
        "white_collar": np.where(d["dot"].str[:1].isin(["", "X"]), np.nan,
                                 d["dot"].str[:1].isin(["0", "1", "2"]).astype(float)),
        "disloc_same_emp": d["disloc1"], "disloc_same_ind": d["disloc2"], "disloc_cont_emp": d["disloc3"],
        "disloc_lmea": d["dislmea"],
    })
    iaws = d["iaws"].astype(str).where(d["iaws"].astype(str).isin(["9", "S", "U"]), "other")
    x = pd.concat([x, _dummies(iaws, "iaws", drop_first=False).drop(columns="iaws_9"),
                   _dummies(d["jsc"], "jsc")], axis=1)

    # Pre-claim UI-covered wages: the four calendar quarters before the filing quarter.
    fq = _wa_quarter_index(d["fileqtr"].astype(int))
    for k in range(1, 5):
        tq = fq - k
        col = "wg" + ((tq // 4) % 100).astype(str) + ((tq % 4) + 1).astype(str)
        vals = np.array([d.at[i, c] if c in d.columns else np.nan for i, c in enumerate(col)], dtype=float)
        x[f"wage_q_minus{k}"] = np.where(vals >= WA_WAGE_MISSING, np.nan, vals)

    # Local labor market in the claim month (IUR/TUR month 1 = October 1987).
    mi = (d["edc"].dt.year - 1987) * 12 + d["edc"].dt.month - 9
    for pre, name in (("iur", "iur_claim_month"), ("tur", "tur_claim_month")):
        x[name] = [d.at[i, f"{pre}{m}"] if 1 <= m <= 26 else np.nan for i, m in enumerate(mi)]

    act = np.where(d["ppi"].astype(str) == "6", "bonus_6x_long", "control")
    p = 1.0 / 3.0  # designed allocation: 1,500 T6 vs 3,000 control
    return frame("w2_wa_bonus", x, pd.Series(act), y, WA_BONUS,
                 propensity=pd.Series(np.where(act == "bonus_6x_long", p, 1 - p)),
                 cluster=d["idno"].astype(int), row_id=d["idno"].astype(int))


# ---------------------------------------------------------------- Job Search Assistance Demonstration, Florida
JSA_FL = DatasetInfo(
    name="w2_jsa_florida", tier="C", outcome_name="exhausted_ui_year1", better="lower",
    reference="control", actions=("control", "sjsa"), license="Upjohn ERDC public use", commercial_use=False,
    citation="Decker PT, Olsen RB, Freeman L, Klepinger DH. Assisting Unemployment Insurance Claimants: The "
    "Long-Term Impacts of the Job Search Assistance Demonstration. Mathematica Policy Research; 2000.",
)
JSA_FL_PUBLISHED = {"source": "Decker et al. 2000 Table V.1 (FL SJSA 43.2% vs control 45.0%)", "arm": "sjsa",
                    "metric": "diff", "value": -0.018, "tol": 0.005}
# MISRACE and INDMISS are dropped: both are 0 for every control and SJSA row.
JSA_X = ["MALE", "AGE", "AGEMISS", "BLACK", "LATINO", "OTHER", "NOHS", "ASSOC", "BACHELOR", "GRAD",
         "EDUCMISS", "CONSTR", "MANUF", "PROFES", "CLERIC", "SERVICE", "OCCMISS", "TEN_YRS", "TENMISS",
         "WKBNAMT1", "ORG_ENT1", "BASEARN", "PSCORE"]


def load_jsa_florida() -> pd.DataFrame:
    # jsapubfl.dat is comma-separated with a header row (layout: separator=,).
    d = pd.read_csv(RAW / "wave2_jsa_florida" / "extracted" / "data" / "jsapubfl.dat", dtype=str,
                    keep_default_na=False)
    d = d[d["TG"].isin(["C", "1"]) & d["EXHAUST1"].isin(["0", "1"])].reset_index(drop=True)
    y = d["EXHAUST1"].astype(float)

    x = d[JSA_X].apply(pd.to_numeric, errors="coerce")  # "." and "" are missing
    x["local_unemp_rate"] = pd.to_numeric(d["LO_UN_RT"], errors="coerce") / 10.0
    byb = pd.to_datetime(d["BYB1"], format="%m/%d/%Y", errors="coerce")
    x["claim_day"] = (byb - pd.Timestamp("1995-01-01")).dt.days
    x = pd.concat([x, _dummies(d["ORI_LO"], "office")], axis=1)

    # Pre-claim quarterly UI-covered earnings: the four calendar quarters before the benefit-year-begin quarter.
    cq = byb.dt.year * 4 + byb.dt.quarter - 1
    for k in range(1, 5):
        tq = cq - k
        vals = []
        for i, q in enumerate(tq):
            c = None if pd.isna(q) else f"EARN{int(q) // 4 % 100:02d}Q{int(q) % 4 + 1}"
            vals.append(pd.to_numeric(d.at[i, c], errors="coerce") if c in d.columns else np.nan)
        x[f"earn_q_minus{k}"] = vals

    act = np.where(d["TG"] == "1", "sjsa", "control")
    return frame("w2_jsa_florida", x, pd.Series(act), y, JSA_FL, propensity=0.5,
                 cluster=d["MPRID"], stratum=d["ORI_LO"], row_id=d["MPRID"])


# ---------------------------------------------------------------- National JTPA Study, adults, 30-month sample
JTPA = DatasetInfo(
    name="w2_jtpa", tier="C", outcome_name="any_earnings_30mo", better="higher", reference="control",
    actions=("control", "jtpa_access"), license="Upjohn ERDC public use", commercial_use=False,
    citation="Bloom HS, Orr LL, Bell SH, Cave G, Doolittle F, Lin W, Bos JM. The benefits and costs of JTPA "
    "Title II-A programs: key findings from the National Job Training Partnership Act study. J Hum Resour "
    "1997;32:549-576.",
)
# No binary headline is published. Orr et al. 1994 Exhibits 2.4 and 4.4: adult 30-month earnings sample
# = 6,102 adult women + 5,102 adult men.
JTPA_PUBLISHED = {"source": "Orr et al. 1994 Exhibit 2.4 (adult 30-month earnings sample)", "arm": "jtpa_access",
                  "metric": "n_units", "value": 11204, "tol": 0}


def load_jtpa() -> pd.DataFrame:
    base = RAW / "wave2_jtpa" / "extracted" / "Data"
    h = _dta(base / "expfup1" / "stata" / "head_fp1.dta")
    b = _dta(base / "expbif" / "stata" / "expbif.dta")
    e = _dta(base / "replacement_files" / "stata" / "earns2.dta")  # replaces analysis/earns (RECID fix)

    h = h[h["target"].isin(["1", "2"]) & (h["excesst"] == "2")]
    d = h.merge(e, on="recid", how="inner").merge(b, on="recid", how="left", suffixes=("", "_bif"))
    ern_cols = [f"newern{m:02d}" for m in range(1, 31)]
    d = d[d[ern_cols].notna().all(axis=1)].reset_index(drop=True)  # spec: non-missing 30-month earnings
    y = (d[ern_cols].sum(axis=1) > 0).astype(float)

    m9, m99 = (9,), (99,)
    ra = pd.to_datetime(d["ra_dt"], format="%y%m%d", errors="coerce")
    x = pd.DataFrame({
        "female": (d["target"] == "1").astype(float),
        "age": _num(d["age"], m99),
        "black": (d["race"] == "2").astype(float),
        "hispanic": (d["race"] == "3").astype(float),
        "other_race": d["race"].isin(["4", "5"]).astype(float),
        "highest_grade": _num(d["gradcmpt"], m99),
        "hs_or_ged": _num(d["hsgrdged"], m9),
        "married": _num(d["mstatmar"], m9),
        "never_married": _num(d["mstatsi"], m9),
        "wid_div_sep": _num(d["mstatwds"], m9),
        "hh_size": _num(d["bfhhsize"], m99),
        "n_children_lt19": _num(d["numlt19"], m99),
        "child_under4": _num(d["chunder4"], m9),
        "hh_afdc": _num(d["hhafdc"], m9),
        "hh_food_stamps": _num(d["hhfdst"], m9),
        "hh_gen_assist": _num(d["genass"], m9),
        "afdc_now": _num(d["afdcnow"], m9),
        "employed_at_app": _num(d["curempl"], m9),
        "ever_worked": _num(d["wkforpy"], m9),
        "weeks_worked_12m": _num(d["weekswrk"], m99),
        "earnings_12m": _num(d["yearearn"], (99999,)),
        "english_primary": _num(d["english"], m9),
        "ra_month": (ra.dt.year - 1987) * 12 + ra.dt.month - 11,
    })
    x = pd.concat([x, _dummies(d["sda"], "site"), _dummies(d["treatmt"], "strategy")], axis=1)

    act = np.where(d["tcstat"] == "1", "jtpa_access", "control")
    p = 2.0 / 3.0
    return frame("w2_jtpa", x, pd.Series(act), y, JTPA,
                 propensity=pd.Series(np.where(act == "jtpa_access", p, 1 - p)),
                 cluster=d["recid"], stratum=d["sda"] + "|" + d["treatmt"], row_id=d["recid"])
