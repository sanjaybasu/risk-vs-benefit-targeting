"""Pre-registered Wave 3 confirmatory pipeline (Zenodo pre-registration DOI: 10.5281/zenodo.24487742).

Six never-before-seen public health-care uptake and clinical randomized trials:
  1. w3_campos_vaccine   (Campos-Mercade et al., Science 2021; COVID-19 vaccination uptake, Sweden)
  2. w3_mtics_crc_fit    (Vives et al., JMIR Public Health Surveill 2024; CRC FIT kit return, Spain)
  3. w3_mtics_crc_remind (Vives et al., JMIR mHealth uHealth 2025; CRC screening reminder, Spain)
  4. w3_alsan_covid      (Alsan et al., Ann Intern Med 2021; physician COVID video messaging, US)
  5. w3_ist3             (IST-3 Collaborative Group, Lancet 2012; IV alteplase acute stroke)
  6. w3_kat_aki          (Wilson et al., JAMA 2024 / Dryad 2026; Kidney Action Team AKI inpatient RCT)

Stages executed cleanly in strict sealed-split order:
  Stage 1: Load raw files, verify published headline reproduction check, split 50/20/30 by salted
           SHA-256 hash (SALT_W3 = "system-three-wave3-2026-11-23"), write processed context/dev/test_state,
           and seal test outcomes outside the repository in ../system-three-sealed/.
  Stage 2: Fit all comparators strictly on context (plus dev for calibrated ensemble, tuned risk-augmented
           DR, and 70% scale comparator), write predictions_w3/<trial>.parquet, and record SHA-256 hashes
           in results/w3_manifest.json BEFORE reading any sealed test outcome.
  Stage 3: Verify SHA-256 hashes and score ONCE on sealed test splits using the exact AIPW estimator and
           2,000-replicate cluster bootstrap (SEED = 20261112), evaluating W3-H1 through W3-H4.
  Output:  results/wave3_results.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEALED = ROOT.parent / "system-three-sealed"
PRED_W3 = ROOT / "predictions_w3"
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

spec_mr = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec_mr)
sys.modules["mr"] = mr
spec_mr.loader.exec_module(mr)

spec_diag = importlib.util.spec_from_file_location("diag", ROOT / "scripts" / "28_proportionality_diagnostic.py")
diag = importlib.util.module_from_spec(spec_diag)
spec_diag.loader.exec_module(diag)

spec_s50 = importlib.util.spec_from_file_location("s50", ROOT / "scripts" / "50_ben_additional_exploratory.py")
s50 = importlib.util.module_from_spec(spec_s50)
spec_s50.loader.exec_module(s50)

from system_three import rankers  # noqa: E402
from system_three.estimators.dr import aipw_scores, sign  # noqa: E402

SALT_W3 = "system-three-wave3-2026-11-23"
SEED = 20261112
N_BOOT = 2000


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hash_w3(trial: str, cid: str) -> float:
    h = hashlib.sha256(f"{SALT_W3}|{trial}|{cid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def one_hot_encode(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in num_cols:
        out[f"x_{c}"] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat_cols:
        s = df[c].astype(str).fillna("MISSING")
        top = s.value_counts().index[:12]
        for val in top:
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(val))[:24]
            out[f"x_{c}_{safe}"] = (s == val).astype(float)
    return out


# ------------------------------------------------------------------
# Loaders for the 6 pre-registered Wave 3 trials
# ------------------------------------------------------------------
def load_w3_campos_vaccine() -> tuple[pd.DataFrame, dict]:
    """Campos-Mercade et al., Science 2021: guaranteed monetary incentive vs control, COVID-19 vaccination."""
    df = pd.read_stata(RAW / "w3_campos_vaccine" / "behavioral_vaccine_trial" / "data_analysis" / "data_analysis.dta",
                       convert_categoricals=False)
    df = df[(df["already_vacced"] == 0) & (df["intention1"].notna())].copy()
    df = df[(df["treat_pay"] == 1) | (df["control"] == 1)].reset_index(drop=True)
    w = (df["treat_pay"] == 1).astype(int).to_numpy()
    y = (df["vaccinated"] / 100.0).astype(float).to_numpy()
    # Pre-treatment covariates matching pre-specified pap_contr + baseline preference/belief items
    num_cols = ["age", "female", "haschildren", "single", "university", "foreign", "unemployed",
                "covid1_1", "covid1_2", "covid2_1", "covid2_2", "covid2_3", "covid2_4",
                "falk1_1", "falk1_2", "falk1_3", "falk2_1", "falk2_2", "falk2_3", "falk2_4",
                "belief_prot_death", "belief_cause_death", "income", "education", "week_num"]
    cat_cols = ["age_band", "region", "civilstatus", "occupation"]
    X = one_hot_encode(df, cat_cols, [c for c in num_cols if c in df.columns])
    p_treat = float(w.mean())
    prop = np.where(w == 1, p_treat, 1.0 - p_treat)
    pub_diff = float(y[w == 1].mean() - y[w == 0].mean())
    meta = {
        "trial": "w3_campos_vaccine", "label": "Sweden COVID vaccine incentives",
        "domain": "health_uptake", "better": "higher",
        "action": "monetary_incentive", "reference": "control",
        "p_action": p_treat, "p_reference": 1.0 - p_treat,
        "published_target": 0.042, "published_desc": "Science 2021 Fig 1: 75.8% vs 71.6% (+4.2 pp)",
        "reproduced": pub_diff, "tol": 0.005, "n_total": len(df),
        "license": "CC BY 4.0 (Zenodo doi:10.5281/zenodo.5529625)",
    }
    return _pack("w3_campos_vaccine", X, w, y, prop, meta), meta


def load_w3_mtics_crc_fit() -> tuple[pd.DataFrame, dict]:
    """Vives et al., JMIR Public Health Surveill 2024: SMS + letter vs letter for CRC FIT return."""
    df = pd.read_csv(RAW / "w3_mtics_crc_fit" / "FIT reminder sms data.tab", sep="\t")
    df = df[df["trialarm"].isin(["sms arm", "control arm"])].reset_index(drop=True)
    w = (df["trialarm"] == "sms arm").astype(int).to_numpy()
    y = df["participacionbin126d"].astype(str).str.strip().str.lower().isin(["1", "1.0", "yes", "si", "true"]).astype(float).to_numpy()
    assert 0.01 < y.mean() < 0.99, f"Unexpected y.mean()={y.mean()}"
    num_cols = ["age", "deprivationlevel", "numberFITpickup"]
    cat_cols = ["sex", "agecat2", "tertiledeprivation", "previousscreening"]
    X = one_hot_encode(df, cat_cols, num_cols)
    p_treat = float(w.mean())
    prop = np.where(w == 1, p_treat, 1.0 - p_treat)
    meta = {
        "trial": "w3_mtics_crc_fit", "label": "Catalonia CRC FIT return SMS",
        "domain": "health_uptake", "better": "higher",
        "action": "sms_plus_letter", "reference": "letter_only",
        "p_action": p_treat, "p_reference": 1.0 - p_treat,
        "published_target": 9369, "published_desc": "JMIR PHS 2024: N = 9,369 randomized (4,563 SMS vs 4,806 control)",
        "reproduced": float(len(df)), "tol": 0.0, "n_total": len(df),
        "rate_t": float(y[w == 1].mean()), "rate_c": float(y[w == 0].mean()),
        "license": "CC BY 4.0 (CORA Dataverse doi:10.34810/data1481)",
    }
    return _pack("w3_mtics_crc_fit", X, w, y, prop, meta), meta


def load_w3_mtics_crc_remind() -> tuple[pd.DataFrame, dict]:
    """Vives et al., JMIR mHealth uHealth 2025: SMS vs postal reminder for CRC FIT participation."""
    df = pd.read_csv(RAW / "w3_mtics_crc_remind" / "MTICS_CRC reminder intervention.tab", sep="\t")
    df = df[df["trialarm"].isin(["sms arm", "control arm"])].reset_index(drop=True)
    w = (df["trialarm"] == "sms arm").astype(int).to_numpy()
    col_y = "participacioninter"
    y = df[col_y].astype(str).str.strip().str.lower().isin(["1", "1.0", "yes", "si", "true"]).astype(float).to_numpy()
    assert 0.01 < y.mean() < 0.99, f"Unexpected y.mean()={y.mean()}"
    num_cols = ["age", "deprivationlevel"]
    cat_cols = ["sex", "agecat2", "tertiledeprivation", "previousscreening"]
    X = one_hot_encode(df, cat_cols, num_cols)
    p_treat = float(w.mean())
    prop = np.where(w == 1, p_treat, 1.0 - p_treat)
    meta = {
        "trial": "w3_mtics_crc_remind", "label": "Catalonia CRC invitation SMS vs mail",
        "domain": "health_uptake", "better": "higher",
        "action": "sms_reminder", "reference": "postal_letter",
        "p_action": p_treat, "p_reference": 1.0 - p_treat,
        "published_target": 24388, "published_desc": "JMIR mHealth 2025: N = 24,388 (12,167 SMS vs 12,221 letter)",
        "reproduced": float(len(df)), "tol": 0.0, "n_total": len(df),
        "rate_t": float(y[w == 1].mean()), "rate_c": float(y[w == 0].mean()),
        "license": "CC BY 4.0 (CORA Dataverse doi:10.34810/data1713)",
    }
    return _pack("w3_mtics_crc_remind", X, w, y, prop, meta), meta


def load_w3_alsan_covid() -> tuple[pd.DataFrame, dict]:
    """Alsan et al., Ann Intern Med 2021: physician COVID video messaging vs control."""
    df = pd.read_csv(RAW / "w3_alsan_covid" / "COVID USA T1_May 27, 2020_07.20.csv",
                     skiprows=[1, 2], encoding="latin1", low_memory=False)
    # Match exact main_regressions.R & basic_cleaning.R sample filter
    st = pd.to_datetime(df["StartDate"], errors="coerce")
    # StartDate in CSV is Mountain/UTC offset; check exact filter from basic_cleaning.R
    q2_ok = df["Q2"].astype(str).str.contains("I LIVE IN THE U.S., and I am 18 or older", na=False)
    df = df[(df["DistributionChannel"] == "anonymous") & q2_ok & df["test"].isna()].copy()
    df = df[st.loc[df.index] > pd.Timestamp("2020-05-13 13:00:00")].copy()
    df["pure_control_num"] = pd.to_numeric(df["pure_control"], errors="coerce")
    df = df[(df["doctor_race"].notna() & df["desc_vid1"].notna()) | (df["pure_control_num"] == 1)].copy()
    # Primary binary outcome: zero knowledge gap (1 - any_knowledge_gap; higher is better)
    k3p = (df["Q43"] == "Stay six feet away from other people when outside,Wash your hands when you come home ,Wear a mask or facial covering when outside").astype(float)
    kas = (df["Q44"] == "Yes").astype(float)
    ksym = df["Q46"].isin([
        "Fever,Cough,Difficulty breathing",
        "Fever,Cough,New loss of taste or smell",
        "Fever,Difficulty breathing,New loss of taste or smell",
        "Cough,Difficulty breathing,New loss of taste or smell",
    ]).astype(float)
    # Drop rows with missing knowledge responses
    ok_y = df["Q43"].notna() & df["Q44"].notna() & df["Q46"].notna()
    df = df[ok_y].reset_index(drop=True)
    k3p, kas, ksym = k3p[ok_y].to_numpy(), kas[ok_y].to_numpy(), ksym[ok_y].to_numpy()
    zero_gap = ((k3p == 1) & (kas == 1) & (ksym == 1)).astype(float)
    w = (1 - pd.to_numeric(df["pure_control"], errors="coerce").fillna(0).astype(int)).to_numpy()
    # Baseline pre-treatment covariates only (strictly Q3..Q19 and pre-loaded demographics)
    df["hh_size"] = df["Q14"].replace({"8 or more": "8"})
    df["medical_cond"] = (df["Q19"].notna() & (df["Q19"] != "None")).astype(float)
    df["health_ins"] = (df["Q17"] == "Yes").astype(float)
    df["essential_wk"] = (df["Q10"] == "Yes").astype(float)
    df["food_insec"] = (df["Q13"] == "Yes").astype(float)
    df["work_home"] = (df["Q11"] == "Yes").astype(float)
    df["spanish"] = df["Q3"].astype(str).str.contains("Spanish", case=False, na=False).astype(float)
    num_cols = ["age", "standard_education", "standard_hhi", "hh_size", "Q15",
                "medical_cond", "health_ins", "essential_wk", "food_insec", "work_home", "spanish"]
    cat_cols = ["gender", "race_resp", "region", "Q8", "Q9", "Q12", "Q18"]
    X = one_hot_encode(df, cat_cols, num_cols)
    # Stratified randomization propensity (within stratum)
    if "stratum" in df.columns and df["stratum"].notna().any():
        p_s = pd.Series(w).groupby(df["stratum"].fillna("NA").to_numpy()).transform("mean").to_numpy()
        prop = np.where(w == 1, p_s, 1.0 - p_s)
        p_act, p_ref = None, None
    else:
        p_treat = float(w.mean())
        prop = np.where(w == 1, p_treat, 1.0 - p_treat)
        p_act, p_ref = p_treat, 1.0 - p_treat
    meta = {
        "trial": "w3_alsan_covid", "label": "US physician COVID video messages",
        "domain": "health_uptake", "better": "higher",
        "action": "physician_video", "reference": "control",
        "p_action": p_act, "p_reference": p_ref,
        "published_target": 0.076, "published_desc": "Ann Intern Med 2021: zero knowledge gap 80.3% vs 72.7% (+7.6 pp)",
        "reproduced": float(zero_gap[w == 1].mean() - zero_gap[w == 0].mean()), "tol": 0.015,
        "n_total": len(df), "rate_t": float(zero_gap[w == 1].mean()), "rate_c": float(zero_gap[w == 0].mean()),
        "license": "CC0 1.0 (Harvard Dataverse doi:10.7910/DVN/CJPVOD)",
    }
    return _pack("w3_alsan_covid", X, w, zero_gap, prop, meta), meta


def load_w3_ist3() -> tuple[pd.DataFrame, dict]:
    """IST-3 Collaborative Group, Lancet 2012: IV alteplase vs open control, 6-mo dead or dependent."""
    df = pd.read_sas(RAW / "w3_ist3" / "datashare_aug2015.sas7bdat", encoding="latin1")
    w = (df["treatment"].astype(str).str.strip() == "rt-PA").astype(int).to_numpy()
    # deadordep6 == 1 means dead or dependent (OHS 3-6, 63.4% rt-PA vs 64.9% control), lower is better
    y = (df["deadordep6"] == 1).astype(float).to_numpy()
    num_cols = ["age", "randdelay", "sbprand", "dbprand", "weight", "glucose",
                "gcs_eye_rand", "gcs_motor_rand", "gcs_verbal_rand", "gcs_score_rand", "nihss", "konprob"]
    cat_cols = ["gender", "livealone_rand", "indepinadl_rand", "nobleed_rand", "infarct",
                "antiplat_rand", "atrialfib_rand", "liftarms_rand", "ablewalk_rand",
                "weakface_rand", "weakarm_rand", "weakleg_rand", "dysphasia_rand",
                "hemianopia_rand", "visuospat_rand", "brainstemsigns_rand", "stroketype", "country"]
    X = one_hot_encode(df, [c for c in cat_cols if c in df.columns], [c for c in num_cols if c in df.columns])
    p_treat = float(w.mean())
    prop = np.where(w == 1, p_treat, 1.0 - p_treat)
    # Lancet 2012 primary: alive & independent 37% vs 35% -> dead/dep 63.0% vs 65.0% (-2.0 pp)
    diff = float(y[w == 1].mean() - y[w == 0].mean())
    meta = {
        "trial": "w3_ist3", "label": "IST-3 acute stroke alteplase",
        "domain": "medicine", "better": "lower",
        "action": "alteplase", "reference": "control",
        "p_action": p_treat, "p_reference": 1.0 - p_treat,
        "published_target": -0.017, "published_desc": "Lancet 2012: 6-mo dead or dependent 63.3% vs 65.0% (-1.7 pp, N=3,035)",
        "reproduced": diff, "tol": 0.01, "n_total": len(df),
        "rate_t": float(y[w == 1].mean()), "rate_c": float(y[w == 0].mean()),
        "license": "ODC-By (Edinburgh DataShare doi:10.7488/ds/1350)",
    }
    return _pack("w3_ist3", X, w, y, prop, meta), meta


def load_w3_kat_aki() -> tuple[pd.DataFrame, dict]:
    """Wilson et al., JAMA 2024 / Dryad 2026: Kidney Action Team EHR recommendations for AKI."""
    df = pd.read_csv(RAW / "w3_kat_aki" / "KAT-AKI_limited_dataset.04-23-26.csv")
    w = (df["randomization"] == 1).astype(int).to_numpy()
    y = df["primary_outcome"].astype(float).to_numpy()
    # Strictly pre-randomization covariates (recommendations were generated prior to randomization!)
    num_cols = ["age_int", "femalesex", "ckd_pmhx", "chf_pmhx", "dm_pmhx", "htn_pmhx",
                "cirrhosis_pmhx", "baseline_creat", "start2enrollment_days", "age_over_90",
                "general", "general_diag_rec_count", "volume", "volume_rec_count",
                "potas", "potas_rec_count", "acidosis", "acidosis_rec_count",
                "meds", "meds_rec_count", "renal_consult_rec", "total_recs"]
    cat_cols = ["hospital"]
    X = one_hot_encode(df, cat_cols, num_cols)
    p_s = pd.Series(w).groupby(df["hospital"].to_numpy()).transform("mean").to_numpy()
    prop = np.where(w == 1, p_s, 1.0 - p_s)
    diff = float(y[w == 1].mean() - y[w == 0].mean())
    meta = {
        "trial": "w3_kat_aki", "label": "KAT-AKI inpatient EHR team",
        "domain": "medicine", "better": "lower",
        "action": "kat_note", "reference": "usual_care",
        "p_action": None, "p_reference": None,
        "published_target": 0.014, "published_desc": "JAMA 2024: 14-d AKI composite 19.8% vs 18.4% (+1.4 pp, N=4,003)",
        "reproduced": diff, "tol": 0.005, "n_total": len(df),
        "rate_t": float(y[w == 1].mean()), "rate_c": float(y[w == 0].mean()),
        "license": "CC0 1.0 (Dryad doi:10.5061/dryad.sqv9s4ncs)",
    }
    return _pack("w3_kat_aki", X, w, y, prop, meta), meta


def _pack(trial: str, X: pd.DataFrame, w: np.ndarray, y: np.ndarray, prop: np.ndarray, meta: dict) -> pd.DataFrame:
    n = len(y)
    out = pd.DataFrame({
        "dataset": trial,
        "row_id": np.arange(n, dtype=np.int64),
        "cluster_id": [str(i) for i in range(n)],
        "stratum": "all",
        "action": np.where(w == 1, meta["action"], meta["reference"]),
        "reference": meta["reference"],
        "W": w.astype(int),
        "outcome": y.astype(float),
        "better": meta["better"],
        "propensity": prop.astype(float),
        "e": np.where(w == 1, prop, 1.0 - prop).astype(float),
    })
    for col in X.columns:
        out[col] = X[col].to_numpy(float)
    return out


def cluster_mean_se(x: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    n = len(x)
    mu = float(x.mean())
    df = pd.DataFrame({"d": x - mu, "c": cl})
    sums = df.groupby("c", sort=False)["d"].sum().to_numpy(float)
    G = len(sums)
    var = (G / (G - 1.0)) * float(np.sum(sums ** 2)) / (n ** 2) if G > 1 else float("nan")
    return mu, float(np.sqrt(max(var, 0.0))), G


def cluster_diff_se(g: np.ndarray, m1: np.ndarray, m2: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    n1, n2 = int(m1.sum()), int(m2.sum())
    if n1 < 50 or n2 < 50:
        return float("nan"), float("nan"), 0
    mu1, mu2 = float(g[m1].mean()), float(g[m2].mean())
    n = len(g)
    psi = np.where(m1, (g - mu1) / (n1 / n), 0.0) - np.where(m2, (g - mu2) / (n2 / n), 0.0)
    df = pd.DataFrame({"psi": psi, "c": cl})
    sums = df.groupby("c", sort=False)["psi"].sum().to_numpy(float)
    G = len(sums)
    var = (G / (G - 1.0)) * float(np.sum(sums ** 2)) / (n ** 2)
    return mu1 - mu2, float(np.sqrt(max(var, 0.0))), G


def fit_rose_ensemble(Xc, wc, yc, ec, gc, Xd, wd, yd, ed, Xt, better):
    """30-learner cluster-bootstrap DR ensemble with isotonic dev calibration (matches Rose v2)."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(gc, return_inverse=True)
    cl_rows = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    m0c, m1c, f0c, f1c = rankers.cross_fit_p(Xc, wc, yc, gc)
    pseudo_c = m1c - m0c + wc * (yc - m1c) / ec - (1 - wc) * (yc - m0c) / (1 - ec)
    m0d, m1d = rankers._avg(f0c, Xd), rankers._avg(f1c, Xd)
    pseudo_d = m1d - m0d + wd * (yd - m1d) / ed - (1 - wd) * (yd - m0d) / (1 - ed)
    d_preds = np.empty((30, len(Xd)))
    t_preds = np.empty((30, len(Xt)))
    for b in range(30):
        idx = np.concatenate([cl_rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
        mdl = HistGradientBoostingRegressor(**{**rankers.GB, "random_state": SEED + b}).fit(Xc[idx], pseudo_c[idx])
        d_preds[b] = mdl.predict(Xd)
        t_preds[b] = mdl.predict(Xt)
    s_sign = sign(better)
    raw_d = s_sign * d_preds.mean(axis=0)
    raw_t = s_sign * t_preds.mean(axis=0)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_d, s_sign * pseudo_d)
    cal_t = iso.predict(raw_t)
    # Break isotonic ties using raw_t (tiny jitter preserving ranking order)
    return cal_t + 1e-7 * (raw_t - raw_t.mean())


def main() -> None:
    t0_all = time.time()
    PRED_W3.mkdir(exist_ok=True)
    loaders = [
        load_w3_campos_vaccine,
        load_w3_mtics_crc_fit,
        load_w3_mtics_crc_remind,
        load_w3_alsan_covid,
        load_w3_ist3,
        load_w3_kat_aki,
    ]

    # ------------------------------------------------------------------
    # STAGE 1: Load, verify reproduction, split 50/20/30, SEAL test split
    # ------------------------------------------------------------------
    metas = {}
    print("=== STAGE 1: Load, Reproduce, Split, Seal ===")
    for fn in loaders:
        df, meta = fn()
        t = meta["trial"]
        ok_rep = abs(meta["reproduced"] - meta["published_target"]) <= meta["tol"]
        meta["reproduction_ok"] = bool(ok_rep)
        u = df["cluster_id"].map(lambda c: hash_w3(t, c)).to_numpy()
        sp = np.where(u < 0.50, "context", np.where(u < 0.70, "dev", "test"))
        pdir = PROC / t
        pdir.mkdir(parents=True, exist_ok=True)
        ctx = df[sp == "context"].reset_index(drop=True)
        dev = df[sp == "dev"].reset_index(drop=True)
        tst = df[sp == "test"].reset_index(drop=True)
        ctx.to_parquet(pdir / "context.parquet", index=False)
        dev.to_parquet(pdir / "dev.parquet", index=False)
        x_cols = [c for c in df.columns if c.startswith("x_")]
        tst[["dataset", "row_id", "cluster_id", "stratum"] + x_cols].to_parquet(pdir / "test_state.parquet", index=False)
        # Seal test outcomes outside repo
        tst[["row_id", "cluster_id", "stratum", "action", "outcome", "propensity"]].to_parquet(SEALED / f"{t}_test.parquet", index=False)
        (SEALED / f"{t}_meta.json").write_text(json.dumps({
            "dataset": t, "better": meta["better"], "reference": meta["reference"], "action": meta["action"],
            "p_action": meta["p_action"], "p_reference": meta["p_reference"],
        }, indent=1))
        meta["n_context"] = int(len(ctx))
        meta["n_dev"] = int(len(dev))
        meta["n_test"] = int(len(tst))
        meta["n_covariates"] = len(x_cols)
        metas[t] = meta
        print(f"  {t:22s}: N={len(df):6d} (ctx={len(ctx):5d}, dev={len(dev):4d}, test={len(tst):4d}, p={len(x_cols):2d}) "
              f"| rep={meta['reproduced']:.4f} vs pub={meta['published_target']} ({'OK' if ok_rep else 'CHECK'})", flush=True)

    # ------------------------------------------------------------------
    # STAGE 2: Fit strictly on context (+dev), write predictions, HASH BEFORE SCORING
    # ------------------------------------------------------------------
    print("\n=== STAGE 2: Fit Models & Post SHA-256 Hashes BEFORE Scoring ===")
    manifest = {}
    ctx_info = {}
    for t, meta in metas.items():
        t0 = time.time()
        ctx = pd.read_parquet(PROC / t / "context.parquet")
        dev = pd.read_parquet(PROC / t / "dev.parquet")
        tst_s = pd.read_parquet(PROC / t / "test_state.parquet")

        Xc, cols = rankers.design_matrix(ctx)
        Xd, _ = rankers.design_matrix(dev, cols)
        Xt, _ = rankers.design_matrix(tst_s, cols)
        wc, yc, ec, gc = ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy(), ctx["cluster_id"].to_numpy()
        wd, yd, ed, gd = dev["W"].to_numpy(), dev["outcome"].to_numpy(), dev["e"].to_numpy(), dev["cluster_id"].to_numpy()
        better = meta["better"]

        # Registered context-only models (risk, scale, DR, response)
        f = rankers.fit(Xc, wc, yc, ec, better, groups=gc)
        s = rankers.scores(f, Xt)
        resp_score = s["_m1"] if better == "higher" else (1.0 - s["_m1"])

        # Context-only diagnostic + 4-param PATH interaction
        diag_res = diag.diagnose(Xc, wc, yc, ec, gc, better)
        m0_oof, _, _, _ = rankers.cross_fit_p(Xc, wc, yc, gc)
        off_c = logit(m0_oof)
        mean_off = float(off_c.mean())
        wt_c = np.where(wc == 1, 1.0 / ec, 1.0 / (1.0 - ec))
        th1, _ = diag.glm(np.column_stack([np.ones_like(off_c), off_c, wc, wc * (off_c - mean_off)]), yc, wt_c, gc)
        off_t = logit(np.clip(s["_m0"], 1e-4, 1.0 - 1e-4))
        lin_t = th1[0] + th1[1] * off_t
        path_score = sign(better) * (expit(lin_t + th1[2] + th1[3] * (off_t - mean_off)) - expit(lin_t))

        # Calibrated ensemble (Rose v2)
        rose_score = fit_rose_ensemble(Xc, wc, yc, ec, gc, Xd, wd, yd, ed, Xt, better)

        # Upgraded tuned risk-augmented 70%-train DR AND 70%-train scale
        m0c, m1c, f0c, f1c = rankers.cross_fit_p(Xc, wc, yc, gc)
        pseudo_c = m1c - m0c + wc * (yc - m1c) / ec - (1 - wc) * (yc - m0c) / (1 - ec)
        m0d, m1d = rankers._avg(f0c, Xd), rankers._avg(f1c, Xd)
        pseudo_d = m1d - m0d + wd * (yd - m1d) / ed - (1 - wd) * (yd - m0d) / (1 - ed)
        Xc_aug = np.column_stack([Xc, logit(m0c)])
        Xd_aug = np.column_stack([Xd, logit(m0d)])
        grid = [
            {"max_leaf_nodes": 7, "min_samples_leaf": 200, "l2_regularization": 10.0, "max_iter": 150, "learning_rate": 0.03},
            {"max_leaf_nodes": 7, "min_samples_leaf": 100, "l2_regularization": 5.0, "max_iter": 200, "learning_rate": 0.05},
            {"max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 5.0, "max_iter": 200, "learning_rate": 0.05},
            {"max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0, "max_iter": 300, "learning_rate": 0.05},
            {"max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0, "max_iter": 300, "learning_rate": 0.05},
            {"max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1, "max_iter": 300, "learning_rate": 0.05},
        ]
        best_hp = min(grid, key=lambda hp: float(np.mean((pseudo_d - HistGradientBoostingRegressor(**hp, early_stopping=False, random_state=SEED).fit(Xc_aug, pseudo_c).predict(Xd_aug)) ** 2)))
        X_cd = np.vstack([Xc, Xd])
        w_cd = np.concatenate([wc, wd])
        y_cd = np.concatenate([yc, yd])
        e_cd = np.concatenate([ec, ed])
        g_cd = np.concatenate([gc, gd])
        m0_cd, m1_cd, f0_cd, f1_cd = rankers.cross_fit_p(X_cd, w_cd, y_cd, g_cd)
        wt_cd = np.where(w_cd == 1, 1.0 / e_cd, 1.0 / (1.0 - e_cd))
        a_cd, g_cd_s, b_cd, se_cd = rankers.scale_fit(logit(m0_cd), w_cd, y_cd, wt_cd)
        m0_t_cd = rankers._avg(f0_cd, Xt)
        scale_70 = rankers.scale_benefit(rankers.Fitted(f0_cd, f1_cd, None, better, a_cd, g_cd_s, b_cd, se_cd, 0.0), m0_t_cd)
        pseudo_cd = m1_cd - m0_cd + w_cd * (y_cd - m1_cd) / e_cd - (1 - w_cd) * (y_cd - m0_cd) / (1 - e_cd)
        mdl_str = HistGradientBoostingRegressor(**best_hp, early_stopping=False, random_state=SEED).fit(np.column_stack([X_cd, logit(m0_cd)]), pseudo_cd)
        dr_strong = sign(better) * mdl_str.predict(np.column_stack([Xt, logit(m0_t_cd)]))

        # Context-split constant-OR check & scale-predicted gap (posted BEFORE scoring!)
        beneficial = bool((better == "higher" and f.beta > 0) or (better == "lower" and f.beta < 0))
        lin_c = f.alpha + f.gamma * logit(m0_oof)
        r_bad_c = expit(lin_c) if better == "lower" else expit(-lin_c)
        u_bad_c = logit(np.clip(r_bad_c, 1e-6, 1 - 1e-6))
        c_abs = abs(float(f.beta))
        u_max = float(np.max(u_bad_c))
        u_q = float(np.quantile(u_bad_c, 0.80))
        check_val = u_max + u_q - c_abs
        tau_c = expit(u_bad_c) - expit(u_bad_c - c_abs) if beneficial else np.zeros_like(u_bad_c)
        k20_c = max(1, int(round(0.2 * len(u_bad_c))))
        jit_c = np.random.default_rng(SEED).random(len(u_bad_c))
        gap_posted = float(1000.0 * (tau_c[np.lexsort((jit_c, -tau_c))[:k20_c]].mean() - tau_c[np.lexsort((jit_c, -r_bad_c))[:k20_c]].mean())) if beneficial else None

        # Context rates for constant RR_bad and RD
        bad_c = yc if better == "lower" else (1.0 - yc)
        w0_c, w1_c = (1 - wc) / (1 - ec), wc / ec
        r0_ctx = float(np.sum(w0_c * bad_c) / np.sum(w0_c))
        r1_ctx = float(np.sum(w1_c * bad_c) / np.sum(w1_c))

        pred_df = pd.DataFrame({
            "row_id": tst_s["row_id"].to_numpy(),
            "risk": np.asarray(s["risk"], float),
            "scale": np.asarray(s["scale"], float),
            "benefit": np.asarray(s["benefit"], float),
            "response": np.asarray(resp_score, float),
            "rose": np.asarray(rose_score, float),
            "path_int": np.asarray(path_score, float),
            "dr_strong": np.asarray(dr_strong, float),
            "scale_70": np.asarray(scale_70, float),
            "m0": np.asarray(s["_m0"], float),
            "m1": np.asarray(s["_m1"], float),
        })
        pp = PRED_W3 / f"{t}.parquet"
        pred_df.to_parquet(pp, index=False)
        h_sha = sha256_file(pp)
        manifest[t] = {"sha256": h_sha, "n_test": int(len(pred_df))}
        ctx_info[t] = {
            "beneficial": beneficial, "alpha": float(f.alpha), "gamma": float(f.gamma),
            "beta": float(f.beta), "c": c_abs, "r_star": float(expit(c_abs / 2.0)),
            "u_max": u_max, "u_q": u_q, "cutoff_risk": float(expit(u_q)),
            "check_val": check_val, "posted_gap": gap_posted,
            "r0_ctx": r0_ctx, "r1_ctx": r1_ctx, "auc_p0_oof": float(f.auc_p0_oof),
            "diagnostic": diag_res, "best_hp_dr_strong": best_hp,
        }
        print(f"  {t:22s} ({time.time()-t0:4.1f}s) | benef={str(beneficial):5s} | cutoff_r={100*expit(u_q):4.1f}% | "
              f"check={check_val:+5.2f} | posted_gap={10*gap_posted if gap_posted is not None else 0:5.0f} | sha={h_sha[:12]}", flush=True)

    (R / "w3_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("POSTED W3 MANIFEST:", R / "w3_manifest.json")

    # ------------------------------------------------------------------
    # STAGE 3: Verify hashes & score ONCE on sealed test splits (W3-H1 to W3-H4)
    # ------------------------------------------------------------------
    print("\n=== STAGE 3: Sealed Scoring (Hash-Verified) ===")
    scored_trials = {}
    BIN_EDGES = [(-1.0, -0.35), (-0.35, -0.15), (-0.15, 0.05), (0.05, 0.25), (0.25, 0.38), (0.38, 1.0)]
    BIN_LABELS = ["< -0.35", "-0.35 to -0.15", "-0.15 to +0.05 (peak)", "+0.05 to +0.25", "+0.25 to +0.38", "> +0.38"]
    bin_recs = {lbl: [] for lbl in BIN_LABELS}
    downturn_t = {}
    mse_t = {}

    for t, meta in metas.items():
        pp = PRED_W3 / f"{t}.parquet"
        assert sha256_file(pp) == manifest[t]["sha256"], f"Hash mismatch for {t}!"
        pred = pd.read_parquet(pp)
        sealed = pd.read_parquet(SEALED / f"{t}_test.parquet")
        smeta = json.loads((SEALED / f"{t}_meta.json").read_text())
        d = sealed.merge(pred, on="row_id", how="inner", validate="1:1").reset_index(drop=True)
        w = (d["action"] == smeta["action"]).to_numpy(int)
        if smeta.get("p_action") is not None:
            e = np.full(len(d), smeta["p_action"] / (smeta["p_action"] + smeta["p_reference"]))
        else:
            p = d["propensity"].to_numpy(float)
            e = np.where(w == 1, p, 1.0 - p)
        g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float), d["m1"].to_numpy(float), smeta["better"])
        cl = d["cluster_id"].to_numpy()

        scores = {k: d[k].to_numpy(float) for k in ("risk", "scale", "benefit", "response", "rose", "path_int", "dr_strong", "scale_70")}
        pairs = [
            ("scale", "risk"), ("response", "risk"), ("benefit", "risk"), ("rose", "risk"),
            ("path_int", "risk"), ("dr_strong", "risk"),
            ("benefit", "scale"), ("benefit", "response"), ("rose", "scale"), ("rose", "response"),
            ("scale", "response"), ("path_int", "scale"),
            ("dr_strong", "scale_70"), ("dr_strong", "scale"), ("dr_strong", "response"),
            ("dr_strong", "benefit"), ("scale_70", "scale"),
        ]
        res_t = s50.fast_boot_contrasts(
            g, scores, cl, pairs=pairs, q=0.2,
            include_vs_ate=["risk", "scale", "response", "benefit", "dr_strong"],
            plugin_scores={"scale": scores["scale"], "benefit": scores["benefit"], "rose": scores["rose"], "dr_strong": scores["dr_strong"]},
        )
        # Also test W3-H1 on beneficial trials
        ci = ctx_info[t]
        if ci["beneficial"]:
            m0 = np.clip(d["m0"].to_numpy(float), 1e-4, 1.0 - 1e-4)
            lin = ci["alpha"] + ci["gamma"] * logit(m0)
            r_bad = expit(lin) if smeta["better"] == "lower" else expit(-lin)
            c_abs = ci["c"]
            r_star = ci["r_star"]
            tau_or = expit(logit(np.clip(r_bad, 1e-6, 1 - 1e-6))) - expit(logit(np.clip(r_bad, 1e-6, 1 - 1e-6)) - c_abs)
            rd_ctx = max(0.0, ci["r0_ctx"] - ci["r1_ctx"])
            rr_bad_rel = max(0.0, (ci["r0_ctx"] - ci["r1_ctx"]) / max(ci["r0_ctx"], 1e-6))
            tau_rr_bad = np.clip(rr_bad_rel * r_bad, 0.0, r_bad)
            tau_rd = np.full_like(r_bad, rd_ctx)
            # Paired MSE
            mse_pair = {}
            for nm, a_arr, b_arr in [("or_minus_rr_bad", tau_or, tau_rr_bad), ("or_minus_rd", tau_or, tau_rd)]:
                diff_sq = ((g - a_arr) ** 2 - (g - b_arr) ** 2) * 1e4
                em, sm, G = cluster_mean_se(diff_sq, cl)
                mse_pair[nm] = {"est": em, "se": sm, "z": em / sm if sm > 0 else 0.0}
            mse_t[t] = {"domain": meta["domain"], **mse_pair}
            delta = r_bad - r_star
            for (lo_e, hi_e), lbl in zip(BIN_EDGES, BIN_LABELS):
                mask = (delta >= lo_e) & (delta < hi_e)
                if mask.sum() >= 50:
                    mu_g, se_g, _ = cluster_mean_se(1000.0 * g[mask], cl[mask])
                    bin_recs[lbl].append({
                        "trial": t, "domain": meta["domain"], "n": int(mask.sum()),
                        "obs": mu_g, "se": se_g,
                        "or": float(1000.0 * tau_or[mask].mean()),
                        "rr_bad": float(1000.0 * tau_rr_bad[mask].mean()),
                        "rd": float(1000.0 * tau_rd[mask].mean()),
                        "mean_delta": float(delta[mask].mean()),
                    })
            m_pk = (delta >= -0.15) & (delta <= 0.15)
            m_hi = delta > 0.25
            if m_pk.sum() >= 50 and m_hi.sum() >= 50:
                de, dse, _ = cluster_diff_se(1000.0 * g, m_hi, m_pk, cl)
                downturn_t[t] = {
                    "domain": meta["domain"], "n_peak": int(m_pk.sum()), "n_high": int(m_hi.sum()),
                    "obs_high_minus_peak": de, "se_diff": dse,
                    "pred_or_diff": float(1000.0 * (tau_or[m_hi].mean() - tau_or[m_pk].mean())),
                    "pred_rr_diff": float(1000.0 * (tau_rr_bad[m_hi].mean() - tau_rr_bad[m_pk].mean())),
                }

        scored_trials[t] = {"meta": meta, "context": ci, "scored": res_t}
        p = res_t["pairs"]
        print(f"  {t:22s} ({meta['domain']:13s}): ATE={10*res_t['ate']['estimate']:+5.0f} | "
              f"Sc-R={10*p['scale_minus_risk']['estimate']:+5.0f} [{10*p['scale_minus_risk']['lo95']:+5.0f},{10*p['scale_minus_risk']['hi95']:+5.0f}] | "
              f"Resp-R={10*p['response_minus_risk']['estimate']:+5.0f} | DR-R={10*p['benefit_minus_risk']['estimate']:+5.0f} | "
              f"DRstr-Sc70={10*p['dr_strong_minus_scale_70']['estimate']:+5.0f} | OptDR={10*res_t['optimism']['benefit']['overstatement']:+5.0f}", flush=True)

    # Pool Wave 3 results (all 6, health_uptake 4, beneficial health_uptake, medicine 2)
    all_w3 = list(metas.keys())
    hu_all = [t for t in all_w3 if metas[t]["domain"] == "health_uptake"]
    hu_ben = [t for t in hu_all if ctx_info[t]["beneficial"]]
    med_w3 = [t for t in all_w3 if metas[t]["domain"] == "medicine"]
    ben_w3 = [t for t in all_w3 if ctx_info[t]["beneficial"]]

    def pool_pair(ts, key, skip_zero=True):
        es, sds = [], []
        for t in ts:
            r = scored_trials[t]["scored"]["pairs"][key]
            if skip_zero and r["boot_sd"] < 1e-9 and abs(r["estimate"]) < 1e-9:
                continue
            es.append(r["estimate"]); sds.append(r["boot_sd"])
        return s50.re_pool(es, sds, floor=mr.V_FLOOR)

    def pool_opt(ts, nm):
        es, sds, pls, hos, ov = [], [], [], [], 0
        for t in ts:
            r = scored_trials[t]["scored"]["optimism"][nm]
            if nm == "scale" and r["boot_sd"] < 1e-9 and abs(r["overstatement"]) < 1e-9:
                continue
            es.append(r["overstatement"]); sds.append(r["boot_sd"])
            pls.append(r["plugin"]); hos.append(r["heldout"])
            ov += int(r["overstatement"] > 0)
        p_ = s50.re_pool(es, sds, floor=mr.V_FLOOR)
        p_["mean_plugin"] = float(np.mean(pls)) if pls else 0.0
        p_["mean_heldout"] = float(np.mean(hos)) if hos else 0.0
        p_["n_overstated"] = ov
        p_["n_trials"] = len(es)
        return p_

    def pool_ate(ts, nm):
        es = [scored_trials[t]["scored"]["vs_ate"][nm]["estimate"] for t in ts]
        sds = [scored_trials[t]["scored"]["vs_ate"][nm]["boot_sd"] for t in ts]
        return s50.re_pool(es, sds, floor=mr.V_FLOOR)

    # Pool binned curve & downturn & MSE on Wave 3 beneficial trials
    w3_binned = {}
    for lbl, recs in bin_recs.items():
        if recs:
            p_b = s50.re_pool([r["obs"] for r in recs], [r["se"] for r in recs])
            v = np.array([r["se"] ** 2 for r in recs])
            w_re = 1.0 / (v + p_b.get("tau", 0.0) ** 2); w_re /= w_re.sum()
            p_b["or_mean"] = float(np.sum(w_re * np.array([r["or"] for r in recs])))
            p_b["rr_bad_mean"] = float(np.sum(w_re * np.array([r["rr_bad"] for r in recs])))
            p_b["total_test_n"] = int(sum(r["n"] for r in recs))
            w3_binned[lbl] = p_b

    w3_downturn = s50.re_pool([v["obs_high_minus_peak"] for v in downturn_t.values()],
                              [v["se_diff"] for v in downturn_t.values()]) if downturn_t else {"k": 0}
    w3_mse = {}
    for cmp_nm in ("or_minus_rr_bad", "or_minus_rd"):
        sub = [v[cmp_nm] for v in mse_t.values()]
        p_m = s50.re_pool([s["est"] for s in sub], [s["se"] for s in sub]) if sub else {"k": 0}
        p_m["n_first_better"] = int(sum(s["est"] < 0 for s in sub))
        p_m["n_ci_first"] = int(sum(s["est"] + 1.96 * s["se"] < 0 for s in sub))
        p_m["n_ci_second"] = int(sum(s["est"] - 1.96 * s["se"] > 0 for s in sub))
        w3_mse[cmp_nm] = p_m

    pooled = {
        "all6": {k: pool_pair(all_w3, k) for k in ("scale_minus_risk", "response_minus_risk", "benefit_minus_risk", "rose_minus_risk", "dr_strong_minus_risk", "benefit_minus_scale", "benefit_minus_response", "dr_strong_minus_scale_70", "dr_strong_minus_response", "path_int_minus_scale")},
        "health_uptake_all4": {k: pool_pair(hu_all, k) for k in ("scale_minus_risk", "response_minus_risk", "benefit_minus_risk", "dr_strong_minus_risk", "benefit_minus_scale", "dr_strong_minus_scale_70")},
        "health_uptake_beneficial": {k: pool_pair(hu_ben, k) for k in ("scale_minus_risk", "response_minus_risk", "benefit_minus_risk", "dr_strong_minus_risk", "benefit_minus_scale", "dr_strong_minus_scale_70")},
        "medicine2": {k: pool_pair(med_w3, k) for k in ("scale_minus_risk", "response_minus_risk", "benefit_minus_risk", "dr_strong_minus_scale_70")},
        "vs_ate_all6": {nm: pool_ate(all_w3, nm) for nm in ("risk", "scale", "response", "benefit", "dr_strong")},
        "vs_ate_hu_ben": {nm: pool_ate(hu_ben, nm) for nm in ("risk", "scale", "response", "benefit", "dr_strong")},
        "optimism_all6": {nm: pool_opt(all_w3, nm) for nm in ("benefit", "rose", "dr_strong", "scale")},
        "w3_h1_binned": w3_binned,
        "w3_h1_downturn_per_trial": downturn_t,
        "w3_h1_downturn_pooled": w3_downturn,
        "w3_h1_mse_per_trial": mse_t,
        "w3_h1_mse_pooled": w3_mse,
    }

    out = {
        "preregistration_doi": "10.5281/zenodo.24487742",
        "trials": all_w3,
        "beneficial_trials": ben_w3,
        "health_uptake_trials": hu_all,
        "health_uptake_beneficial": hu_ben,
        "medicine_trials": med_w3,
        "pooled": pooled,
        "per_trial": scored_trials,
    }
    (R / "wave3_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nWROTE {R / 'wave3_results.json'} in {time.time() - t0_all:.1f}s")

    print("\n=== WAVE 3 SUMMARY (per 10,000 reached) ===")
    print("Beneficial trials:", ben_w3, "| Health-uptake beneficial:", hu_ben)
    for grp in ("all6", "health_uptake_beneficial", "medicine2"):
        print(f"\n  [{grp}]")
        for k, v in pooled[grp].items():
            if v.get("k", 0) > 0:
                print(f"    {k:28s} (k={v['k']}): {10*v['mean']:+5.0f} ({10*v['lo95']:+5.0f} to {10*v['hi95']:+5.0f})")
    print("\n  [W3-H3 Circular Plug-in Optimism across all 6 Wave-3 trials]")
    for nm, v in pooled["optimism_all6"].items():
        print(f"    {nm:10s} (k={v['n_trials']}): plug={10*v['mean_plugin']:+5.0f} held={10*v['mean_heldout']:+5.0f} over={10*v['mean']:+5.0f} ({10*v['lo95']:+5.0f} to {10*v['hi95']:+5.0f}) [{v['n_overstated']}/{v['n_trials']}]")
    print("\n  [W3-H1 Individual-level Mechanism on Wave 3 Beneficial Trials]")
    for lbl, b in w3_binned.items():
        print(f"    {lbl:22s} (k={b['k']}, n={b['total_test_n']:5d}): obs={10*b['mean']:+5.0f} ({10*b['lo95']:+5.0f} to {10*b['hi95']:+5.0f}) | OR={10*b['or_mean']:+4.0f} | RR={10*b['rr_bad_mean']:+4.0f}")
    print("    Downturn per trial:", {t: (round(10*v["obs_high_minus_peak"]), round(10*v["se_diff"]), round(10*v["pred_or_diff"])) for t, v in downturn_t.items()})
    if w3_downturn.get("k", 0) > 0:
        print(f"    Downturn pooled (k={w3_downturn['k']}): {10*w3_downturn['mean']:+.0f} ({10*w3_downturn['lo95']:+.0f} to {10*w3_downturn['hi95']:+.0f})")
    for nm, m in w3_mse.items():
        if m.get("k", 0) > 0:
            print(f"    Paired MSE {nm:16s} (k={m['k']}): {m['mean']:+.3f} ({m['lo95']:+.3f} to {m['hi95']:+.3f}) wins {m['n_first_better']}/{m['k']}")


if __name__ == "__main__":
    main()
