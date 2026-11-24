"""Pre-registered Continuous and Count Outcome Evaluation (W3-H5; Zenodo DOI: 10.5281/zenodo.24487742).

Tests directly how targeting behavior changes on continuous and count outcomes (visits, dollars,
earnings) compared with bounded binary outcomes, and quantifies what happens when a practitioner
binarizes a continuous/count outcome into Y > 0 out of modeling convenience.

Trials (all individually/household randomized, evaluated on exact 30% held-out splits using the
same salted cluster hash as Waves 1-2):
  1. stepup_visits    (N = 61,293): 4-week gym visit count (higher is better)
  2. karlan_amount    (N = 50,083): charitable donation USD (higher is better)
  3. hillstrom_spend  (N = 42,613): 2-week purchase spend USD (higher is better)
  4. jtpa_earnings    (N = 11,204): 30-month earnings USD (higher is better)
  5. ffh_amount       (N = 11,261): charitable donation USD (higher is better)
  6. illinois_spend   (N =  3,239): Year-1 monthly medical spending USD (lower is better)
  7. oregon_ed_visits (N = 24,646): ED visit count (lower is better; adverse ATE)

For each trial, compares four rankings at a 20% budget on the TRUE continuous/count outcome Y
(reported both in native units and in control-arm outcome SDs x 100 for cross-trial pooling):
  - baseline_mean:  rank by predicted baseline count/amount mu0(x) (exact optimal under a
                    constant rate ratio / flat percentage effect)
  - bin_risk:       binarize Y > 0 and rank by predicted risk of undesirable binary outcome
  - bin_scale:      binarize Y > 0 and rank by 3-parameter constant-OR implied binary benefit
  - cont_dr:        continuous DR-learner fit directly on native-scale AIPW pseudo-outcomes

Also checks circular plug-in optimism on the continuous DR-learner vs held-out sealed AIPW.
Output: results/continuous_count_results.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SEALED = ROOT.parent / "system-three-sealed"
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

spec_mr = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec_mr)
sys.modules["mr"] = mr
spec_mr.loader.exec_module(mr)

spec_s50 = importlib.util.spec_from_file_location("s50", ROOT / "scripts" / "50_ben_additional_exploratory.py")
s50 = importlib.util.module_from_spec(spec_s50)
spec_s50.loader.exec_module(s50)

from system_three import rankers  # noqa: E402
from system_three.estimators.dr import aipw_scores, sign  # noqa: E402
from system_three.splits import assign as split_assign  # noqa: E402

SEED = 20261112


def _load_stepup_visits() -> tuple[pd.DataFrame, dict]:
    z = zipfile.ZipFile(RAW / "stepup" / "Data_April21.zip")
    cols = ["participant_id", "week", "visits", "any_visit", "cohort", "age", "signup_date_local",
            "customer_state", "gender", "exp_condition", "new_member"]
    w = pd.read_csv(z.open("Data/StepUp Data/pptdata.csv"), usecols=cols)
    pre = w[(w["week"] >= -52) & (w["week"] <= 0)]
    dur = w[(w["week"] >= 1) & (w["week"] <= 4)]
    g = pre.groupby("participant_id")
    feats = pd.DataFrame({
        "pre_weeks_obs": g["week"].size(),
        "pre_any_visit_rate": g["any_visit"].mean(),
        "pre_visits_per_week": g["visits"].mean(),
        "pre_any_visit_last4": pre[pre["week"] >= -3].groupby("participant_id")["any_visit"].mean(),
        "pre_any_visit_last12": pre[pre["week"] >= -11].groupby("participant_id")["any_visit"].mean(),
    })
    y_cnt = dur.groupby("participant_id")["visits"].sum().rename("y_cont")
    base = w.drop_duplicates("participant_id").set_index("participant_id")
    d = base.join(feats).join(y_cnt, how="inner").reset_index()
    signup = pd.to_datetime(d["signup_date_local"], errors="coerce")
    x = feats.reindex(d["participant_id"]).reset_index(drop=True)
    x["age"] = d["age"]; x["male"] = (d["gender"] == "M").astype(float)
    x["new_member"] = d["new_member"]; x["signup_year"] = signup.dt.year + signup.dt.dayofyear / 365.25
    for s in d["customer_state"].value_counts().index[:8]:
        x[f"state_{s}"] = (d["customer_state"] == s).astype(float)
    treated = (d["exp_condition"] != "Placebo Control").astype(int).to_numpy()
    p_c = pd.Series(treated).groupby(d["cohort"]).transform("mean").to_numpy()
    return _mk("stepup", "stepup_visits", d["participant_id"].astype(str), x, treated,
               d["y_cont"].to_numpy(float), p_c, "higher", "4-week gym visits")


def _load_karlan_amount() -> tuple[pd.DataFrame, dict]:
    from system_three.loaders.expansion import KARLAN_X
    d = pd.read_stata(RAW / "karlan_list" / "AER merged.dta", convert_categoricals=False)
    w = (d["treatment"] == 1).astype(int).to_numpy()
    p = np.full(len(d), 2.0 / 3.0)
    return _mk("karlan_list", "karlan_amount", pd.Series(np.arange(len(d)).astype(str)),
               d[KARLAN_X].copy(), w, d["amount"].to_numpy(float), p, "higher", "Donation (USD)")


def _load_hillstrom_spend() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(RAW / "hillstrom" / "hillstrom.csv.gz")
    d = d[d["segment"].isin(["Mens E-Mail", "No E-Mail"])].reset_index(drop=True)
    w = (d["segment"] == "Mens E-Mail").astype(int).to_numpy()
    p = np.full(len(d), 0.5)
    x = pd.DataFrame({
        "recency": d["recency"].astype(float),
        "history": d["history"].astype(float),
        "mens": d["mens"].astype(float),
        "womens": d["womens"].astype(float),
        "newbie": d["newbie"].astype(float),
        "zip_urban": (d["zip_code"] == "Urban").astype(float),
        "zip_suburban": (d["zip_code"] == "Surburban").astype(float),
        "ch_web": (d["channel"] == "Web").astype(float),
        "ch_phone": (d["channel"] == "Phone").astype(float),
    })
    return _mk("hillstrom", "hillstrom_spend", pd.Series(np.arange(len(d)).astype(str)),
               x, w, d["spend"].to_numpy(float), p, "higher", "2-week spend (USD)")


def _load_jtpa_earnings() -> tuple[pd.DataFrame, dict]:
    from system_three.loaders.wave2_labor import _dta, _num, _dummies
    base = RAW / "wave2_jtpa" / "extracted" / "Data"
    h = _dta(base / "expfup1" / "stata" / "head_fp1.dta")
    b = _dta(base / "expbif" / "stata" / "expbif.dta")
    e = _dta(base / "replacement_files" / "stata" / "earns2.dta")
    h = h[h["target"].isin(["1", "2"]) & (h["excesst"] == "2")]
    d = h.merge(e, on="recid", how="inner").merge(b, on="recid", how="left", suffixes=("", "_bif"))
    ern_cols = [f"newern{m:02d}" for m in range(1, 31)]
    d = d[d[ern_cols].notna().all(axis=1)].reset_index(drop=True)
    y_cont = d[ern_cols].sum(axis=1).to_numpy(float)
    m9, m99 = (9,), (99,)
    ra = pd.to_datetime(d["ra_dt"], format="%y%m%d", errors="coerce")
    x = pd.DataFrame({
        "female": (d["target"] == "1").astype(float), "age": _num(d["age"], m99),
        "black": (d["race"] == "2").astype(float), "hispanic": (d["race"] == "3").astype(float),
        "other_race": d["race"].isin(["4", "5"]).astype(float),
        "highest_grade": _num(d["gradcmpt"], m99), "hs_or_ged": _num(d["hsgrdged"], m9),
        "married": _num(d["mstatmar"], m9), "never_married": _num(d["mstatsi"], m9),
        "wid_div_sep": _num(d["mstatwds"], m9), "hh_size": _num(d["bfhhsize"], m99),
        "n_children_lt19": _num(d["numlt19"], m99), "child_under4": _num(d["chunder4"], m9),
        "hh_afdc": _num(d["hhafdc"], m9), "hh_food_stamps": _num(d["hhfdst"], m9),
        "hh_gen_assist": _num(d["genass"], m9), "afdc_now": _num(d["afdcnow"], m9),
        "employed_at_app": _num(d["curempl"], m9), "ever_worked": _num(d["wkforpy"], m9),
        "weeks_worked_12m": _num(d["weekswrk"], m99), "earnings_12m": _num(d["yearearn"], (99999,)),
        "english_primary": _num(d["english"], m9), "ra_month": (ra.dt.year - 1987) * 12 + ra.dt.month - 11,
    })
    x = pd.concat([x, _dummies(d["sda"], "site"), _dummies(d["treatmt"], "strategy")], axis=1)
    w = (d["tcstat"] == "1").astype(int).to_numpy()
    p = np.full(len(d), 2.0 / 3.0)
    return _mk("w2_jtpa", "jtpa_earnings", d["recid"].astype(str), x, w, y_cont, p, "higher", "30-month earnings (USD)")


def _load_ffh_amount() -> tuple[pd.DataFrame, dict]:
    from system_three.loaders.wave2 import _dta, FFH_X
    d = _dta(RAW / "wave2_ffh_charity" / "ffh_analysis.orig")
    d = d[(d["HS_round"] == 1) & d["Treatment_f1"].isin([1, 2])].reset_index(drop=True)
    x = d[FFH_X].apply(pd.to_numeric, errors="coerce")
    mailer = pd.Timestamp("2007-06-15")
    last = pd.to_datetime(d["date_lastpre"], errors="coerce")
    x["months_since_last_gift"] = (mailer - last).dt.days / 30.44
    y_cont = pd.to_numeric(d["amount_after"], errors="coerce").fillna(0.0).to_numpy(float)
    w = (d["Treatment_f1"] == 1).astype(int).to_numpy()
    p = np.full(len(d), 0.5)
    return _mk("w2_ffh_charity", "ffh_amount", d["p_id"].astype(str), x, w, y_cont, p, "higher", "Donation (USD)")


def _load_illinois_spend() -> tuple[pd.DataFrame, dict]:
    ctx = pd.read_parquet(PROC / "illinois" / "context.parquet")
    dev = pd.read_parquet(PROC / "illinois" / "dev.parquet")
    tst_s = pd.read_parquet(PROC / "illinois" / "test_state.parquet")
    tst_y = pd.read_parquet(SEALED / "illinois_test.parquet")
    tst = tst_s.merge(tst_y, on=["row_id", "cluster_id", "stratum"], validate="1:1")
    ctx["split"] = "context"; dev["split"] = "dev"; tst["split"] = "test"
    df = pd.concat([ctx, dev, tst], ignore_index=True)
    ref = ctx["reference"].iat[0]
    act = next(a for a in ctx["action"].unique() if a != ref)
    df["W"] = (df["action"] == act).astype(int)
    df["e"] = np.where(df["W"] == 1, df["propensity"], 1.0 - df["propensity"])
    df["y_cont"] = df["outcome"].astype(float)
    df["y_bin"] = (df["y_cont"] > 0).astype(float)
    meta = {"name": "illinois_spend", "label": "Illinois wellness medical spend", "better": "lower",
            "unit": "Monthly spend (USD)", "n": len(df)}
    return df, meta


def _load_oregon_ed_visits() -> tuple[pd.DataFrame, dict]:
    import io
    z = zipfile.ZipFile(RAW / "oregon" / "oregon_puf.zip")
    read_dta = lambda fn: pd.read_stata(io.BytesIO(z.open(fn).read()), convert_categoricals=False)  # noqa: E731
    desc = read_dta("OHIE_Public_Use_Files/OHIE_Data/oregonhie_descriptive_vars.dta")
    ed = read_dta("OHIE_Public_Use_Files/OHIE_Data/oregonhie_ed_vars.dta")
    d = desc.merge(ed, on="person_id", how="inner")
    d = d[d["sample_ed"] == 1].reset_index(drop=True)
    w = (d["treatment"] == 1).astype(int).to_numpy()
    p_s = pd.Series(w).groupby(d["numhh_list"].fillna(1)).transform("mean").to_numpy()
    x = pd.DataFrame({
        "birthyear": pd.to_numeric(d["birthyear_list"], errors="coerce"),
        "female": pd.to_numeric(d["female_list"], errors="coerce"),
        "english": pd.to_numeric(d["english_list"], errors="coerce"),
        "self_list": pd.to_numeric(d["self_list"], errors="coerce"),
        "first_day": pd.to_numeric(d["first_day_list"], errors="coerce"),
        "have_phone": pd.to_numeric(d["have_phone_list"], errors="coerce"),
        "pobox": pd.to_numeric(d["pobox_list"], errors="coerce"),
        "zip_msa": pd.to_numeric(d["zip_msa_list"], errors="coerce"),
        "numhh": pd.to_numeric(d["numhh_list"], errors="coerce"),
        "pre_any_ed": pd.to_numeric(d["any_visit_pre_ed"], errors="coerce"),
        "pre_num_ed": pd.to_numeric(d["num_visit_pre_cens_ed"], errors="coerce"),
        "pre_any_hosp": pd.to_numeric(d["any_hosp_pre_ed"], errors="coerce"),
        "pre_num_hosp": pd.to_numeric(d["num_hosp_pre_cens_ed"], errors="coerce"),
        "pre_any_out": pd.to_numeric(d["any_out_pre_ed"], errors="coerce"),
        "pre_num_out": pd.to_numeric(d["num_out_pre_cens_ed"], errors="coerce"),
    })
    y_cont = pd.to_numeric(d["num_visit_cens_ed"], errors="coerce").fillna(0.0).to_numpy(float)
    return _mk("oregon", "oregon_ed_visits", d["household_id"].astype(str), x, w, y_cont, p_s, "lower", "ED visits (18 mo)")


def _mk(fam: str, name: str, cid: pd.Series, x: pd.DataFrame, w: np.ndarray, y_cont: np.ndarray,
        e: np.ndarray, better: str, unit: str) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({
        "dataset": fam,
        "row_id": np.arange(len(w), dtype=np.int64),
        "cluster_id": cid.to_numpy(str),
        "W": w.astype(int),
        "e": e.astype(float),
        "y_cont": y_cont.astype(float),
        "y_bin": (y_cont > 0).astype(float),
    })
    for c in x.columns:
        df[f"x_{c}"] = pd.to_numeric(x[c], errors="coerce").astype(float)
    df["split"] = split_assign(df)
    meta = {"name": name, "label": name, "better": better, "unit": unit, "n": len(df)}
    return df, meta


def cross_fit_reg(X, w, y, groups):
    folds = rankers.folds_for(len(y), groups)
    m0, m1 = np.zeros(len(y)), np.zeros(len(y))
    f0, f1 = [], []
    for tr, te in folds:
        for arm, store, keep in ((0, m0, f0), (1, m1, f1)):
            idx = tr[w[tr] == arm]
            mdl = HistGradientBoostingRegressor(**rankers.GB).fit(X[idx], y[idx])
            store[te] = mdl.predict(X[te])
            keep.append(mdl)
    return m0, m1, f0, f1


def eval_trial(df: pd.DataFrame, meta: dict) -> dict:
    ctx = df[df["split"] == "context"].reset_index(drop=True)
    tst = df[df["split"] == "test"].reset_index(drop=True)
    Xc, cols = rankers.design_matrix(ctx)
    Xt, _ = rankers.design_matrix(tst, cols)
    wc, yc, ec, gc = ctx["W"].to_numpy(), ctx["y_cont"].to_numpy(), ctx["e"].to_numpy(), ctx["cluster_id"].to_numpy()
    yb_c = ctx["y_bin"].to_numpy()
    better = meta["better"]
    s_sign = sign(better)

    # 1. Continuous outcome nuisance models & Continuous DR-learner
    m0c, m1c, f0c, f1c = cross_fit_reg(Xc, wc, yc, gc)
    pseudo_c = m1c - m0c + wc * (yc - m1c) / ec - (1 - wc) * (yc - m0c) / (1 - ec)
    tau_dr_mdl = HistGradientBoostingRegressor(**rankers.GB).fit(Xc, pseudo_c)
    m0t = np.mean([m.predict(Xt) for m in f0c], axis=0)
    m1t = np.mean([m.predict(Xt) for m in f1c], axis=0)
    dr_t = s_sign * tau_dr_mdl.predict(Xt)

    # 2. Baseline-count/mean ranking (exact optimal under constant rate ratio / flat % effect)
    # Larger baseline count mu0(x) -> larger absolute change |rho * mu0(x)| regardless of better direction!
    base_mean_t = m0t.copy()

    # 3. Practitioner convenience binarization: binarize Y > 0, fit binary risk & constant-OR scale
    f_bin = rankers.fit(Xc, wc, yb_c, ec, better, groups=gc)
    s_bin = rankers.scores(f_bin, Xt)
    bin_risk_t = s_bin["risk"]
    bin_scale_t = s_bin["scale"]

    # 4. Evaluate ALL 4 rankings on TRUE continuous/count outcome Y on held-out test split (AIPW)
    wt = tst["W"].to_numpy(int)
    yt = tst["y_cont"].to_numpy(float)
    et = tst["e"].to_numpy(float)
    clt = tst["cluster_id"].to_numpy()
    g_raw = s_sign * (m1t - m0t + wt * (yt - m1t) / et - (1 - wt) * (yt - m0t) / (1 - et))

    # Control-arm test SD for standardized cross-trial pooling (SD x 100 units)
    sd0 = float(np.std(yt[wt == 0], ddof=1))
    g_sd = g_raw / sd0  # in outcome SDs; fast_boot_contrasts multiplies by 1000 -> milli-SDs (divide by 10 -> SD x 100)

    scores = {
        "risk": bin_risk_t,         # binary risk ranking (from binarizing Y > 0)
        "bin_scale": bin_scale_t,   # binarized constant-OR scale ranking
        "base_mean": base_mean_t,   # baseline expected count/mean (exact under constant rate ratio!)
        "cont_dr": dr_t,            # continuous DR-learner on native Y
    }
    pairs = [
        ("base_mean", "bin_scale"), ("base_mean", "risk"),
        ("cont_dr", "base_mean"), ("cont_dr", "bin_scale"), ("cont_dr", "risk"),
        ("bin_scale", "risk"),
    ]
    res_sd = s50.fast_boot_contrasts(
        g_sd, scores, clt, pairs=pairs, q=0.2,
        include_vs_ate=["base_mean", "bin_scale", "risk", "cont_dr"],
        plugin_scores={"cont_dr": dr_t / sd0},
    )
    # Also native-unit point estimates & SEs (multiply g_sd contrasts by sd0 / 1000)
    def to_native(blk):
        f_ = sd0 / 1000.0
        return {k: float(blk[k] * f_) for k in ("estimate", "lo95", "hi95", "boot_sd")}

    # Check whether binary and continuous effects are beneficial in context
    ctx_ate = float(s_sign * np.mean(pseudo_c))
    bin_benef = bool((better == "higher" and f_bin.beta > 0) or (better == "lower" and f_bin.beta < 0))
    return {
        "meta": {**meta, "n_context": int(len(ctx)), "n_test": int(len(tst)), "control_sd": sd0,
                 "ctx_ate_native": ctx_ate, "ctx_ate_sd": ctx_ate / sd0,
                 "zero_share_ctrl": float(np.mean(yt[wt == 0] == 0)),
                 "bin_beneficial_ctx": bin_benef},
        "milli_sd": res_sd,
        "native": {
            "ate": to_native(res_sd["ate"]),
            "pairs": {k: to_native(v) for k, v in res_sd["pairs"].items()},
            "vs_ate": {k: to_native(v) for k, v in res_sd["vs_ate"].items()},
            "optimism_cont_dr": {
                "plugin": float(res_sd["optimism"]["cont_dr"]["plugin"] * sd0 / 1000.0),
                "heldout": float(res_sd["optimism"]["cont_dr"]["heldout"] * sd0 / 1000.0),
                "overstatement": float(res_sd["optimism"]["cont_dr"]["overstatement"] * sd0 / 1000.0),
                "lo95": float(res_sd["optimism"]["cont_dr"]["lo95"] * sd0 / 1000.0),
                "hi95": float(res_sd["optimism"]["cont_dr"]["hi95"] * sd0 / 1000.0),
            },
        },
    }


def main() -> None:
    t0 = time.time()
    loaders = [
        _load_stepup_visits,
        _load_karlan_amount,
        _load_hillstrom_spend,
        _load_jtpa_earnings,
        _load_ffh_amount,
        _load_illinois_spend,
        _load_oregon_ed_visits,
    ]
    per_trial = {}
    for fn in loaders:
        df, meta = fn()
        r = eval_trial(df, meta)
        per_trial[meta["name"]] = r
        p = r["milli_sd"]["pairs"]
        n_ = r["native"]["pairs"]
        m_ = r["meta"]
        print(f"  {meta['name']:18s} ({meta['better']:6s}, zero={100*m_['zero_share_ctrl']:4.1f}%, N_t={m_['n_test']:5d}) | "
              f"ATE={r['native']['ate']['estimate']:+7.2f} {meta['unit'][:8]} | "
              f"BaseMean-BinScale={p['base_mean_minus_bin_scale']['estimate']:+6.1f} mSD ({n_['base_mean_minus_bin_scale']['estimate']:+7.2f}) | "
              f"ContDR-BaseMean={p['cont_dr_minus_base_mean']['estimate']:+6.1f} mSD | "
              f"BaseMean-ATE={r['milli_sd']['vs_ate']['base_mean']['estimate']:+6.1f} mSD | "
              f"BinScale-ATE={r['milli_sd']['vs_ate']['bin_scale']['estimate']:+6.1f} mSD", flush=True)

    # Pool across beneficial-direction trials (ctx_ate_sd > 0) and across all 7 trials (in milli-SD units; divide by 10 for 0.01 SD)
    all7 = list(per_trial.keys())
    benef = [t for t in all7 if per_trial[t]["meta"]["ctx_ate_sd"] > 0]
    higher5 = [t for t in all7 if per_trial[t]["meta"]["better"] == "higher"]

    def pool_k(ts, getter):
        es, sds = [], []
        for t in ts:
            v = getter(per_trial[t]["milli_sd"])
            if v["boot_sd"] < 1e-9 and abs(v["estimate"]) < 1e-9:
                continue
            es.append(v["estimate"]); sds.append(v["boot_sd"])
        return s50.re_pool(es, sds, floor=0.0)

    pooled = {}
    for grp, ts in [("all7", all7), ("beneficial", benef), ("higher_is_better_5", higher5)]:
        pooled[grp] = {
            "trials": ts,
            "base_mean_minus_bin_scale": pool_k(ts, lambda d: d["pairs"]["base_mean_minus_bin_scale"]),
            "base_mean_minus_bin_risk": pool_k(ts, lambda d: d["pairs"]["base_mean_minus_risk"]),
            "cont_dr_minus_base_mean": pool_k(ts, lambda d: d["pairs"]["cont_dr_minus_base_mean"]),
            "cont_dr_minus_bin_scale": pool_k(ts, lambda d: d["pairs"]["cont_dr_minus_bin_scale"]),
            "base_mean_minus_ate": pool_k(ts, lambda d: d["vs_ate"]["base_mean"]),
            "bin_scale_minus_ate": pool_k(ts, lambda d: d["vs_ate"]["bin_scale"]),
            "bin_risk_minus_ate": pool_k(ts, lambda d: d["vs_ate"]["risk"]),
            "cont_dr_minus_ate": pool_k(ts, lambda d: d["vs_ate"]["cont_dr"]),
            "optimism_cont_dr": pool_k(ts, lambda d: d["optimism"]["cont_dr"]),
        }

    out = {
        "note": "Pre-registered continuous/count outcome evaluation (W3-H5, Zenodo doi:10.5281/zenodo.24487742). Standardized units are milli-SDs (1/1,000 of control-arm SD).",
        "pooled": pooled,
        "per_trial": per_trial,
    }
    (R / "continuous_count_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nWROTE {R / 'continuous_count_results.json'} in {time.time()-t0:.1f}s")
    print("\n=== POOLED CONTINUOUS/COUNT RESULTS (milli-SD = 0.001 SD of control outcome) ===")
    for grp in ("beneficial", "higher_is_better_5", "all7"):
        print(f"\n  [{grp}: {pooled[grp]['trials']}]")
        for k, v in pooled[grp].items():
            if isinstance(v, dict) and v.get("k", 0) > 0:
                print(f"    {k:28s} (k={v['k']}): {v['mean']:+6.1f} ({v['lo95']:+6.1f} to {v['hi95']:+6.1f}) P={v.get('p_two_sided', 1.0):.3f}")


if __name__ == "__main__":
    main()
