"""Camden Coalition hotspotting RCT (Finkelstein, Zhou, Taubman, Doyle; NEJM 2020).

782 patients in the analysis sample, randomized 1:1 to the Camden Core Model or usual care.
Outcome: any readmission within 180 days of index discharge (lower is better).
State: demographics, payer, pre-index utilization (claims), baseline survey, and the trial's own
pre-enrollment risk scores (TriageRiskScore, TotalRiskScore). Fields measured after assignment
(contact counts, enrollment duration, follow-up timing, discharge disposition) are excluded.
License: CC0 1.0 (Harvard Dataverse doi:10.7910/DVN/ZJVVQZ).
"""

import zipfile

import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

INFO = DatasetInfo(
    name="camden", tier="C", outcome_name="readmit_180d", better="lower",
    reference="usual_care", actions=("usual_care", "camden_core_model"),
    license="CC0-1.0", commercial_use=True,
    citation="Finkelstein A, Zhou A, Taubman S, Doyle J. Health care hotspotting: a randomized, "
    "controlled trial. N Engl J Med 2020;382:152-162. doi:10.1056/NEJMsa1906848",
    risk_field="x_TotalRiskScore",
)
# Replication package Table 4, row 1 (percentage points): control 61.70, treatment 62.34.
PUBLISHED = {"source": "Table 4 (replication package)", "arm": "camden_core_model",
             "metric": "diff", "value": 0.0064, "tol": 0.0005}

EXCLUDE_PREFIX = ("post_", "Ipost_", "Ireadmit", "first_readmit", "endtime", "days_cnt",
                  "Idischarged_", "Isample_", "Iconsented", "Icontrol", "Itreatment")


def load() -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "camden" / "dataverse_files.zip")
    base = "replication_package/datasets/"
    claims = pd.read_stata(z.open(base + "analysis_claims.dta"), convert_categoricals=False)
    sub = pd.read_stata(z.open(base + "analysis_tv_2_MAN_REV_SUB.dta"), convert_categoricals=False)
    claims = claims[claims["Isample_180"] == 1]
    survey = sub[["RCT_ID_TV2", "TriageRiskScore", "TotalRiskScore"]]
    d = claims.merge(survey, on="RCT_ID_TV2", how="left", validate="1:1")
    keep = [c for c in d.columns if c != "RCT_ID_TV2" and not c.startswith(EXCLUDE_PREFIX)]
    x = d[keep]
    action = d["Itreatment"].map({1.0: "camden_core_model", 0.0: "usual_care"})
    y = d["Ipost_ie_admit_180_IP_100"] / 100.0
    return frame("camden", x, action, y, INFO, propensity=0.5, row_id=d["RCT_ID_TV2"])
