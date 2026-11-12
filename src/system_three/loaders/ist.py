"""International Stroke Trial (IST Collaborative Group; Lancet 1997), corrected dataset v2.

19,435 patients with suspected acute ischaemic stroke, randomized in a 2x2-style factorial to
aspirin versus avoid aspirin, and to heparin (low or medium dose) versus avoid heparin. This
loader uses the aspirin comparison; heparin allocation (randomized independently) is kept as a
stratum. Outcome: dead or dependent at 6 months (OCCODE 1 or 2; lower is better); patients with
unknown 6-month status are excluded. State: randomization-form fields only (R* columns, age, sex).
License: ODC-By 1.0 (Edinburgh DataShare handle 10283/128).
"""

import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

INFO = DatasetInfo(
    name="ist", tier="C", outcome_name="dead_or_dependent_6m", better="lower",
    reference="avoid_aspirin", actions=("avoid_aspirin", "aspirin"),
    license="ODC-By-1.0", commercial_use=True,
    citation="International Stroke Trial Collaborative Group. The International Stroke Trial (IST). "
    "Lancet 1997;349:1569-1581. Data: Sandercock PAG et al., Trials 2011;12:101.",
)
# Lancet 1997: dead or dependent at 6 months 62.2% aspirin vs 63.5% avoid aspirin (-1.3 points).
PUBLISHED = {"source": "IST Lancet 1997, 6-month dead or dependent", "arm": "aspirin",
             "metric": "diff", "value": -0.013, "tol": 0.004}

RFORM = ["RDELAY", "RCONSC", "SEX", "AGE", "RSLEEP", "RATRIAL", "RCT", "RVISINF", "RHEP24",
         "RASP3", "RSBP", "RDEF1", "RDEF2", "RDEF3", "RDEF4", "RDEF5", "RDEF6", "RDEF7", "RDEF8",
         "STYPE"]
CODES = {"Y": 1, "N": 0, "C": 0.5, "U": None, "M": 1, "F": 0, "D": 1, "A": 0}


def load() -> pd.DataFrame:
    d = pd.read_csv(RAW / "ist" / "IST_corrected.csv", low_memory=False, encoding="latin-1")
    d = d[d["OCCODE"].isin([1, 2, 3, 4])].reset_index(drop=True)
    x = pd.DataFrame(index=d.index)
    for c in RFORM:
        if d[c].dtype == object:
            if c in ("RCONSC", "STYPE"):
                x = x.join(pd.get_dummies(d[c], prefix=c, dtype=float))
            else:
                x[c] = d[c].map(CODES)
        else:
            x[c] = d[c]
    action = d["RXASP"].map({"Y": "aspirin", "N": "avoid_aspirin"})
    y = d["OCCODE"].isin([1, 2]).astype(float)
    return frame("ist", x, action, y, INFO, propensity=0.5, stratum=d["RXHEP"],
                 row_id=pd.Series(d.index))
