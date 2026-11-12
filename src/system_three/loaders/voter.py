"""Michigan social-pressure mailing (Gerber, Green, Larimer; APSR 2008).

344,084 registered voters in 180,002 households. Households were randomized to control or one of
four mailings (Hawthorne, civic duty, neighbors, self). Outcome: voted in the August 2006 primary
(higher is better). Clusters are households; the dataset's `cluster` field is the randomization
block and is used as the stratum. Design probabilities: control 5/9, each mailing 1/9.
License: CC0 1.0 (Yale Dataverse doi:10.60600/YU/CGMWNW).
"""

import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

ARMS = {0: "control", 1: "hawthorne", 2: "civic_duty", 3: "neighbors", 4: "self"}
ARM_PROBS = {"control": 5 / 9, "hawthorne": 1 / 9, "civic_duty": 1 / 9, "neighbors": 1 / 9, "self": 1 / 9}

INFO = DatasetInfo(
    name="voter", tier="B", outcome_name="voted_aug2006_primary", better="higher",
    reference="control", actions=tuple(ARMS.values()), license="CC0-1.0", commercial_use=True,
    citation="Gerber AS, Green DP, Larimer CW. Social pressure and voter turnout: evidence from a "
    "large-scale field experiment. Am Polit Sci Rev 2008;102(1):33-48. doi:10.1017/S000305540808009X",
    risk_field=None,  # "risk" here = predicted non-voting under control; built in Phase 2
)
# Authors' R output, unadjusted OLS: neighbors coefficient 0.08131.
PUBLISHED = {"source": "GerberGreenLarimer_APSR_2008_r_output.txt", "arm": "neighbors",
             "metric": "diff", "value": 0.08131, "tol": 0.0005}


def load() -> pd.DataFrame:
    v = pd.read_stata(RAW / "voter" / "GerberGreenLarimer_APSR_2008_social_pressure.dta",
                      convert_categoricals=False)
    x = v[["sex", "yob", "g2000", "g2002", "p2000", "p2002", "p2004", "hh_size"]]
    action = v["treatment"].astype(int).map(ARMS)
    df = frame("voter", x, action, v["voted"], INFO, propensity=action.map(ARM_PROBS),
               # hh_id is stored as float and 2 of 180,002 ids are non-integer; keep as stored.
               cluster=v["hh_id"], stratum=v["cluster"].astype(int))
    df.attrs["arm_probs"] = ARM_PROBS
    return df
