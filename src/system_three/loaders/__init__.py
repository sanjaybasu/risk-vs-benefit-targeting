"""Registry: dataset name -> (INFO, PUBLISHED, load function).

Not loaded in Phase 1 (reported in the dataset table instead):
  crash2       public file has no treatment allocation column
  upworthy     aggregated per headline arm; fits the `choose` question, needs its own design
  llmrouterbench  JSON per model per benchmark; layout check pending
  lmarena      pairwise preferences only; cannot give every model's outcome per prompt
"""

from functools import partial

from system_three.loaders import camden, expansion, ist, tier_a, tier_b, tier_c, voter


def registry() -> dict:
    reg = {
        "camden": (camden.INFO, camden.PUBLISHED, camden.load),
        "voter": (voter.INFO, voter.PUBLISHED, voter.load),
        "ist": (ist.INFO, ist.PUBLISHED, ist.load),
        "illinois": (tier_c.ILLINOIS, tier_c.ILLINOIS_PUBLISHED, tier_c.load_illinois),
        "oregon": (tier_c.OREGON, tier_c.OREGON_PUBLISHED, tier_c.load_oregon),
        "hillstrom": (tier_b.HILLSTROM, tier_b.HILLSTROM_PUBLISHED, tier_b.load_hillstrom),
        "criteo": (tier_b.CRITEO, tier_b.CRITEO_PUBLISHED, tier_b.load_criteo),
        "nsw": (tier_b.NSW, tier_b.NSW_PUBLISHED, tier_b.load_nsw),
        "ihdp": (tier_b.IHDP, tier_b.IHDP_PUBLISHED, tier_b.load_ihdp),
        "gusto": (expansion.GUSTO, expansion.GUSTO_PUBLISHED, expansion.load_gusto),
        "sf_email": (expansion.SF_EMAIL, expansion.SF_PUBLISHED, expansion.load_sf_email),
        "stepup": (expansion.STEPUP, expansion.STEPUP_PUBLISHED, expansion.load_stepup),
        "elaia1": (expansion.ELAIA1, expansion.ELAIA1_PUBLISHED, expansion.load_elaia1),
        "elaia2": (expansion.ELAIA2, expansion.ELAIA2_PUBLISHED, expansion.load_elaia2),
        "synbiotic": (expansion.SYNBIOTIC, expansion.SYNBIOTIC_PUBLISHED, expansion.load_synbiotic),
        "karlan_list": (expansion.KARLAN, expansion.KARLAN_PUBLISHED, expansion.load_karlan_list),
        "sixcity": (expansion.SIXCITY, expansion.SIXCITY_PUBLISHED, expansion.load_sixcity),
        "newhaven": (expansion.NEWHAVEN, expansion.NEWHAVEN_PUBLISHED, expansion.load_newhaven),
        "lenta": (expansion.LENTA, expansion.LENTA_PUBLISHED, expansion.load_lenta),
        "x5": (expansion.X5, expansion.X5_PUBLISHED, expansion.load_x5),
    }
    for pair in tier_a.RB_PAIRS:
        info = tier_a.rb_info(pair)
        reg[info.name] = (info, tier_a.RB_PUBLISHED, partial(tier_a.load_routerbench, pair))
    for pair in tier_a.swebench_pairs():
        info = tier_a.DatasetInfo(name=f"swebench__{pair}", **tier_a.SWE_INFO_BASE)
        reg[info.name] = (info, tier_a.RB_PUBLISHED, partial(tier_a.load_swebench, pair))
    return reg
