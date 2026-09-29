"""Every measurement parameter of experiment the lever run, in one place.

Inherited unchanged from the regime-dose lever run (and the first lever run / the showcase run before it): the OLMo-2-32B member, its layer-32
route-B pole bank (sha256 pinned below), the ``inj = injected_l2 / resid_rms`` dose convention,
the frozen 16-direction random pool (same generator, k and seed), the 2 + 2 + 8 turn structure,
the sampler (temperature 0.7, top-p 0.95, <= 200 tokens, per-turn seed ``conv*100 + turn*7 + 1``),
the Gated tool regime, and the blind judge.

New here: the dose axis is the paper's signed d (OLMo rho = 0.30, so |d| = 1 is injection ratio
0.30 and |d| = 0.5 is 0.15), seven conditions on that axis, a null cell whose tool applies the
positive pole, and the exposure-turn task sentence as a parameter (self-report vs the section 3.2
zone instruction with a nonce cue).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "steering_vectors" / "banks"
RESULTS_DIR = ROOT / "results"

MEMBER = "olmo_32b"
HF_ID = "allenai/OLMo-2-0325-32B-Instruct"
LAYER = 32
BANK_SHA256 = "d45d2af2348c50240f624966efe0c508fe5b7f674c26daa243c30a2bfe84263d"  # sha256 of the committed bank file; the run-time file (recorded in data/lever/analysis.json) differed only in two corpus-source path strings  # sha256 of the committed bank file; the run-time file (recorded in data/lever/analysis.json) differed only in two corpus-source path strings

# The frozen orthonormal random pool (the showcase run config: RANDOM_K, RANDOM_POOL_SEED). Direction
# ``conv % RANDOM_K`` is used for conversation ``conv``, exactly as the first lever run and the regime-dose lever run random cells.
RANDOM_K = 16
RANDOM_POOL_SEED = 20260723

# Conversation structure (the first lever run Part C / the regime-dose lever run Gated).
REGIME = "gated"
N_BASELINE = 2
N_EXPOSURE = 2
N_OFFER = 8
MAX_TOKENS = 200
TEMPERATURE = 0.7
SEED_STRIDE = 100          # conversation k uses seed k*100; turn t adds t*7 + 1

# The paper's dose axis. d = +-1 <-> injection ratio RHO; d = +-0.5 <-> RHO / 2.
RHO = 0.30


def ratio_for(d_abs: float) -> float:
    return round(float(d_abs) * RHO, 6)


# Seven conditions. ``op_role``/``op_dose``: what the operator imposes in the exposure turns and
# offers 1-7 (``zero`` = nothing). ``tool_role``/``tool_dose``: the vector adjust_context applies
# and its intensity ceiling. ``kind``: removal (operator re-imposes in offers 1-7) or null.
CONDITIONS: dict[str, dict] = {
    "neg_d1":   {"d": -1.0, "op_role": "neg",  "op_dose": ratio_for(1.0), "tool_role": "neg",  "tool_dose": ratio_for(1.0), "kind": "removal"},
    "neg_d05":  {"d": -0.5, "op_role": "neg",  "op_dose": ratio_for(0.5), "tool_role": "neg",  "tool_dose": ratio_for(0.5), "kind": "removal"},
    "null":     {"d":  0.0, "op_role": "zero", "op_dose": 0.0,            "tool_role": "pos",  "tool_dose": ratio_for(1.0), "kind": "null"},
    "pos_d05":  {"d": +0.5, "op_role": "pos",  "op_dose": ratio_for(0.5), "tool_role": "pos",  "tool_dose": ratio_for(0.5), "kind": "removal"},
    "pos_d1":   {"d": +1.0, "op_role": "pos",  "op_dose": ratio_for(1.0), "tool_role": "pos",  "tool_dose": ratio_for(1.0), "kind": "removal"},
    "rand_d05": {"d":  0.5, "op_role": "rand", "op_dose": ratio_for(0.5), "tool_role": "rand", "tool_dose": ratio_for(0.5), "kind": "removal", "abs_only": True},
    "rand_d1":  {"d":  1.0, "op_role": "rand", "op_dose": ratio_for(1.0), "tool_role": "rand", "tool_dose": ratio_for(1.0), "kind": "removal", "abs_only": True},
}
CONDITION_ORDER = tuple(CONDITIONS)

# Exposure prompts: the two arms differ only in the two exposure turns' task sentence.
PROMPTS = ("selfreport", "zone")
ZONE_CUE_K = 12            # cue = CUE_POOL[conv % 12], same cue on both exposure turns

GRID_N_CONV = 200
CHUNK = 25


def cell_name(prompt: str, cond: str) -> str:
    return f"gated_{prompt}_{cond}"


CELLS = tuple(cell_name(p, c) for p in PROMPTS for c in CONDITION_ORDER)   # 14

# Bootstrap for per-turn rates (cluster = conversation).
BOOT_N = 10_000
BOOT_SEED = 285
EQUIV_MARGIN = 0.05        # reading (c): |difference in per-turn rate between prompts| < 0.05

# Judge (the ConCon package judge; identity asserted against its recorded scores file).
JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
JUDGE_WORKERS = 12
JUDGE_REF = str(Path(__file__).resolve().parents[2] / "data" / "baselines" / "judge_scores.json")

# the regime-dose lever run reference (Gated, doses 0.50 and 0.60, i.e. d ~ 1.7 and 2.0), post-hoc per-turn rates.
REF_284 = {
    "reset_rate_per_operator_active_turn": {
        "gated_dstar(0.50)": {"neg": (0.14, 0.24), "rand": (0.05, 0.10), "pos": (0.04, 0.09)},
        "note": "ranges quoted across the four regime-dose pairs of the regime-dose lever run posthoc284.json",
    },
    "exposure_valence_gated_0.50": {"neg": 0.43, "rand": 0.9, "pos": 2.6},
    "table": ("<archive path> tree/"
              "experiments/the regime-dose lever run-pg6jpz/results/table284.csv"),
}
