"""Reference suite for run_bifurcation_analysis: models whose answers were verified independently
of the agent's tools (libsbml + scipy simulations and eigenvalues, or the equations by hand;
scripts in Integration_check/v31_live_runs_20261031/reference_scripts). Each case states where its numbers come from.

Slow (AUTO runs, about 15 minutes): run on its own, e.g.
    python -m pytest tests/test_reference_suite.py -q
Cases marked xfail are known open bugs; they flip to passing when the bug is fixed."""
import json

import pytest

pytestmark = pytest.mark.auto   # real AUTO-07p runs: skipped where AUTO is not built

from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis

# (id, call, expected). Expected keys:
#   hopf / folds: list of [lo, hi] brackets, one point in each, and no extra points of that type
#   types: {bracket index: "supercritical" | "subcritical"} for the Hopf points
CASES = [
    # v = -/+ sqrt(1 - 0.064), I = w - v + v^3/3 (analytic); both subcritical (canard explosion)
    ("fhn_I", dict(model_name_or_path="fhn_check.xml", param_name="I", param_min=0, param_max=2),
     dict(hopf=[[0.33127, 0.33129], [1.41871, 1.41873]],
          types={0: "subcritical", 1: "subcritical"})),
    # eigenvalue sign change and simulation (batch 12/16 reviews)
    ("biomd8_vi", dict(model_name_or_path="BIOMD0000000008", param_name="reaction1_vi",
                       param_min=0.01, param_max=1),
     dict(hopf=[[0.0571, 0.0574], [0.2924, 0.2931]],
          types={0: "supercritical", 1: "supercritical"})),
    # simulation: steady below v1_K1 = 4.5, oscillating from 5.0; eigenvalue screen 4.72
    ("biomd39_v1K1", dict(model_name_or_path="BIOMD0000000039", param_name="v1_K1",
                          param_min=2.36, param_max=7.5, plot_state="Ca_cyt"),
     dict(hopf=[[4.5, 5.0]])),
    # simulation: steady at v5_Kpump = 18.25, oscillating at 18.30
    ("biomd39_v5", dict(model_name_or_path="BIOMD0000000039", param_name="v5_Kpump",
                        param_min=10, param_max=30, plot_state="Ca_cyt"),
     dict(hopf=[[18.25, 18.30]])),
    # simulation: oscillates at BrO3 = 0.06 (period 48.75), with no Hopf point in the range
    ("biomd40_BrO3", dict(model_name_or_path="BIOMD0000000040", param_name="BrO3",
                          param_min=0.006, param_max=0.6, plot_state="Ce"),
     dict(hopf=[])),
    # rate-rule ODEs by hand: max Re(eigenvalue) changes sign within +/-2% of 0.6037
    ("biomd299_vs", dict(model_name_or_path="BIOMD0000000299", param_name="vs",
                         param_min=0.3, param_max=1.0),
     dict(hopf=[[0.5916, 0.6158]])),
    # simulation from low and high Mpp: one state at 39.0 and 57.6, two at 39.5 and 57.2
    ("markevich_MAPKK", dict(model_name_or_path="Markevich2004.xml", param_name="MAPKK",
                             param_min=0, param_max=100),
     dict(folds=[[39.0, 39.5], [57.2, 57.6]], hopf=[])),
    # eigenvalue sign change between 0.0717 and 0.0746; cycles stable just above (simulation)
    ("biomd79_a", dict(model_name_or_path="BIOMD0000000079", param_name="reaction_0_a",
                       param_min=0, param_max=0.2, plot_state="P"),
     dict(hopf=[[0.0717, 0.0746]], types={0: "supercritical"})),
    # eigenvalues: complex pair crosses between 0.0100/0.0108 and 0.0710/0.0740 (stiff model)
    ("biomd5_k1aa", dict(model_name_or_path="BIOMD0000000005", param_name="Reaction6_k1aa",
                         param_min=0.0015, param_max=0.15, plot_state="M"),
     dict(hopf=[[0.0100, 0.0108], [0.0710, 0.0740]])),
    # the start exactly at the Hopf point and a partial initial state (R3's call): same Hopf,
    # the total Ca kept at the model's 90
    ("biomd39_v5_start_at_hopf", dict(model_name_or_path="BIOMD0000000039", param_name="v5_Kpump",
                                      param_min=10, param_max=30, plot_state="Ca_cyt",
                                      fixed_params={"v5_Kpump": 18.261},
                                      initial_conditions={"Ca_cyt": 0.258899}),
     dict(hopf=[[18.25, 18.30]])),
    # batch 15 references (ref_b8_o1.py, ref_b15.py): both subcritical
    ("biomd3_vi", dict(model_name_or_path="BIOMD0000000003", param_name="reaction1_vi",
                       param_min=0, param_max=0.5, plot_state="C"),
     dict(hopf=[[0.01190, 0.01193], [0.23700, 0.23725]],
          types={0: "subcritical", 1: "subcritical"})),
    # batch 15 references: both supercritical
    ("biomd98_beta", dict(model_name_or_path="BIOMD0000000098", param_name="beta",
                          param_min=0, param_max=1, plot_state="Z"),
     dict(hopf=[[0.2885, 0.2894], [0.7735, 0.7750]],
          types={0: "supercritical", 1: "supercritical"})),
    # batch 15 references: number of stable states by simulation
    ("biomd203_gamma2", dict(model_name_or_path="BIOMD0000000203", param_name="gamma2",
                             param_min=0, param_max=5),
     dict(folds=[[0.0550, 0.0565], [0.8200, 0.8300]])),
    # batch 15 references (a = 0.1): both supercritical
    ("brusselator_b", dict(model_name_or_path="brusselator.xml", param_name="b",
                           param_min=0.05, param_max=0.85, fixed_params={"a": 0.1}),
     dict(hopf=[[0.10914, 0.10916], [0.77887, 0.77890]],
          types={0: "supercritical", 1: "supercritical"})),
    # batch 13 references: Vm2 Hopf 7.18183 supercritical (oscillations decay below it)
    ("borghans_vm2", dict(model_name_or_path="Borghans_BiophysChem1997.xml", param_name="Vm2",
                          param_min=0.01, param_max=7.2, plot_state="Z_state"),
     dict(hopf=[[7.1810, 7.1827]], types={0: "supercritical"})),
    # extracellular Ca (EC) is bookkeeping and may be negative. Simulation: steady at Vm3 = 18.3
    # and 75.4, oscillating at 19.1 and 74.5; simple stable cycle at 25 (period 0.333), complex
    # (bursting) oscillation at 50 and 70
    ("biomd43_vm3", dict(model_name_or_path="BIOMD0000000043", param_name="v3_Vm3"),
     dict(hopf=[[18.3, 19.1], [74.5, 75.4]], types={0: "supercritical"})),
]


def _points(result, kind, param):
    return sorted(float(p[param]) for p in result.get("bifurcation_points") or []
                  if p.get("type") == kind)


def _check_brackets(found, brackets, what):
    assert len(found) == len(brackets), f"{what}: expected {len(brackets)} in {brackets}, got {found}"
    for x, (lo, hi) in zip(found, sorted(brackets)):
        assert lo <= x <= hi, f"{what} at {x}, expected in [{lo}, {hi}] (all: {found})"


@pytest.mark.parametrize("case,call,expected", CASES)
def test_reference(case, call, expected):
    result = json.loads(run_bifurcation_analysis(**call))
    assert "error" not in result, result.get("error")
    if expected.get("runs"):
        return
    param = call["param_name"]
    hopf = _points(result, "HB", param)
    if "hopf" in expected:
        _check_brackets(hopf, expected["hopf"], "Hopf")
    if "folds" in expected:
        _check_brackets(_points(result, "LP", param), expected["folds"], "fold")
    for i, kind in (expected.get("types") or {}).items():
        lo, hi = sorted(expected["hopf"])[i]
        # a Hopf point's type: AUTO's first Lyapunov coefficient, on the Hopf point itself
        types = [str(p.get("hopf_type", "")) for p in result.get("bifurcation_points") or []
                 if p.get("type") == "HB" and lo <= float(p[param]) <= hi]
        assert types and types[0].startswith(kind), f"Hopf in [{lo}, {hi}]: {types}, expected {kind}"
