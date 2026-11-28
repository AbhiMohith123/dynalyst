"""The readable ODE report (get_odes / inspect_model / CRNT's converted_model):
ODEs with fixed species and compartments by name, reactions and rate laws,
assignment rules, and every value -- for normal, rule-based, time-dependent,
XPP and crnt-converted models."""
import json

import numpy as np
import pytest
import sympy as sp

from dynalyst.analysis.ode_model import get_odes, model_from_sbml, ode_report, resolve_model

MODELS = ["Markevich2004", "pitchfork_model.xml", "Garde2020", "Elowitz_Nature2000",
          "Fiedler_BMC2016", "brannmark", "Bertozzi2020", "Borghans_BiophysChem1997"]


@pytest.mark.parametrize("name", MODELS)
def test_the_readable_odes_are_the_equations_the_analysis_uses(name):
    m = model_from_sbml(name)
    disp = m["display"]
    values = {sp.Symbol(k): v["value"] for k, v in disp["constants"].items()}
    rng = np.random.default_rng(1)
    for _ in range(3):
        point = {s: float(v) for s, v in zip(m["state_symbols"] + m["param_symbols"],
                                             rng.uniform(0.5, 2.0, len(m["state_symbols"]) + len(m["param_symbols"])))}
        for readable, used in zip(disp["rhs"], m["rhs"]):
            e = sp.sympify(readable)
            for _ in range(len(disp["assignment_rules"]) + 1):     # rules may use earlier rules
                e = e.xreplace({sp.Symbol(v): r for v, r in disp["assignment_rules"].items()})
            a = float(e.xreplace(values).xreplace(point))
            b = float(sp.sympify(used).xreplace(point))
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12), (name, readable)


def test_markevich_report_names_the_fixed_enzymes_and_shows_each_reaction():
    r = json.loads(get_odes("Markevich2004"))
    assert "MAPKK" in r["odes"]["dM/dt"] and "50.0*" not in r["odes"]["dM/dt"]
    # clamped enzymes are parameters now (they can be continued in), still named as species
    assert r["fixed_species"] == {"MAPKK": 50.0, "MKP3": 100.0}
    assert r["parameters"]["MAPKK"] == 50.0
    v1 = next(x for x in r["reactions"] if x["id"] == "v1")
    assert v1["equation"] == "M -> Mp" and "k1cat" in v1["rate"]
    assert r["odes_as_reaction_rates"]["dM/dt"] == "-v_v1 + v_v4"
    assert r["parameters"]["k1cat"] == 0.01 and r["parameters"]["Km5"] == 78.0
    assert "1.0*" not in json.dumps(r["odes"])


def test_rule_based_sbml_keeps_its_rules_by_name():
    r = json.loads(get_odes("pitchfork_model.xml"))
    assert r["odes"] == {"dA/dt": "-r1 + r3", "dAPP/dt": "r2 - r4"}
    assert set(r["assignment_rules"]) == {"r1", "r2", "r3", "r4"}
    assert r["parameters"]["ATOT"] == 0.1


def test_xpp_files_are_read_and_unused_parameters_listed_apart():
    r = json.loads(get_odes("Full_model_for_SN.txt"))
    assert r["source"] == "xpp" and r["n_state_variables"] == 6
    assert r["parameters"]["KTOT"] == 500.0
    m = resolve_model("Full_model_for_SN.txt")
    assert set(r["parameters"]) | set(r.get("unused_parameters", {})) >= set(m["param_names"])


def test_a_time_dependent_model_reports_its_frozen_input():
    r = json.loads(get_odes("Fiedler_BMC2016"))
    assert "k1max" in r["parameters"] and "k1max" in r["time_dependent_inputs"]
    assert {"k10", "k11"} <= set(r["unused_parameters"])     # only the frozen input used them


@pytest.fixture(scope="module")
def crnt_result(tmp_path_factory):
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    return json.loads(crnt_bistability("Full_model_for_SN.txt", structure_only=True))


def test_crnt_reports_the_converted_model_with_its_odes_and_values(crnt_result):
    assert "error" not in crnt_result, crnt_result
    net = crnt_result["network"]
    assert isinstance(net["reactions"], list) and len(net["reactions"]) == net["n_reactions"] == 12
    cm = crnt_result["converted_model"]
    assert cm["conversion_check"]["reproduces_original_odes"] is True
    assert set(cm["odes"]) >= {"dAP/dt", "dKTOT_free/dt"}
    k = cm["rate_constants"]["k_re1"]
    assert k == {"value": 0.01, "stands_for": "k2", "original_parameters": ["k2"]}
    assert cm["parameters"]["k_re1"] == 0.01                 # read back from the written file
    assert "KTOT_free" in cm["restored_species"]
    assert cm["original_parameters"]["k2"] == 0.01


def test_the_crnt_sbml_file_says_what_its_rate_constants_are(crnt_result):
    r = json.loads(get_odes(crnt_result["crnt_sbml"]))
    assert r["parameter_meanings"]["k_re1"] == "k2 in the original model"
    assert r["reactions"][0]["equation"] == "AK -> AP + KTOT_free"


def test_a_conversion_that_does_not_reproduce_the_odes_is_caught():
    from dynalyst.analysis.crnt_network import check_conversion
    net = {"species": ["x"], "hidden": {},
           "reactions": [{"id": "re1", "reactants": {"x": 1}, "products": {}, "rate_constant": "k"}]}
    assert check_conversion(net, ["x"], [-sp.Symbol("k") * sp.Symbol("x")], ["k"])["reproduces_original_odes"]
    bad = check_conversion(net, ["x"], [-2 * sp.Symbol("k") * sp.Symbol("x")], ["k"])
    assert bad["reproduces_original_odes"] is False and bad["max_relative_difference"] > 0.1


def test_parameter_names_that_are_sympy_objects_stay_parameters():
    # Garde2020 has a parameter Ge (sympy's GreaterThan); S, E, N are sympy objects too
    from dynalyst.analysis.crnt_network import check_conversion
    from dynalyst.analysis.ode_model import model_from_odes
    m = model_from_odes(["x"], {"x": "Ge - S*x"}, {"Ge": 2.0, "S": 3.0})
    assert [str(s) for s in m["param_symbols"]] == ["Ge", "S"]
    net = {"species": ["x"], "hidden": {},
           "reactions": [{"id": "re1", "reactants": {"x": 1}, "products": {}, "rate_constant": "S"},
                         {"id": "re2", "reactants": {}, "products": {"x": 1}, "rate_constant": "Ge"}]}
    assert check_conversion(net, ["x"], m["rhs"], ["Ge", "S"])["reproduces_original_odes"]


def test_long_results_lose_their_largest_fields_not_their_conclusion():
    from dynalyst.orchestration.tools import for_model
    out = json.dumps({"matrices": "x" * 20000, "conclusion": "Bistable.", "bistable": True})
    seen = json.loads(for_model(out, 2000))
    assert seen["conclusion"] == "Bistable." and seen["matrices"].startswith("<omitted")


def test_the_orchestrator_can_call_get_odes():
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.subagents import owner_of
    assert owner_of("get_odes") == "model_search"
    out = registry.execute("get_odes", {"model_name_or_path": "Markevich2004"})
    assert json.loads(out)["model_identity"]["biomodels_id"] == "BIOMD0000000027"
