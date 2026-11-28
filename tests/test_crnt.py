"""Reaction network theory: network recovery from equations and crnt4sbml (CRNT only:
no continuation, no stability check)."""
import importlib.util
import json

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("crnt4sbml") is None,
                                reason="crnt4sbml not installed (pip install --no-deps crnt4sbml)")


def test_network_recovered_with_conserved_species_restored():
    from dynalyst.analysis.crnt_network import reconstruct
    from dynalyst.analysis.ode_model import resolve_model
    m = resolve_model("Full_model_for_SN")
    net = reconstruct(m["state_names"], m["rhs"], m["param_names"], m["param_defaults"], m["y0"])
    assert set(net["hidden"]) == {"ATOT_free", "KTOT_free", "PTOT_free"}
    rx = {(tuple(sorted(r["reactants"])), tuple(sorted(r["products"])), r["rate_constant"])
          for r in net["reactions"]}
    assert (("ATOT_free", "KTOT_free"), ("AK",), "k1") in rx     # A + K -> AK
    assert (("AK",), ("AP", "KTOT_free"), "k2") in rx             # AK -> AP + K
    assert len(net["reactions"]) == 12


def test_non_mass_action_and_deficiency_zero():
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    d = json.loads(crnt_bistability("Markevich2004"))
    assert d["mass_action"] is False and d["bistable"] is None
    d = json.loads(crnt_bistability(species=["A", "B"], odes={"A": "-k1*A + k2*B", "B": "k1*A - k2*B"},
                                    parameters={"k1": 1, "k2": 2}, initial_conditions={"A": 1, "B": 0}))
    assert d["bistable"] is False and d["deficiency_zero_theorem"] is True


def test_crnt4sbml_exit_becomes_an_answer():
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    d = json.loads(crnt_bistability("Garde2020"))
    assert d["bistable"] is None and "crnt4sbml cannot" in d["conclusion"]


def test_parameter_sets_are_reported_without_a_stability_claim():
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    d = json.loads(crnt_bistability("Full_model_for_SN", iterations=15))
    assert d["multistationary"] is True and d["bistable"] is None
    v = d["multistationary_sets"][0]
    assert v["search_range"] == "near the model's own values" and d["near_model"] is True
    assert set(v["compared_with_model"]["totals"]) == set(v["conservation_totals"])
    assert "candidate bistability parameter set" in d["conclusion"] and "AUTO-07p" in d["conclusion"]


def test_reaction_graph_is_one_connected_network_and_complex_graph_is_kept():
    # species and reactions are both nodes, so the network is drawn connected; the CRNT
    # complex graph (one part per linkage class) is a separate plot
    import networkx as nx
    from dynalyst.analysis.crnt_analysis import crnt_bistability
    from dynalyst.analysis.crnt_analysis import draw_reaction_network
    assert "reaction_graph_plot" not in json.loads(
        crnt_bistability("Full_model_for_SN", structure_only=True))     # drawn only when asked
    d = json.loads(draw_reaction_network("Full_model_for_SN"))
    assert d["reaction_graph_plot"].endswith("_reaction_graph.png")
    assert d["complex_graph_plot"].endswith("_complex_graph.png")
    net = {"reactions": [
        {"id": "r1", "rate_constant": "k1", "reactants": {"A": 1, "K": 1}, "products": {"AK": 1}},
        {"id": "r2", "rate_constant": "k2", "reactants": {"AK": 1}, "products": {"AP": 1, "K": 1}},
        {"id": "r3", "rate_constant": "k3", "reactants": {"AP": 1, "P": 1}, "products": {"APC": 1}},
        {"id": "r4", "rate_constant": "k4", "reactants": {"APC": 1}, "products": {"A": 1, "P": 1}}]}
    g = nx.DiGraph()
    for r in net["reactions"]:
        for s in r["reactants"]:
            g.add_edge(s, r["id"])
        for s in r["products"]:
            g.add_edge(r["id"], s)
    assert nx.is_weakly_connected(g)           # 2 linkage classes, one connected network
