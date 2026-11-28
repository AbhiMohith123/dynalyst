"""Restraint in a chat (prompt ladder Levels 5a and 5c): the user's range is kept, and a
wider one is proposed after the answer when it holds no special point; a word standing for
several parameters is asked about. No language model and no AUTO run here."""
import json

from dynalyst.orchestration.agent import Agent, RunRecord
from dynalyst.orchestration.approval import (ambiguous_phrase, user_ranges, wider_range,
                                             widens_user_range)
from dynalyst.orchestration.state import SessionState

ASK = "Is there a fold in MAPKK over [1, 10] in Markevich2004?"


def _agent(text):
    a = Agent(state=SessionState(persist=False), llms={}, save_records=False, verbose=False,
              approval="ask")
    a._user_text = text
    return a


def test_the_users_range_is_read_next_to_the_parameter_only():
    assert user_ranges(ASK, "MAPKK") == [(1.0, 10.0)]
    assert user_ranges("diagram along stress from 0 to 6 for AMPK", "stress") == [(0.0, 6.0)]
    assert user_ranges("simulate from 0 to 100, then a diagram in stress", "stress") == []
    call = {"param_name": "MAPKK", "param_min": 0, "param_max": 150}
    assert widens_user_range("run_bifurcation_analysis", call, ASK, "ask") == (1.0, 10.0)
    assert widens_user_range("run_bifurcation_analysis", {**call, "param_min": 1, "param_max": 10},
                             ASK, "ask") is None
    assert widens_user_range("run_bifurcation_analysis", call, ASK, "auto") is None   # scripts
    assert wider_range(1, 10, 50) == (1.0, 500.0)        # a decade past the model's own value


def test_a_range_with_no_special_point_ends_with_the_offer_of_a_wider_one():
    a = _agent(ASK)
    rec = RunRecord(query=ASK, model={})
    args = {"model_name_or_path": "Markevich2004.xml", "param_name": "MAPKK",
            "param_min": 1, "param_max": 10, "fixed_params": {"MAPKK": 50}}
    out = json.dumps({"swept_parameter": "MAPKK", "swept_range": [1, 10], "bifurcation_points": []})
    a._maybe_ask("bifurcation", "fold in MAPKK", "run_bifurcation_analysis", args, out,
                 {"status": "ok"}, rec)
    message, pending = a._cycle_offer
    assert "no bifurcation point in your range MAPKK" in message and "[1, 500]" in message
    assert pending["kind"] == "widen"
    rerun = pending["calls"]["MAPKK"][1]["arguments"]
    assert (rerun["param_min"], rerun["param_max"]) == (1.0, 500.0)
    # "no": the answer stands, nothing runs
    a.state.set(pending_approval=pending)
    query, report = a._resolve_pending("no", RunRecord(query="no", model={}))
    assert report.startswith("No wider range")
    # a range with a point in it: no offer
    b = _agent(ASK)
    found = json.dumps({"swept_parameter": "MAPKK", "swept_range": [1, 10],
                        "bifurcation_points": [{"type": "LP", "MAPKK": 5.0}]})
    b._maybe_ask("bifurcation", "t", "run_bifurcation_analysis", args, found, {"status": "ok"}, rec)
    assert b._cycle_offer is None


def test_a_word_for_several_parameters_makes_them_the_candidates():
    names = ["Km1", "Km2", "k1cat", "k2cat", "k3cat", "k4cat"]
    assert ambiguous_phrase("does kcat control the switch?", names) == (
        "kcat", ["k1cat", "k2cat", "k3cat", "k4cat"])
    assert ambiguous_phrase("does k2cat control the switch?", names) is None   # named exactly
    a = _agent("Does kcat control bistability in Markevich2004?")
    screened = {"candidates": [{"parameter": "Km1", "range": [1, 100], "expected": "LP",
                                "reason": "fold", "y_species": "Mpp"},
                               {"parameter": "k3cat", "range": [0.1, 10], "expected": "LP",
                                "reason": "fold", "y_species": "Mpp"}]}
    res = a._phrase_candidates("Markevich2004.xml", screened)
    assert [c["parameter"] for c in res["candidates"]] == ["k3cat", "k1cat", "k2cat", "k4cat"]
    assert "'kcat' stands for several parameters" in res["note"]
    assert all(len(c["range"]) == 2 for c in res["candidates"])


def test_a_range_left_out_or_an_lhs_range_is_not_the_users():
    call = {"param_name": "MAPKK"}                       # no range: the tool's own window
    assert widens_user_range("run_bifurcation_analysis", call, ASK, "ask") == (1.0, 10.0)
    lhs = {"bif_param": "MAPKK", "param_min": 0.1, "param_max": 200}
    assert widens_user_range("lhs_bifurcation_scan", lhs, ASK, "ask") == (1.0, 10.0)
    assert widens_user_range("categorized_lhs_scan", {**lhs, "param_min": 1, "param_max": 10},
                             ASK, "ask") is None


def test_a_parameter_named_like_a_word_is_named_only_as_a_parameter():
    from dynalyst.orchestration.approval import name_ends
    assert not name_ends("Draw a bifurcation diagram of the brusselator", "a")
    assert name_ends("the brusselator along a from 1 to 3", "a")
    assert name_ends("brusselator with a = 2", "a") and name_ends("vary I from 0 to 2", "I")
    assert not name_ends("I want the diagram of fhn_check", "I")
    assert user_ranges("the brusselator along a from 1 to 3", "a") == [(1.0, 3.0)]
    assert name_ends("Draw Markevich2004 along atot", "ATOT")       # a real name: any case


def test_phrase_candidates_keep_the_sign_of_a_negative_value():
    a = _agent("Does kcat move the fold?")
    a_model = {"param_names": ["k1cat", "k2cat"], "param_defaults": {"k1cat": -2.0, "k2cat": 0.0}}
    import dynalyst.orchestration.inputs as inputs
    old = inputs._model_for
    inputs._model_for = lambda m: a_model
    try:
        res = a._phrase_candidates("x.xml", {"candidates": []})
    finally:
        inputs._model_for = old
    ranges = {c["parameter"]: c["range"] for c in res["candidates"]}
    assert ranges["k1cat"] == [-20.0, -0.2] and ranges["k2cat"] == [0.0, 1.0]
