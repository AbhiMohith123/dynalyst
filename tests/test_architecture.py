"""The architecture pieces: one stability-aware labeller for any AUTO result,
shared session state, ranked candidates + human approval before AUTO,
save_sbml, parameter overrides in simulation, the research sub-agent and
paper-to-model mapping. AUTO is stubbed where a scan runs."""
import contextlib
import io
import json

import numpy as np
import pytest

from dynalyst import config
from dynalyst.orchestration.agent import Agent
from dynalyst.orchestration.state import SessionState
from tests.fakes import call, say, scripted, tools, via_teams


# -- classification ------------------------------------------------------------

def _s_curve():
    """An S-shaped response: two folds, two stable branches overlapping in x 1..2."""
    lower = (np.linspace(0, 2, 30), np.linspace(0, 1, 30), np.ones(30))
    middle = (np.linspace(2, 1, 30), np.linspace(1, 2, 30), -np.ones(30))
    upper = (np.linspace(1, 3, 30), np.linspace(2, 3, 30), np.ones(30))
    return [lower, middle, upper]


def test_two_folds_enclosing_two_stable_states_are_bistable():
    from dynalyst.regime.taxonomy import classify_diagram
    c = classify_diagram(["LP", "LP"], _s_curve())
    assert c["category"] == "bistable" and c["evidence"]["max_coexisting_stable"] == 2
    assert 1 <= c["evidence"]["at"] <= 2


def test_two_folds_without_coexisting_stable_states_are_not_bistable():
    from dynalyst.regime.taxonomy import classify_diagram
    lower, middle, upper = _s_curve()
    upper = (upper[0], upper[1], -np.ones(30))          # the upper branch is unstable
    c = classify_diagram(["LP", "LP"], [lower, middle, upper])
    assert c["category"] == "multiple_folds" and "not demonstrated" in c["evidence"]["note"]


def test_a_pitchforks_two_stable_arms_count_as_bistable_and_as_a_branch_point():
    from dynalyst.regime.taxonomy import classify_diagram
    x = np.linspace(1, 2, 20)
    arms = [(x, np.sqrt(x - 1), np.ones(20)), (x, -np.sqrt(x - 1) - 1e-3, np.ones(20)),
            (np.linspace(0, 2, 40), np.zeros(40), np.r_[np.ones(20), -np.ones(20)])]
    c = classify_diagram(["BP"], arms)
    assert c["features"] == ["bistable", "branch_point"]


def test_without_stability_two_folds_do_not_establish_bistability():
    from dynalyst.regime.taxonomy import classify_diagram
    assert classify_diagram(["LP", "LP", "HB"])["features"] == ["hopf", "multiple_folds"]


# -- shared state ------------------------------------------------------------------

def test_state_keeps_the_real_values_and_resets_on_a_new_model(tmp_path):
    st = SessionState(path=tmp_path / "s.json")
    st.update_from_tool("lhs_bifurcation_scan", {"model": "Markevich2004.xml", "n_samples": 5},
                        "LHS done (bif_param=k2cat, y_species=Mp).\nBounds (lo, hi) per parameter;"
                        " 1 varied\nSummary CSV: out/s.csv\nCategories (primary, per sample): "
                        "{'bistable': 5}")
    assert st["bif_param"] == "k2cat" and st["last_scan"]["csv"] == "out/s.csv"
    assert "k2cat" in st.as_text() and "out/s.csv" in st.as_text()
    again = SessionState(path=tmp_path / "s.json")               # persisted
    assert again["model"] == "Markevich2004.xml"
    again.update_from_tool("simulate_model", {"model_name_or_path": "Elowitz_Nature2000.xml"}, "{}")
    assert again["model"] == "Elowitz_Nature2000.xml" and again["bif_param"] is None
    again.update_from_tool("simulate_model", {"model_name_or_path": "x"}, '{"error": "nope"}')
    assert again["model"] == "Elowitz_Nature2000.xml"            # a failed call changes nothing


# -- candidates + approval ----------------------------------------------------------

CANDS = [{"parameter": "k4cat", "value": 0.06, "range": [0.02, 0.09], "y_species": "Mp",
          "expected": "LP (fold)", "reason": "fold near 0.047"},
         {"parameter": "k2cat", "value": 15.0, "range": [7.5, 30.0], "y_species": "Mp",
          "expected": "LP (fold)", "reason": "fold near 20"}]


@pytest.mark.parametrize("reply,expected", [
    ("yes", ("approve", 0)), ("ok go ahead", ("approve", 0)), ("2", ("approve", 1)),
    ("use k2cat please", ("approve", 1)), ("no", ("reject", None)),
    ("just the report", ("reject", None)), ("now simulate Elowitz_Nature2000.xml", ("other", None))])
def test_the_reply_to_an_approval_is_read_by_meaning(reply, expected):
    from dynalyst.orchestration.approval import parse
    assert parse(reply, CANDS) == expected


def test_only_runs_without_a_named_parameter_pause():
    from dynalyst.orchestration import approval
    assert approval.needed("lhs_bifurcation_scan", {"model": "Markevich2004.xml"}, "ask")
    assert not approval.needed("lhs_bifurcation_scan", {"model": "Markevich2004.xml",
                                                        "bif_param": "k2cat"}, "ask")
    assert approval.needed("run_bifurcation_analysis",
                           {"model_name_or_path": "pdp_mm_bistable.txt"}, "ask")
    assert not approval.needed("run_bifurcation_analysis",
                               {"model_name_or_path": "pdp_mm_bistable.txt", "param_name": "ATOT"}, "ask")
    # called from code: nobody to ask
    assert not approval.needed("lhs_bifurcation_scan", {"model": "Markevich2004.xml"}, "auto")


def test_only_the_chat_entry_points_ask():
    from dynalyst.orchestration.agent import Agent
    assert Agent(llms={}, save_records=False, verbose=False).approval == "auto"
    assert Agent(llms={}, save_records=False, verbose=False, approval="ask").approval == "ask"
    with pytest.raises(ValueError):
        Agent(llms={}, save_records=False, approval="sometimes")
    assert config.PARALLEL is True
    assert not hasattr(config, "ARCHITECTURE")          # hierarchical only: no flat form


def test_the_run_pauses_before_auto_and_the_reply_runs_the_chosen_candidate(stubbed_auto, monkeypatch):
    monkeypatch.setattr("dynalyst.analysis.candidates.propose",
                        lambda m, n=5: {"model": m, "candidates": CANDS, "method": "test"})
    request = "Draw 3 parameter sets of Markevich2004.xml and look for bistability"
    st = SessionState(persist=False)
    scan = call("lhs_bifurcation_scan", model="Markevich2004.xml", n_samples=3)

    def agent(sub_msgs, orch_msgs):
        return Agent(llms=via_teams({"orchestrator": scripted(*orch_msgs),
                                     "subagent": scripted(*sub_msgs)}),
                     verbose=False, save_records=False, state=st, approval="ask")

    with contextlib.redirect_stdout(io.StringIO()):
        rec = agent([tools(scan)], [tools(call("delegate_to_bifurcation", task=request))]).run(request)
    assert rec.awaiting_approval and "k4cat" in rec.final_answer and "k2cat" in rec.final_answer
    assert st["pending_approval"]["tool"] == "lhs_bifurcation_scan"
    assert not any(s.get("output") for s in rec.steps)                    # no AUTO run

    with contextlib.redirect_stdout(io.StringIO()):
        rec2 = agent([tools(call("lhs_bifurcation_scan", model="Markevich2004.xml", n_samples=3)),
                      say("done")],
                     [tools(call("delegate_to_bifurcation", task=request)), say("Done.")]).run("2")
    step = next(s for s in rec2.steps if s.get("tool") == "lhs_bifurcation_scan")
    assert step["arguments"]["bif_param"] == "k2cat"                 # the approved candidate
    assert step["arguments"]["param_min"] == 7.5 and step["arguments"]["y_species"] == "Mp"
    assert rec2.approval["choice"] == 2 and "Approved by the user" in rec2.query
    assert st["pending_approval"] is None and st["bif_param"] == "k2cat"
    # after the approved run the supervisor writes the answer (not the budget fallback)
    assert "budget ran out" not in rec2.final_answer and rec2.final_answer.startswith("Done.")
    assert rec2.model_calls["orchestrator"] >= 1


def test_no_keeps_the_report_only(monkeypatch):
    st = SessionState(persist=False)
    st.set(pending_approval={"query": "q", "tool": "lhs_bifurcation_scan", "model": "M.xml",
                             "candidates": CANDS, "created": __import__("time").time()})
    agent = Agent(llms={}, verbose=False, save_records=False, state=st)
    rec = agent.run("no")
    assert "No AUTO-07p run" in rec.final_answer and "k2cat" in rec.final_answer
    assert st["pending_approval"] is None and sum(rec.model_calls.values()) == 0


# -- save_sbml / simulation overrides ---------------------------------------------------

def test_save_sbml_writes_a_derived_copy_and_rejects_unknown_names(tmp_path, monkeypatch):
    from dynalyst.analysis.annotations import model_identity
    from dynalyst.analysis.save_model import save_sbml
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    out = json.loads(save_sbml("Markevich2004", "mk_fast", parameters={"k2cat": 5}))
    assert out["changed"]["k2cat"]["value"] == 5.0
    import libsbml
    m = libsbml.readSBMLFromFile(out["path"]).getModel()
    assert m.getParameter("k2cat").getValue() == 5.0
    assert model_identity(out["path"])["biomodels_id"] is None      # not the published model
    bad = json.loads(save_sbml("Markevich2004", "mk_bad", parameters={"nope": 1}))
    assert "Nothing was saved" in bad["error"] and not (tmp_path / "mk_bad.xml").exists()


def test_save_sbml_takes_an_lhs_sample_from_the_scan_csv(tmp_path, monkeypatch):
    from dynalyst.analysis.save_model import save_sbml
    monkeypatch.setattr(config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    csv = tmp_path / "summary.csv"
    csv.write_text("sample_id,category,features,signature,error,k2cat,Km1\n"
                   "control,bistable,bistable,LP,,15,50\n3,bistable,bistable,LP,,22.5,61\n")
    out = json.loads(save_sbml("Markevich2004", "mk_s3", scan_csv=str(csv), sample_id="3"))
    assert out["changed"]["k2cat"]["value"] == 22.5 and out["changed"]["Km1"]["value"] == 61.0


def test_every_model_is_simulated_from_its_own_file():
    # no hand-coded model equations: a file jaxkineticmodel cannot load is read by
    # generic_sbml and integrated, with parameter overrides and other start values
    from dynalyst.analysis.jaxkm_tools import compare_initial_conditions, simulate_model
    res = json.loads(simulate_model("Markevich2004", t_end=10, n_points=20,
                                    parameters={"k1cat": 0.02}))
    assert "error" not in res, res
    assert res["parameters_changed"]["k1cat"] == {"model_value": 0.01, "used": 0.02}
    res = json.loads(compare_initial_conditions("p_dynamics", [{"p": 0.1}, {"p": 2.0}], t_end=5))
    assert res["n_conditions"] == 2


# -- research sub-agent / paper mapping / CRNT ---------------------------------------------

def test_steady_states_are_shared_with_bifurcation_and_simulation():
    from dynalyst.orchestration.subagents import SUBAGENTS, owner_of
    assert owner_of("find_steady_states") == "steady_state"
    for name in ("bifurcation", "simulation"):
        assert "find_steady_states" in SUBAGENTS[name].objective_tools
    assert "find_steady_states" not in SUBAGENTS["stochastic"].objective_tools


def test_research_and_model_agents_have_their_tools():
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.subagents import SUBAGENTS
    research = set(SUBAGENTS["research"].objective_tools)
    assert {"search_literature", "read_pdf", "retrieve_knowledge", "index_pdf",
            "map_paper_to_model"} <= research <= set(registry.all_tools())
    assert "save_sbml" in SUBAGENTS["model_search"].objective_tools
    assert "propose_bifurcation_candidates" in SUBAGENTS["bifurcation"].objective_tools


def test_paper_values_are_matched_through_typeset_names():
    from dynalyst.analysis.paper_map import _find_values
    pages = [(4, "values are: Km1 \x03 50, Km2 \x03 500 (nM); k1\ncat \x03 0.01, k2\ncat 15 (s-1)")]
    assert _find_values("k1cat", pages)[0]["value"] == 0.01
    assert _find_values("k2cat", pages)[0]["value"] == 15
    assert _find_values("Km2", pages)[0]["value"] == 500
    assert _find_values("Km9", pages) == []


def test_map_paper_to_model_on_the_markevich_paper():
    from dynalyst.analysis.paper_map import map_paper_to_model
    if not (config.PAPERS_DIR / "Markevich.pdf").exists():
        pytest.skip("Markevich.pdf not in data/papers")
    r = json.loads(map_paper_to_model("Markevich.pdf", "Markevich2004", resolve_proteins=False))
    assert r["summary"]["same"] >= 6 and r["summary"]["different"] == 0


def test_semi_diffusive_result_is_evidence_not_a_confirmed_set():
    from dynalyst.analysis import crnt_analysis as ca

    class Approach:
        get_optimization_bounds = staticmethod(lambda: [(0, 1)])
        run_optimization = staticmethod(lambda **kw: ([np.array([0.5])], [0.0]))
        get_decision_vector = staticmethod(lambda: ["v1"])

    class Net:
        get_semi_diffusive_approach = staticmethod(lambda: Approach())

    res = ca._semi_diffusive(Net(), 5, 0)
    assert res["multistationarity_possible"] and res["bistable"] is None


def test_a_downloaded_model_takes_the_focus_and_old_names_are_scoped(tmp_path):
    st = SessionState(path=tmp_path / "s.json")
    st.update_from_tool("run_bifurcation_analysis",
                        {"model_name_or_path": "BIOMD0000000016", "param_name": "rM_Vs"}, "{}")
    st.update_from_tool("fetch_biomodels_model", {"model_id": "BIOMD0000000015"},
                        json.dumps({"use_as": "BIOMD0000000015", "saved_as": "x.xml"}))
    assert st["model"] == "BIOMD0000000015" and st["bif_param"] is None
    st.set(bif_param="p1")
    assert "for BIOMD0000000015 ONLY" in st.as_text()


def test_jaxkm_and_agent_names_of_a_local_constant_are_the_same_parameter():
    from dynalyst.analysis.jaxkm_tools import agent_name, override_key
    params = {"lp_gdna__fdnap9": 0.42, "lp_adna__fdnap9": 0.42, "k1": 1.0}
    assert override_key(params, "gdna_fdnap9") == "lp_gdna__fdnap9"
    assert override_key(params, "k1") == "k1"
    assert override_key(params, "fdnap9") is None                 # ambiguous bare name
    assert agent_name("lp_gdna__fdnap9") == "gdna_fdnap9"


def test_a_downloaded_model_takes_the_focus_and_old_names_are_scoped(tmp_path):
    st = SessionState(path=tmp_path / "s.json")
    st.update_from_tool("run_bifurcation_analysis",
                        {"model_name_or_path": "BIOMD0000000016", "param_name": "rM_Vs"}, "{}")
    st.update_from_tool("fetch_biomodels_model", {"model_id": "BIOMD0000000015"},
                        json.dumps({"use_as": "BIOMD0000000015", "saved_as": "x.xml"}))
    assert st["model"] == "BIOMD0000000015" and st["bif_param"] is None
    st.set(bif_param="p1")
    assert "for BIOMD0000000015 ONLY" in st.as_text()


def test_jaxkm_and_agent_names_of_a_local_constant_are_the_same_parameter():
    from dynalyst.analysis.jaxkm_tools import agent_name, override_key
    params = {"lp_gdna__fdnap9": 0.42, "lp_adna__fdnap9": 0.42, "k1": 1.0}
    assert override_key(params, "gdna_fdnap9") == "lp_gdna__fdnap9"
    assert override_key(params, "k1") == "k1"
    assert override_key(params, "fdnap9") is None                 # ambiguous bare name
    assert agent_name("lp_gdna__fdnap9") == "gdna_fdnap9"



def test_an_lhs_scan_with_no_range_runs_over_the_range_the_session_diagrams_swept():
    st = SessionState(persist=False)
    for rng in ([-0.01, 500], [150, 230], [0, 1000]):        # three diagrams, one model
        st.update_from_tool("run_bifurcation_analysis",
                            {"model_name_or_path": "data/sbml_models/Full_model_for_SN_plus_TB.xml",
                             "param_name": "ATOT"},
                            json.dumps({"swept_parameter": "ATOT", "swept_range": rng,
                                        "bifurcation_points": []}))
    agent = Agent(llms={}, verbose=False, save_records=False, state=st)
    scan = {"model": "Full_model_for_SN_plus_TB.xml", "bif_param": "ATOT", "n_samples": 30}
    assert agent._diagram_range(scan) == {**scan, "param_min": -0.01, "param_max": 1000.0}
    given = {**scan, "param_min": 1, "param_max": 5}                # the user's own range stays
    assert agent._diagram_range(given) == given
    other = {**scan, "bif_param": "k1"}                             # no diagram in k1: tool default
    assert agent._diagram_range(other) == other
