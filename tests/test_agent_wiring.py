"""Everything added to the tools, driven through the real agent: orchestrator ->
sub-agent -> guard -> tool -> final check, with scripted models and only AUTO-07p
stubbed (so it runs on any machine). The first argument of _run is the plan the
orchestrator writes (write_plan), kept so each test states its plan."""
import contextlib
import io

import pytest

from dynalyst.orchestration.agent import Agent
from dynalyst.regime import batch_lhs_runner, categorize_lhs
from tests.fakes import call, say, scripted, tools, via_teams


def _plan(*goals):
    return [f"{a}: {g}" for a, g in goals]


def _run(plan, orch, sub, query, history=None):
    orch = scripted(tools(call("write_plan", steps=plan)), *orch.messages)
    agent = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                  verbose=False, save_records=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.run(query, history)


def _delegation(rec, tool):
    step = next(s for s in rec.steps if s.get("kind") == "delegation" and s.get("tool") == tool)
    assert step["status"] == "ok", step
    return step


@pytest.fixture
def no_plots(monkeypatch):
    for mod in (categorize_lhs,):
        monkeypatch.setattr(mod, "mark_special_points", lambda *a, **k: set())
        monkeypatch.setattr(mod, "_plot_branch_with_stability", lambda *a, **k: None)
    monkeypatch.setattr("dynalyst.regime.interactive_plotting.plot_categorized_interactive",
                        lambda *a, **k: None)
    monkeypatch.setattr("dynalyst.regime.interactive_plotting.plot_category_counts_interactive",
                        lambda *a, **k: None)


def test_goal_driven_lhs_with_range_constraints_and_a_split(stubbed_auto, no_plots):
    stubbed_auto["specials"][:] = [[(1, "LP", 1), (2, "LP", 2)], []]
    args = dict(model="Markevich2004.xml", n_samples=4, bif_param="k2cat", goal="bistability",
                separate=True, fixed_params='["Km2"]', baseline_values='{"Km1": 40}',
                param_windows='{"Km3": {"fold": 1.5}}', seed=0)
    rec = _run((_plan(("bifurcation", "Find bistability in Markevich2004.xml"))),
               scripted(tools(call("delegate_to_bifurcation", task="Find bistability")),
                        say("Two of four samples are bistable.")),
               scripted(tools(call("lhs_bifurcation_scan", **args))),
               "Find bistability in Markevich2004.xml with LHS")
    out = _delegation(rec, "lhs_bifurcation_scan")["output"]
    assert "Goal check:" in out and "FOUND bistability" in out and "Baseline control" in out
    assert "Per-category subplots:" in out and "Km2: held at" in out and "(changed baseline)" in out
    assert "BioModels ID BIOMD0000000027" in out
    assert rec.follow_ups is not None


def test_categorized_tool_accepts_separate_and_a_goal(stubbed_auto, no_plots):
    rec = _run((_plan(("bifurcation", "Split an LHS scan of Garde2020 by category"))),
               scripted(tools(call("delegate_to_bifurcation", task="scan")), say("Done.")),
               scripted(tools(call("categorized_lhs_scan", model="Garde2020.xml", n_samples=3,
                                   bif_param="Ge", goal="branch point", separate=True, seed=1))),
               "LHS Garde2020 in Ge, find a branch point, split by category")
    out = _delegation(rec, "categorized_lhs_scan")["output"]
    assert "Categorized LHS scan complete" in out and "Goal check:" in out


def test_get_odes_runs_in_the_model_team_and_its_values_pass_the_final_check():
    # the orchestrator calls no tool itself: a model's ODEs come from model_search
    rec = _run((_plan()),
               scripted(tools(call("delegate_to_model_search",
                                   task="Show the ODEs and parameter values of Markevich2004.")),
                        say("dM/dt = -v_v1 + v_v4 with MAPKK = 50 and k1cat = 0.01.")),
               scripted(tools(call("get_odes", model_name_or_path="Markevich2004")),
                        say("Ran: get_odes.")),
               "Show me the ODEs and parameter values of Markevich2004")
    step = next(s for s in rec.steps if s["tool"] == "get_odes")
    assert step["sub_agent"] == "model_search" and "MAPKK" in step["output"]
    assert rec.final_check["unsupported_values"] == []


def test_crnt_on_an_xpp_file_reports_the_converted_model():
    rec = _run((_plan(("network_theory", "Can Full_model_for_SN.txt be bistable?"))),
               scripted(tools(call("delegate_to_network_theory", task="CRNT")), say("Done."),
                        say("Done.")),           # once more: equations asked for, none given
               scripted(tools(call("crnt_bistability", model_name_or_path="Full_model_for_SN.txt",
                                   structure_only=True))),
               "Run CRNT on Full_model_for_SN.txt and show the converted ODEs")
    out = _delegation(rec, "crnt_bistability")["output"]
    assert '"converted_model"' in out and '"reproduces_original_odes": true' in out


def test_a_time_dependent_model_through_the_steady_state_sub_agent():
    rec = _run((_plan(("steady_state", "Steady states of brannmark"))),
               scripted(tools(call("delegate_to_steady_state", task="steady states")), say("Done.")),
               scripted(tools(call("find_steady_states", model_name_or_path="brannmark"))),
               "Find the steady states of brannmark")
    out = _delegation(rec, "find_steady_states")["output"]
    assert "time_dependent_inputs" in out and "insulin" in out


def test_a_borrowed_biomodels_id_is_removed_from_the_answer():
    history = [{"role": "user", "content": "fetch BIOMD0000000010"},
               {"role": "assistant", "content": "Saved BIOMD0000000010 (Kholodenko2000)."}]
    wrong = "Markevich2004 (BIOMD0000000010) has steady states."
    rec = _run((_plan(("steady_state", "Steady states of Markevich2004"))),
               scripted(tools(call("delegate_to_steady_state", task="steady states")), say(wrong),
                        say(wrong)),                                    # the rewrite keeps it
               scripted(tools(call("find_steady_states", model_name_or_path="Markevich2004"))),
               "Find the steady states of Markevich2004", history)
    assert "BIOMD0000000010" not in rec.final_answer.split("_Note")[0]
    assert rec.final_check["model_ids_removed"] == ["BIOMD0000000010"]


def test_refusals_reach_the_agent_as_reasons_not_crashes():
    rec = _run((_plan(("bifurcation", "LHS of saddlenode.xml"))),
               scripted(tools(call("delegate_to_bifurcation", task="LHS")), say("It cannot be sampled.")),
               scripted(tools(call("lhs_bifurcation_scan", model="saddlenode.xml", n_samples=5)),
                        say("Nothing to sample.")),
               "LHS sample 10 points of saddlenode.xml")
    step = next(s for s in rec.steps if s.get("tool") == "lhs_bifurcation_scan")
    assert "Nothing to sample" in (step.get("output") or "")


