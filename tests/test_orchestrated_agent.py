"""The agent's architecture end to end, with scripted fake models and the real tools /
AUTO-07p: orchestrator (plan) -> team -> sub-agents -> guard -> tool -> reviewer -> final-answer
check."""
import io
import contextlib

import pytest

from dynalyst.orchestration.agent import Agent
from dynalyst.orchestration.final_check import unsupported_values
from tests.fakes import call, say, scripted, tools, via_teams


def _run(agent, query):
    with contextlib.redirect_stdout(io.StringIO()):
        return agent.run(query)


@pytest.mark.auto
def test_full_loop_guard_review_and_final_check(tmp_path):
    from tests import xpp_sets
    model = xpp_sets.write_xpp(tmp_path / "markevich_set.ode", xpp_sets.MARKEVICH)
    sub = scripted(
        # bifurcation delegation: look at the models, pick a tool it does not own
        # (guard rejects), then the right one, then report
        tools(call("list_available_models")),
        tools(call("classify_model", model_name_or_path="pitchfork_model.xml")),
        tools(call("run_bifurcation_analysis", model_name_or_path=model, param_name="ATOT", plot_state="APP",
                   param_min=0.1, param_max=200)),
        say("Ran: run_bifurcation_analysis. Result: folds as returned. Problems: none"),
        # classifier delegation
        tools(call("classify_model", model_name_or_path="pitchfork_model.xml")),
        say("Ran: classify_model. Result: 2D. Problems: none"),
    )
    orch = scripted(
        tools(call("write_plan", steps=["dynamics_team: run the Markevich set once",
                                        "model_team: classify pitchfork_model.xml"])),
        tools(call("delegate_to_bifurcation", task="Run the Markevich set once")),
        tools(call("delegate_to_model_classifier", task="Classify pitchfork_model.xml")),
        say("The Markevich set has folds at ATOT = 85.96 and 76.58 and a third at 123.4. "
            "The model is 2D."),
        say("The Markevich set has folds at ATOT = 85.96 and 76.58. The model is 2D."),
    )
    rec = _run(Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                     save_records=False), "Run the Markevich set and classify the model")

    assert len(rec.plan_versions) == 1 and len(rec.plan) == 2
    rejected, bif, cls = [s for s in rec.steps if s["kind"] == "delegation"]
    assert rejected["status"] == "rejected" and "does not own" in rejected["error"]
    assert bif["tool"] == "run_bifurcation_analysis" and bif["status"] == "ok"
    verified = [c for c in bif["reviewer"]["checks"] if c["check"] == "verify_LP"]
    # the reviewer re-derived them: 85.96 and 76.58 on the start branch, and 102.03 on a branch
    # the search off the start branch finds (three steady states at ATOT = 110 and 200, one at 95)
    assert len(verified) == 3 and all(c["passed"] for c in verified)
    assert any("102.03" in c["detail"] for c in verified)
    assert cls["status"] == "ok" and '"independent_dimension": 2' in cls["output"]
    # the invented 123.4 triggered exactly one rewrite
    assert rec.final_check["unsupported_values"] == ["123.4"]
    assert rec.final_check["rewritten"] and "123.4" not in rec.final_answer
    assert rec.model_calls == {"orchestrator": 5, "team": 4, "subagent": 6}


def test_guard_rejection_then_duplicate_reuse():
    sub = scripted(
        tools(call("find_steady_states", model_name_or_path="Markevich2004",        # rejected
                   no_such_argument=1.0)),
        tools(call("find_steady_states", model_name_or_path="Markevich2004")),     # corrected
        say("Ran: find_steady_states. Problems: none"),
        tools(call("find_steady_states", model_name_or_path="Markevich2004")),     # duplicate
        say("Same steady states as before."),
    )
    orch = scripted(
        tools(call("delegate_to_steady_state", task="Steady states of Markevich2004")),
        tools(call("delegate_to_steady_state", task="Again, steady states of Markevich2004")),
        say("Markevich2004 has steady states."),
    )
    rec = _run(Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                     save_records=False), "Find the steady states of Markevich2004")
    rejected, ran, reused = [s for s in rec.steps if s["kind"] == "delegation"]
    assert rejected["status"] == "rejected" and "unknown" in rejected["error"]
    assert ran["status"] == "ok"
    assert reused["duplicate"] is True and reused["status"] == "ok"
    # the sub-agent was told why its call was not run
    assert rejected["error"] in sub.seen[1][-1].content


def test_final_check_rules():
    ev = ['{"ATOT": 85.9622, "loss": 9.38e-13, "k4": 0.4}']
    assert unsupported_values("fold at ATOT=86.0 and 85.96, loss 9.4e-13", ev) == []
    assert unsupported_values("k4 = 0.40 fitted; 3 samples; model Markevich2004", ev) == []
    assert unsupported_values("fold at 91.5", ev) == ["91.5"]


def test_reduction_delegation_and_follow_ups():
    from dynalyst import config
    from tests.test_reduction_and_followups import MM
    sub = scripted(tools(call("reduce_model", **MM, model_name="test_mm_reduced")))
    orch = scripted(
        tools(call("delegate_to_model_reduction", task="Reduce the typed MM system with QSSA")),
        say("Reduced to one equation for S, but the full-state initial transient does not match; keep the full model."),
    )
    try:
        # the equations are typed in the request: inline odes are only for those (guard)
        rec = _run(Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                         save_records=False),
                   "Simplify this enzyme model with QSSA: dS/dt = -k1*E*S + km1*C, "
                   "dE/dt = -k1*E*S + (km1+k2)*C, dC/dt = k1*E*S - (km1+k2)*C, dP/dt = k2*C "
                   "with k1 = 10, km1 = 1, k2 = 1, S(0) = 10, E(0) = 0.1, C(0) = 0, P(0) = 0")
    finally:
        (config.LOCAL_SBML_MODELS_DIR / "test_mm_reduced.xml").unlink(missing_ok=True)
    step = [s for s in rec.steps if s["kind"] == "delegation"][0]
    assert step["tool"] == "reduce_model" and step["status"] == "ok"
    assert rec.follow_ups and "not accurate enough" in rec.follow_ups[0]["question"]
    assert all("Run a bifurcation analysis of test_mm_reduced" not in f["query"] for f in rec.follow_ups)
    assert "**What next?**" in rec.final_answer
