"""The redesigned agent: one orchestrator that plans for itself, scoped context, failures
reported by the tools, and the fixed list of code exceptions (file safety, permission,
retry limit, budgets, duplicates, approval, final check). Scripted models: no API key."""
import json
import os

import numpy as np
import pytest
from types import SimpleNamespace

from dynalyst import config
from dynalyst.orchestration import safety, tools as registry
from dynalyst.orchestration.agent import Agent
from dynalyst.orchestration.guard import ExecutionGuard
from dynalyst.orchestration.state import SessionState
from tests.fakes import call, say, scripted, tools, via_teams

MODEL = "saddle_node.xml"         # one species P, one parameter alpha


def agent(orch, team=None, sub=None, **kw):
    llms = {"orchestrator": orch, "subagent": sub or scripted(say("unused"))}
    if team is not None:
        llms["team"] = team
    return Agent(llms=llms, verbose=False, save_records=False,
                 state=SessionState(persist=False), **kw)


def human_texts(chat):
    return [str(m.content) for prompt in chat.seen for m in prompt]


def test_the_orchestrator_plans_delegates_and_answers():
    orch = scripted(
        tools(call("write_plan", steps=["dynamics_team: steady states of saddle_node.xml"])),
        tools(call("delegate_to_dynamics_team", task="Steady states of saddle_node.xml.")),
        say("Two steady states [step 1]."))
    team = scripted(
        tools(call("delegate_to_steady_state", task="Find the steady states of saddle_node.xml.")),
        say("steady_state ran find_steady_states."))
    sub = scripted(tools(call("find_steady_states", model_name_or_path=MODEL)),
                   say("Ran: find_steady_states. Result: as returned. Problems: none"))
    rec = agent(orch, team, sub).run("What are the steady states of saddle_node.xml?")
    assert rec.plan_versions[0]["steps"] == ["dynamics_team: steady states of saddle_node.xml"]
    assert [s["tool"] for s in rec.steps] == ["find_steady_states"]
    assert rec.steps[0]["status"] == "ok"
    assert rec.model_calls == {"orchestrator": 3, "team": 2, "subagent": 2}


def test_a_sub_agent_sees_only_its_task_not_the_request_or_the_session():
    state = SessionState(persist=False)
    state.set(model="BIOMD0000000010")             # an earlier question's model
    orch = scripted(tools(call("delegate_to_dynamics_team", task="Steady states of saddle_node.xml.")),
                    say("done"))
    team = scripted(tools(call("delegate_to_steady_state", task="Steady states of saddle_node.xml.")),
                    say("done"))
    sub = scripted(say("Not my task: test."))
    a = Agent(llms={"orchestrator": orch, "team": team, "subagent": sub}, verbose=False,
              save_records=False, state=state)
    a.run("SECRET-WORDING please find the steady states of saddle_node.xml")
    seen_by_sub, seen_by_team = " ".join(human_texts(sub)), " ".join(human_texts(team))
    assert "SECRET-WORDING" not in seen_by_sub and "SECRET-WORDING" not in seen_by_team
    assert "BIOMD0000000010" not in seen_by_sub and "BIOMD0000000010" not in seen_by_team
    assert "BIOMD0000000010" in " ".join(human_texts(orch))   # the orchestrator has the facts


def test_a_wrong_parameter_fails_in_the_tool_and_the_sub_agent_reads_the_error():
    orch = scripted(tools(call("delegate_to_bifurcation", task="Diagram of saddle_node.xml in k.")),
                    say("k is not a parameter of saddle_node.xml; it has alpha."))
    sub = scripted(tools(call("run_bifurcation_analysis", model_name_or_path=MODEL, param_name="k")),
                   say("Ran: run_bifurcation_analysis. Problems: k is not a parameter."))
    llms = via_teams({"orchestrator": orch, "subagent": sub})
    rec = agent(orch, llms["team"], sub).run("Diagram of saddle_node.xml in k")
    step = rec.steps[0]
    assert step["guard"]["allowed"] and step["status"] == "error"
    assert "'k' is not a parameter" in step["error"] and "alpha" in step["error"]
    tool_reply = next(str(m.content) for m in sub.seen[-1] if "ERROR from the tool" in str(m.content))
    assert "alpha" in tool_reply


def test_the_guard_never_sees_the_request():
    import inspect
    from dynalyst.orchestration import guard
    assert "query" not in inspect.signature(guard.ExecutionGuard.check).parameters
    assert not hasattr(ExecutionGuard(), "set_request")


def test_permission_and_the_retry_limit():
    g = ExecutionGuard()
    assert not g.check("simulation", "run_bifurcation_analysis", {"model_name_or_path": MODEL}).allowed
    assert "unknown" in g.check("simulation", "simulate_model", {"model_name_or_path": MODEL,
                                                                 "nonsense": 1}).reason
    for i in range(config.MAX_TOOL_RETRIES + 1):
        args = {"model_name_or_path": MODEL, "t_end": i + 1}
        assert g.check("simulation", "simulate_model", args).allowed
        g.record("simulate_model", args, {}, error="boom")
    stopped = g.check("simulation", "simulate_model", {"model_name_or_path": MODEL, "t_end": 99})
    assert not stopped.allowed and "Do not call it again" in stopped.reason


def test_a_successful_call_is_reused_not_rerun():
    g = ExecutionGuard()
    args = {"model_name_or_path": MODEL}
    g.record("simulate_model", args, {"status": "ok", "output": "{}"})
    assert g.check("simulation", "simulate_model", args).duplicate_of is not None


@pytest.mark.parametrize("tool,args,why", [
    ("save_sbml", {"model_name_or_path": MODEL, "name": "copy", "overwrite": True}, "never replaced"),
    ("save_sbml", {"model_name_or_path": MODEL, "name": "../../outside"}, "plain file name"),
    ("save_sbml", {"model_name_or_path": MODEL, "name": "C:/Windows/x"}, "plain file name"),
    ("odes_to_sbml", {"species": ["x"], "odes": {"x": "-x"}, "model_name": "a/b"}, "plain file name"),
    ("crnt_bistability", {"job_id": "../../x"}, "job id"),
])
def test_file_safety_refuses_before_the_tool_runs(tool, args, why):
    out = json.loads(registry.execute(tool, args))
    assert why in out["error"]


def test_an_existing_model_file_is_never_replaced():
    existing = next(config.LOCAL_SBML_MODELS_DIR.glob("*.xml"))
    problem = safety.problem("save_sbml", {"model_name_or_path": MODEL, "name": existing.stem})
    assert "already exists" in problem
    assert "overwrite" not in registry.schema_of(registry.all_tools()["save_sbml"])["properties"]


def test_a_crnt_job_id_cannot_name_another_folder():
    from dynalyst.analysis.crnt_analysis import _job_dir
    with pytest.raises(ValueError):
        _job_dir("../../important")
    assert _job_dir("0123456789ab").name == "0123456789ab"


def _overlap_counter():
    import threading
    import time as _time
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def slow(self, name, task, rec, guard, *rest):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        _time.sleep(0.3)
        with lock:
            state["active"] -= 1
        return f"{name} done"
    return state, slow


def test_teams_delegated_in_one_turn_run_in_parallel(monkeypatch):
    state, slow = _overlap_counter()
    monkeypatch.setattr(config, "PARALLEL", True)
    monkeypatch.setattr(Agent, "_team", slow)
    orch = scripted(tools(call("delegate_to_dynamics_team", task="a"),
                          call("delegate_to_model_team", task="b")), say("both done"))
    rec = agent(orch).run("q")
    assert state["peak"] == 2 and rec.final_answer.startswith("both done")


def test_sub_agents_delegated_in_one_turn_run_in_parallel(monkeypatch):
    state, slow = _overlap_counter()
    monkeypatch.setattr(config, "PARALLEL", True)
    monkeypatch.setattr(Agent, "_delegate", slow)
    orch = scripted(tools(call("delegate_to_dynamics_team", task="a and b")), say("both done"))
    team = scripted(tools(call("delegate_to_simulation", task="a"),
                          call("delegate_to_stochastic", task="b")), say("simulation and stochastic done"))
    rec = agent(orch, team).run("q")
    assert state["peak"] == 2 and rec.team_runs[0]["summary"].startswith("simulation and stochastic")


def test_the_plan_version_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_PLAN_VERSIONS", 2)
    orch = scripted(tools(call("write_plan", steps=["a"])), tools(call("write_plan", steps=["b"])),
                    tools(call("write_plan", steps=["c"])), say("answer"))
    rec = agent(orch).run("q")
    assert [v["steps"] for v in rec.plan_versions] == [["a"], ["b"]]
    assert rec.interventions == [{"kind": "plan version limit"}]


def test_an_unsupported_number_is_flagged_by_the_final_check():
    orch = scripted(say("The fold is at alpha = 7.77."), say("{}"))
    rec = agent(orch).run("Where is the fold of saddle_node.xml?")
    assert "7.77" in rec.final_check["unsupported_values"]


def test_clear_chat_clears_the_session_facts(tmp_path):
    s = SessionState(path=tmp_path / "s.json")
    s.set(model="earlier.xml")
    s.clear()
    assert SessionState(path=tmp_path / "s.json")["model"] is None


# -- questions asked mid-run (chat only): limit cycles after a Hopf point, a long CRNT search --

def _accepting_reviewer(monkeypatch):
    from dynalyst.orchestration import reviewer as rv
    monkeypatch.setattr(rv.Reviewer, "review", lambda self, *a, **k: rv.Verdict("accept"))


def _fake_tools(monkeypatch, outputs):
    """registry.execute with the named tools faked: outputs[name](args) -> result dict."""
    real, calls = registry.execute, []

    def fake(name, args, capture_output=True):
        if name not in outputs:
            return real(name, args, capture_output)
        calls.append((name, dict(args)))
        return json.dumps(outputs[name](args))
    monkeypatch.setattr(registry, "execute", fake)
    return calls


def _hopf(args):
    out = {"swept_parameter": "alpha", "swept_range": [0, 5], "plot_state": "P",
           "bifurcation_points": [{"type": "HB", "alpha": 2.5}]}
    if args.get("_cycles") is False:
        out["cycle_continuation"] = "not run yet"
    else:
        out["cycles"] = {"period": 4.2}
    return out


def test_a_hopf_point_in_a_chat_gets_its_limit_cycles_in_the_same_run(monkeypatch):
    """A person chatting gets the limit cycles from a Hopf point in the same diagram run:
    no "continue the limit cycles?" question, no second run."""
    _accepting_reviewer(monkeypatch)
    calls = _fake_tools(monkeypatch, {"run_bifurcation_analysis": _hopf})
    state = SessionState(persist=False)
    orch = scripted(tools(call("delegate_to_bifurcation", task="Diagram of saddle_node.xml in alpha.")),
                    say("Answer with the cycles."))
    sub = scripted(tools(call("run_bifurcation_analysis", model_name_or_path=MODEL, param_name="alpha")))
    a = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False,
              save_records=False, state=state, approval="ask")
    rec = a.run("Bifurcation diagram of saddle_node.xml in alpha")
    assert not rec.awaiting_approval and state["pending_approval"] is None
    assert rec.final_answer.startswith("Answer with the cycles.") and "Reply **yes** or **no**" not in rec.final_answer
    assert len(calls) == 1 and "_cycles" not in calls[0][1]  # one run, cycles not switched off


def test_a_task_that_wants_the_limit_cycles_runs_them_without_a_question(monkeypatch):
    _accepting_reviewer(monkeypatch)
    calls = _fake_tools(monkeypatch, {"run_bifurcation_analysis": _hopf})
    orch = scripted(tools(call("delegate_to_bifurcation", task="Period of the cycles in alpha.")), say("done"))
    sub = scripted(tools(call("run_bifurcation_analysis", model_name_or_path=MODEL, param_name="alpha",
                              cycles=True)))
    rec = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False, save_records=False,
                state=SessionState(persist=False), approval="ask").run("Period of the cycles in alpha.")
    assert not rec.awaiting_approval and "_cycles" not in calls[0][1]


def test_after_the_budget_question_the_limit_cycles_come_without_a_question(monkeypatch):
    from dynalyst import config
    _accepting_reviewer(monkeypatch)
    monkeypatch.setattr(config, "SUBAGENT_MAX_ANALYSES", 1)
    calls = _fake_tools(monkeypatch, {"run_bifurcation_analysis": _hopf})
    state = SessionState(persist=False)
    orch = scripted(tools(call("delegate_to_bifurcation", task="Diagram in alpha.")), say("Answer."))
    sub = scripted(tools(call("run_bifurcation_analysis", model_name_or_path=MODEL, param_name="alpha")),
                   tools(call("run_bifurcation_analysis", model_name_or_path=MODEL, param_name="alpha",
                              n_pts=100)), say("Done."))
    a = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False, save_records=False,
              state=state, approval="ask")
    assert a.run("Diagram in alpha.").awaiting_approval and state["pending_approval"]["kind"] == "budget_wait"
    rec = a.run("yes")
    assert not rec.awaiting_approval and state["pending_approval"] is None
    assert "Reply **yes** or **no**" not in rec.final_answer
    assert all("_cycles" not in c[1] for c in calls)


def test_called_from_code_the_limit_cycles_run_without_a_question(monkeypatch):
    _accepting_reviewer(monkeypatch)
    calls = _fake_tools(monkeypatch, {"run_bifurcation_analysis": _hopf})
    orch = scripted(tools(call("delegate_to_bifurcation", task="Diagram.")), say("done"))
    sub = scripted(tools(call("run_bifurcation_analysis", model_name_or_path=MODEL, param_name="alpha")))
    rec = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False,
                save_records=False, state=SessionState(persist=False)).run("q")
    assert not rec.awaiting_approval and "_cycles" not in calls[0][1]


@pytest.mark.parametrize("reply", ["continue", "plot"])
def test_a_long_crnt_search_pauses_and_the_reply_decides(monkeypatch, reply):
    _accepting_reviewer(monkeypatch)
    stopped = []
    monkeypatch.setattr("dynalyst.analysis.crnt_analysis.stop_crnt_job", stopped.append)
    calls = _fake_tools(monkeypatch, {"crnt_bistability": lambda args: (
        {"still_running": True, "job_id": "0123456789ab", "seconds": 1200, "bistable": None}
        if args.get("_pause_after") else {"bistable": None, "multistationary_sets": []})})
    state = SessionState(persist=False)
    orch = scripted(tools(call("delegate_to_network_theory", task="CRNT of saddle_node.xml.")),
                    say("Answer after the reply."))
    sub = scripted(tools(call("crnt_bistability", model_name_or_path=MODEL)))
    a = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False,
              save_records=False, state=state, approval="ask")
    rec = a.run("Can saddle_node.xml be bistable by CRNT?")
    assert rec.awaiting_approval and state["pending_approval"]["kind"] == "crnt_wait"

    rec2 = a.run(reply)
    assert rec2.approval["choice"] == reply and rec2.final_answer.startswith("Answer after")
    if reply == "continue":     # the search is collected, not started again
        assert calls[-1] == ("crnt_bistability", {"job_id": "0123456789ab"}) and not stopped
    else:
        assert stopped == ["0123456789ab"] and len(calls) == 1


def test_a_long_result_keeps_the_fields_that_state_what_it_found():
    # a bifurcation result longer than the budget: settings and logs go first, the points
    # and the classification stay (a live run once lost them and repeated AUTO 15 times)
    out = json.dumps({
        "auto_constants_used": {f"c{i}": i * 0.1 for i in range(60)},
        "fixed_parameters": {f"k{i}": i for i in range(80)},
        "interpretation": "x" * 400, "odes": "y" * 900, "conservation_laws_detected": "z" * 700,
        "bifurcation_points": [{"type": "LP", "k2cat": 20.182318758, "M": 354.5},
                               {"type": "LP", "k2cat": 9.2026881015, "M": 49.4}],
        "classification": {"category": "bistable", "n_folds": 2}})
    seen = registry.for_model(out, 2500)
    assert "20.182318758" in seen and "9.2026881015" in seen and "bistable" in seen


def test_a_sub_agent_gets_at_most_the_task_budget(monkeypatch):
    monkeypatch.setattr(config, "MAX_TASKS_PER_SUBAGENT", 2)
    monkeypatch.setattr(Agent, "_delegate", lambda self, sub, task, rec, guard, *rest: f"{sub}: {task}")
    orch = scripted(tools(call("delegate_to_dynamics_team", task="three tries")), say("done"))
    team = scripted(tools(*[call("delegate_to_bifurcation", task=f"try {i}") for i in range(3)]),
                    say("reported"))
    rec = agent(orch, team).run("q")
    replies = [d["report"] for d in rec.team_runs[0]["delegations"]]
    assert sum(r.startswith("Not delegated") for r in replies) == 1
    assert rec.interventions == [{"kind": "task budget", "sub_agent": "bifurcation"}]


@pytest.mark.parametrize("text,claim", [
    ("No branch points or Hopf points were found.", False),     # a live answer, misread before
    ("It is unclear whether Hopf points were found.", False),
    ("AUTO found a Hopf bifurcation.", True),
    ("There is no fold, but AUTO found a Hopf bifurcation.", True),   # the contrast is a claim
])
def test_a_negated_or_hedged_clause_is_not_a_bifurcation_claim(text, claim):
    from dynalyst.orchestration.final_check import unfound_hopf_claim
    assert unfound_hopf_claim(text, []) is claim


def test_a_percentage_matches_the_fraction_a_result_gives():
    from dynalyst.orchestration.final_check import unsupported_values
    assert unsupported_values("the error is 7.51%", ['{"err": 0.07508566857652096}']) == []
    assert unsupported_values("the error is 9.99%", ['{"err": 0.0751}']) == ["9.99"]


def test_a_failure_fixed_later_in_the_run_gets_no_retry_suggestion():
    from dynalyst.orchestration.next_steps import suggest
    rec = type("R", (), {})()
    rec.steps = [{"kind": "delegation", "sub_agent": "stochastic", "tool": "stochastic_simulation",
                  "status": "error", "task": "t", "output": '{"error": "x"}', "arguments": {}},
                 {"kind": "delegation", "sub_agent": "stochastic", "tool": "stochastic_simulation",
                  "status": "ok", "task": "t", "output": "{}", "arguments": {}}]
    assert not any("did not finish cleanly" in f["question"] for f in suggest(rec))


def test_a_few_shrinking_peaks_are_reported_as_too_short_to_tell():
    import numpy as np
    import pandas as pd
    from dynalyst.analysis.jaxkm_tools import time_course_behaviour
    t = np.linspace(0, 1000, 4000)
    x = 1 + np.exp(-t / 1000) * np.sin(2 * np.pi * t / 145)      # still shrinking at the end
    out = time_course_behaviour(pd.DataFrame({"time": t, "X": x}), ["X"])
    assert out["verdict"] == "oscillating" and "cannot tell damped from sustained" in out["note"]


def test_a_model_named_like_a_paper_is_found_locally_first(monkeypatch):
    from dynalyst.analysis import biomodels
    monkeypatch.setattr(biomodels, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("searched")))
    out = json.loads(biomodels.find_biomodels_by_paper("Markevich2004"))
    assert out["local_file"] == "Markevich2004.xml" and "Use it as it is" in out["note"]


def test_only_problems_the_answer_rests_on_are_listed():
    from dynalyst.orchestration.agent import _problem_notes
    wrong_tool = {"tool": "get_odes", "status": "error", "error": "not applicable", "team_run": 0}
    right_tool = {"tool": "run_bifurcation_analysis", "status": "ok", "team_run": 0,
                  "reviewer": {"issues": ["coverage stopped short of k = 0"]}}
    other_task = {"tool": "simulate_model", "status": "error", "error": "blew up", "team_run": 1}
    notes = _problem_notes([wrong_tool, right_tool, other_task], "Folds at k = 2.")
    assert "get_odes" not in notes                  # superseded by the later success
    assert "simulate_model (blew up)" in notes            # nothing replaced it
    assert "coverage stopped short" in notes              # a warning on a result that is used
    assert _problem_notes([right_tool], "coverage stopped short of k = 0, so ...") == ""


def test_fold_ratio_reported_for_bistable_window(monkeypatch):
    """A bistable window carries upper fold / lower fold, a scale-free size."""
    from dynalyst.analysis import bifurcation_auto as ba
    import dynalyst.regime.taxonomy as tx
    monkeypatch.setattr(tx, "classify_diagram", lambda *a, **k: {
        "category": "bistable", "features": ["bistable"],
        "evidence": {"max_coexisting_stable": 2, "at": 50.0}})
    pts = [{"type": "LP", "MAPKK": 39.246}, {"type": "LP", "MAPKK": 57.3798}]
    c = ba._classification(pts, [], "MAPKK", "Mpp")
    assert c["fold_ratio"] == round(57.3798 / 39.246, 4)
    assert "fold_ratio" in c["fold_ratio_note"]
    c = ba._classification(pts[:1], [], "MAPKK", "Mpp")
    assert "fold_ratio" not in c


class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(self.replies.pop(0))


def _review_agent(reviewer_reply, revision):
    from dynalyst.orchestration.agent import Agent, RunRecord
    from dynalyst.orchestration.state import SessionState
    agent = Agent(llms={"orchestrator": _Scripted([revision]),
                        "reviewer": _Scripted([reviewer_reply])},
                  verbose=False, save_records=False, state=SessionState(persist=False))
    rec = RunRecord(query="Which enzyme controls the switch?", model={})
    rec.steps = [{"tool": "run_bifurcation_analysis", "status": "ok",
                  "output": '{"fold_ratio": 1.4621, "folds": [39.246, 57.3798]}'}]
    rec.final_answer = "MAPKK controls more strongly. Folds 39.246 and 57.3798."
    return agent, rec


def test_answer_review_returns_unsupported_claims():
    agent, rec = _review_agent(
        '{"issues": [{"claim": "MAPKK controls more strongly", "why": "equal fold_ratio"}]}', "")
    issues = agent._review_answer(rec.query, [], rec)
    assert issues[0]["claim"] == "MAPKK controls more strongly"
    assert rec.answer_review["issues"] == issues


def test_review_issues_go_back_to_the_orchestrator_conversation():
    from dynalyst.orchestration.agent import _review_note
    agent, rec = _review_agent('{"issues": []}', "Both control it equally.")
    seen = []
    agent._conversation = ("llm", ["system", "question"])
    agent._orchestrate_loop = lambda llm, messages, rec, guard: seen.append(messages) or "new"
    note = _review_note([{"claim": "MAPKK stronger", "why": "equal fold_ratio"}])
    assert agent._orchestrate_more(note, rec, None) == "new"
    assert "MAPKK stronger" in seen[0][-1].content and "beyond the question is proposed" in seen[0][-1].content


def test_answer_review_skipped_without_reviewer_in_scripted_runs():
    from dynalyst.orchestration.agent import Agent, RunRecord
    from dynalyst.orchestration.state import SessionState
    agent = Agent(llms={"orchestrator": _Scripted([])}, verbose=False, save_records=False,
                  state=SessionState(persist=False))
    rec = RunRecord(query="q", model={})
    rec.steps = [{"tool": "x", "status": "ok", "output": "{}"}]
    rec.final_answer = "a"
    agent._review_answer("q", [], rec)
    assert rec.answer_review == {} and rec.final_answer == "a"




def test_model_parameters_reads_each_parameters_part_in_the_rate_laws():
    """The orchestrator's read-only lookup: which parameters a description such as 'the
    catalytic rate' or 'the Michaelis constant' can fit, read from the rate laws."""
    from dynalyst.orchestration import tools as registry
    assert "model_parameters" in registry.UTILITY
    # the orchestrator's own tools only read: every tool that writes a file is a sub-agent's
    assert not set(registry.UTILITY) & {"save_sbml", "odes_to_sbml", "xpp_to_sbml",
                                        "xpp_text_to_sbml"}
    rows = {r["name"]: r for r in json.loads(registry.model_parameters("Markevich2004"))["parameters"]}
    assert rows["k1cat"]["in_reactions"] == ["v1 (phosphorylation of MAPK): the rate is proportional to it"]
    assert any("falls as it rises" in x for x in rows["Km1"]["in_reactions"])
    assert rows["MKP3"]["kind"] == "fixed species"
    assert rows["total_M"]["conserved_total_of"] == "M + Mp + Mpp"
    rows = {r["name"]: r for r in json.loads(registry.model_parameters("BIOMD0000000010"))["parameters"]}
    assert rows["J0_n"]["in_reactions"] == ["J0 (MAPKKK activation): exponent in the rate law"]
    assert rows["J0_Ki"]["name_in_file"] == "Ki of reaction J0"
    assert "error" in json.loads(registry.model_parameters("NoSuchModel"))


def test_filler_words_for_optional_arguments_are_dropped():
    from dynalyst.orchestration.tools import coerce_args
    out = coerce_args("run_bifurcation_analysis", {"model_name_or_path": "Markevich2004.xml",
                                                   "param_name": "MAPKK", "plot_state": "none"})
    assert "plot_state" not in out and out["param_name"] == "MAPKK"


def test_a_rounded_given_state_is_refined_to_the_steady_state_next_to_it():
    import shutil
    from dynalyst.analysis.jacobian_network import jacobian_network
    out = json.loads(jacobian_network("Markevich2004", state={"M": 437.73, "Mp": 12.85, "Mpp": 49.42},
                                      parameter_jacobian=False))
    s = out["steady_states"][0]
    shutil.rmtree(os.path.dirname(s["network_plot"]))
    assert s["refined_from_given"]["M"] == 437.73 and s["residual_max_abs_dxdt"] < 1e-9
    assert s["stability"] == "stable" and "warning" not in s     # conserved totals' zero eigenvalues


def test_a_long_result_is_trimmed_by_field_never_cut_through():
    """A small detail field before a big result list must not stop the trimming: the list
    is shortened, and the fields after it (plot path, interpretation) stay in view."""
    from dynalyst.orchestration.tools import for_model
    out = json.dumps({"initial_conditions": {"A": 1.0, "B": 2.0},
                      "results": [{"k": i, "state": {f"S{j}": 1.0 + j for j in range(8)},
                                   "stability": "stable_focus (within 3 conserved totals)"}
                                  for i in range(100)],
                      "plot_path": "outputs/plots/scan.png", "interpretation": "done"})
    view = for_model(out)
    data = json.loads(view)                       # still valid JSON: nothing cut part-way
    assert data["plot_path"] == "outputs/plots/scan.png" and data["interpretation"] == "done"
    assert len(view) <= 10000 and "evenly spaced" in str(data["results"][-1])
    shown = [r["k"] for r in data["results"] if isinstance(r, dict)]
    assert shown[0] == 0 and shown[-1] == 99      # a sweep: its whole range, not its start


def test_a_cycle_branch_that_leaves_the_range_keeps_its_stretches_and_its_connection():
    from dynalyst.analysis.bifurcation_auto import _branch_in_order, _stretches
    pts = [{"p": p, "stable": st, "period": t} for p, st, t in [
        (97.8, False, 25.0), (99.9, False, 25.5), (116.1, False, 37.0),      # out of the range, to the fold
        (104.5, True, 41.0), (93.1, True, 43.0), (40.0, True, 23588.0)]]     # and back, stable
    assert _stretches(pts, True, (0, 100)) == [[40.0, 100]]     # no saved stable point from 93.1 to 100
    assert _stretches(pts, False, (0, 100)) == [[97.8, 100]]
    assert _stretches(pts, True, (200, 300)) == []
    assert _branch_in_order(pts) == [
        {"stability": "unstable", "from": 97.8, "to": 116.1, "period_from": 25.0, "period_to": 37.0},
        {"stability": "stable", "from": 116.1, "to": 40.0, "period_from": 37.0, "period_to": 23588.0}]


def test_cycle_stability_unknown_and_disjoint_stable_stretches_are_kept_apart():
    """AUTO's stability list not covering a point means unknown, not stable; two stable
    stretches with unstable cycles between them are two stretches; period and amplitude
    ranges are per stability, never mixed."""
    import numpy as np
    from dynalyst.analysis.bifurcation_auto import (_cycles_by_stability, _point_stability_signs,
                                                    _stretches)
    assert list(_point_stability_signs(6, [-2, 4])) == [1, 1, -1, -1, 0, 0]
    pts = [{"p": p, "stable": s, "period": per, "min": 0.0, "max": a}
           for p, s, per, a in [(1, True, 10, 1), (2, True, 11, 2), (3, False, 50, 9),
                                (4, True, 12, 3), (5, None, 13, 4)]]
    assert _stretches(pts, True) == [[1, 2], [4, 4]] and _stretches(pts, None) == [[5, 5]]
    by = _cycles_by_stability(pts, "X")
    assert by["stable"]["period_range"] == [10, 12] and by["unstable"]["period_range"] == [50, 50]
    assert by["stability_unknown"]["X_range"] == [0.0, 4]


def test_a_coarsely_sampled_oscillation_is_read_again_on_a_finer_grid():
    """About 5 samples per cycle miss the true peaks: a constant oscillation reads as
    shrinking with a wrong period. The behaviour is read again on enough points."""
    import numpy as np
    import pandas as pd
    from dynalyst.analysis.jaxkm_tools import resolved_behaviour
    run = lambda n: pd.DataFrame({"time": (t := np.linspace(0, 1000, n)),
                                  "X": 1 + np.sin(2 * np.pi * t / 2.763) ** 9})
    out = resolved_behaviour(run(2000), ["X"], run)
    o = out["oscillating_species"]["X"]
    assert abs(o["period"] - 2.763) < 0.01 and o["amplitude_trend"] == "approximately constant"
    assert "resolution_note" in out


def test_a_species_driven_below_zero_is_named_when_the_run_fails():
    import numpy as np
    import sympy as sp
    from dynalyst.analysis.deterministic import integrate
    x, k = sp.symbols("X k")
    model = {"label": "drain", "state_names": ["X"], "state_symbols": [x], "param_names": ["k"],
             "param_symbols": [k], "rhs": [-k + 0.001 * x ** sp.Float(4.1)], "y0": {"X": 0.1},
             "source": "sbml", "source_path": None}      # a constant drain: X is pushed below 0
    with pytest.raises(ValueError, match="X .*(went below zero|reached zero)"):
        integrate(model, {"k": 1.0}, np.linspace(0, 10, 50))


def test_self_loops_and_the_reactions_behind_each_loop_step():
    """A species raising its own rate (here: Ca2+-induced Ca2+ release, v_2 in Borghans) is a
    positive loop of length 1; each loop step lists the reactions that make it."""
    import shutil
    from dynalyst.analysis.jacobian_network import jacobian_network
    out = json.loads(jacobian_network("Borghans_BiophysChem1997", parameter_jacobian=False))
    s = out["steady_states"][0]
    shutil.rmtree(os.path.dirname(s["network_plot"]))
    loops = {lp["loop"]: lp for lp in s["loops"]["list"]}
    assert loops["Z_state -> Z_state"]["sign"] == "positive"
    assert s["self_regulation_by_reaction"]["Z_state"][0]["reaction"] == "v_2"
    step = loops["Z_state -> A_state -> Z_state"]["steps"][0]
    assert step["step"] == "Z_state -> A_state" and step["by_reaction"][0]["reaction"] == "v_6"



def test_a_stretch_of_a_branch_reaches_the_next_one():
    from dynalyst.analysis.bifurcation_auto import _segment
    xs, ys = [1.0, 2.0, 3.0, 4.0], [0.0, 0.1, 0.2, 0.3]
    first = _segment(xs[0:3], ys[0:3], (0.0, 5.0), "k", "X")       # stops where the next one starts
    assert first["k_to"] == 3.0


def test_a_refused_call_offers_no_next_step_about_a_missing_parameter():
    from types import SimpleNamespace
    from dynalyst.orchestration.next_steps import suggest
    step = {"kind": "delegation", "sub_agent": "bifurcation", "tool": "run_bifurcation_analysis",
            "status": "ok", "task": "t", "arguments": {"model_name_or_path": "m"}, "output": "{}"}
    assert not any("None" in q["question"] for q in suggest(SimpleNamespace(steps=[step])))


def test_the_guard_sees_one_model_under_both_names():
    from dynalyst.orchestration.guard import ExecutionGuard, call_key
    a = {"model_name_or_path": "Markevich2004", "param_name": "MAPKK"}
    b = {"model_name_or_path": "Markevich2004.xml", "param_name": "MAPKK"}
    assert call_key("run_bifurcation_analysis", a) == call_key("run_bifurcation_analysis", b)
    guard = ExecutionGuard()
    for n in range(3):      # three different diagrams in one parameter, then no more
        args = {**(a if n % 2 else b), "param_min": n}
        assert guard.check("bifurcation", "run_bifurcation_analysis", args).allowed
        guard.record("run_bifurcation_analysis", args, {"status": "ok"})
    assert not guard.check("bifurcation", "run_bifurcation_analysis", {**a, "param_min": 9}).allowed
    assert guard.check("bifurcation", "run_bifurcation_analysis",
                       {"model_name_or_path": "Markevich2004", "param_name": "MKP3"}).allowed


def test_a_range_not_covered_is_no_verdict():
    from dynalyst.regime.taxonomy import classify_diagram
    x, y = np.linspace(1, 2, 50), np.linspace(1, 2, 50)
    curves = [(x, y, np.ones(50, dtype=int))]
    assert classify_diagram([], curves, "k", "X", covered=False)["category"] == "incomplete"
    assert classify_diagram([], curves, "k", "X")["category"] in ("graded", "ultrasensitive")
    assert classify_diagram(["LP"], curves, "k", "X", covered=False)["category"] == "single_fold"


def test_a_fold_on_a_branch_point_is_the_same_point():
    from dynalyst.analysis.bifurcation_auto import _without_folds_on_branch_points as clean
    bp = {"type": "BP", "k": 0.0, "X": 0.0}
    lp = {"type": "LP", "k": 1e-9, "X": 1e-9}
    far = {"type": "LP", "k": 2.0, "X": 0.0}
    other_state = {"type": "LP", "k": 0.0, "X": 5.0}
    assert clean([lp, bp, far, other_state], "k", 6.0) == [bp, far, other_state]
    assert clean([far], "k", 6.0) == [far]


def test_a_survey_is_offered_the_same_candidates_as_a_diagram(monkeypatch):
    from dynalyst.analysis import candidates as cd
    from dynalyst.orchestration import approval
    from dynalyst.orchestration.agent import Agent
    from dynalyst.orchestration.state import SessionState
    monkeypatch.setattr(cd, "propose", lambda model: {"candidates": [{"parameter": "A"}, {"parameter": "B"}]})
    monkeypatch.setattr(approval, "message", lambda *x: "pick")
    a = Agent(verbose=False, save_records=False, state=SessionState(persist=False), approval="ask")
    with pytest.raises(approval.AwaitApproval) as asked:
        a._ask_approval("lhs_bifurcation_scan", {"model": "m", "vary_only": "A, C"}, {}, SimpleNamespace(query="q"))
    assert [c["parameter"] for c in asked.value.pending["candidates"]] == ["A", "B"]


def test_a_diagram_from_another_start_state_is_not_a_repeat():
    from dynalyst.orchestration.guard import ExecutionGuard
    a = {"model_name_or_path": "m", "param_name": "k"}
    guard = ExecutionGuard()
    for n in range(3):
        guard.record("run_bifurcation_analysis", {**a, "param_min": n}, {"status": "ok"})
    assert not guard.check("bifurcation", "run_bifurcation_analysis", {**a, "param_min": 9}).allowed
    lower = {**a, "initial_conditions": {"V": -59.0}}
    assert guard.check("bifurcation", "run_bifurcation_analysis", lower).allowed
    for n in range(3):
        guard.record("run_bifurcation_analysis", {**lower, "param_min": n}, {"status": "ok"})
    assert not guard.check("bifurcation", "run_bifurcation_analysis", {**lower, "param_min": 9}).allowed


def test_a_negative_value_is_wrong_only_for_a_state_the_model_keeps_non_negative():
    from dynalyst.orchestration.reviewer import Reviewer
    def sign_check(rhs, value):
        args = dict(species=["x"], odes={"x": rhs}, parameters={"I": -2.0}, initial_conditions={"x": 0.0})
        data = {"steady_states": [{"state": {"x": value}, "stability": "stable node"}],
                "parameters_used": {"I": -2.0}}
        return next(c for c in Reviewer()._review_find_steady_states(args, data, "{}")
                    if c.name == "non_negative").passed
    assert sign_check("I - x", -2.0)            # a voltage-like variable: its rate at 0 is negative
    assert not sign_check("I + 3 - x", -1.0)        # a concentration: its rate at 0 is production


def test_a_provider_server_error_is_tried_again(monkeypatch):
    from dynalyst.orchestration import agent as ag

    class Busy(Exception):
        status_code = 503

    class Llm:
        calls = 0

        def invoke(self, messages):
            Llm.calls += 1
            if Llm.calls == 1:
                raise Busy()
            return "ok"
    monkeypatch.setattr(ag.time, "sleep", lambda s: None)
    assert ag._invoke(Llm(), []) == "ok" and Llm.calls == 2


def test_a_budget_reached_in_chat_asks_once(monkeypatch):
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.agent import Agent
    from dynalyst.orchestration.state import SessionState
    from tests.fakes import call, say, scripted, tools, via_teams
    monkeypatch.setattr(registry, "execute", lambda name, args: json.dumps({"steady_states": [{"state": {}}]}))
    many = [call("find_steady_states", model_name_or_path="Markevich2004.xml", fixed_params={"k": i})
            for i in range(4)]
    sub = scripted(tools(*many), say("Ran them."))
    orch = scripted(tools(call("delegate_to_steady_state", task="Steady states at four values")))
    agent = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}), verbose=False,
                  save_records=False, state=SessionState(persist=False), approval="ask")
    rec = agent.run("Steady states at four values")
    assert rec.awaiting_approval and "limit of **3" in rec.final_answer
    assert agent.state["pending_approval"]["kind"] == "budget_wait"


def test_a_fold_of_cycles_with_no_amplitude_is_named():
    from dynalyst.analysis.bifurcation_auto import _folds_of_no_size
    pts = [{"type": "LP", "p": 0.33128133, "max": -0.9672, "min": -0.9676},   # FHN, on the Hopf point
           {"type": "LP", "p": 1.42582, "max": 1.98, "min": -1.5},
           {"type": "RG", "p": 0.4, "max": -0.9, "min": -0.9}]
    assert _folds_of_no_size(pts) == [0.33128]
    assert _folds_of_no_size([]) == []


def test_hopf_type_does_not_claim_cycle_stability():
    from dynalyst.analysis.bifurcation_auto import _l1_type
    data = {"hopf_l1": [{"parameter": 1.0, "l1": -2.0}]}
    assert "Floquet" in _l1_type(data, 1.0) and "stable cycles" not in _l1_type(data, 1.0)


def test_hopf_type_is_read_from_the_side_its_cycles_appear_on():
    """The Hopf type is AUTO's own first Lyapunov coefficient (l1 < 0 supercritical, > 0
    subcritical); a neutral saddle (real eigenvalues summing to 0) is not a Hopf point."""
    import numpy as np
    from dynalyst.analysis.bifurcation_auto import _is_hopf, _l1_type
    data = {"hopf_l1": [{"parameter": 23.8951, "l1": -0.02}, {"parameter": 7.18, "l1": 0.5}]}
    assert _l1_type(data, 23.8951).startswith("supercritical")
    assert _l1_type(data, 7.18).startswith("subcritical")
    assert _l1_type(data, 1.0).startswith("unknown")
    assert _is_hopf(np.array([1e-9 + 2.0j, 1e-9 - 2.0j, -3.0])) and _is_hopf(None)
    assert not _is_hopf(np.array([1.5, -1.5, -3.0]))                     # neutral saddle


def test_a_loop_has_one_name_whichever_species_networkx_starts_it_at():
    from dynalyst.analysis.jacobian_network import _loops
    edges = [{"source": "Z", "target": "A", "sign": "+"}, {"source": "A", "target": "Z", "sign": "-"}]
    for names in (["Z", "Y", "A"], ["A", "Z"]):
        assert [lp["loop"] for lp in _loops(edges, names, 4)] == [f"{names[0]} -> " + (
            "A -> Z" if names[0] == "Z" else "Z -> A")]


def test_a_missing_file_of_the_run_is_not_reported_as_auto_missing(monkeypatch):
    """AUTO removes and rewrites its fort.* files while it runs: one gone at the wrong moment
    is a failure of that run, not "AUTO-07p cannot run on this machine"."""
    from dynalyst.analysis import bifurcation_auto as ba

    def gone(work_dir, label, timeout_override=None):
        raise FileNotFoundError(2, "The system cannot find the file specified",
                                str(work_dir / "fort.8"))
    monkeypatch.setattr(ba, "_run_auto", gone)
    out = json.loads(ba.run_bifurcation_analysis("Markevich2004.xml", param_name="MAPKK",
                                                 param_min=5, param_max=500))
    assert "auto_unavailable" not in out and "cannot run on this machine" not in out["error"]


def test_why_a_cycle_branch_ends_is_read_from_the_run_not_the_label():
    """An EP inside the range can be NMX running out; MX is no convergence; a branch that
    shrinks into a Hopf point or leaves the range is complete."""
    from dynalyst.analysis.bifurcation_auto import _cycle_end
    req = (0.0, 1.0)
    assert _cycle_end(None, "hopf", 0, 0.5, req, "p") == {"branch_end": "hopf"}
    assert _cycle_end({"point": 120, "type": "EP", "nmx": 500}, None, 0, 1.2, req, "p") == {"branch_end": "range_end"}
    cases = {"step_limit": ({"point": 500, "type": "EP", "nmx": 500}, None, 0),
             "no_convergence": ({"point": 316, "type": "MX", "nmx": 500}, None, 0),
             "diverged": ({"point": 500, "type": "EP", "nmx": 500}, None, 12),
             "zero_period": (None, "zero_period", 0),
             "unexplained": ({"point": 80, "type": "EP", "nmx": 500}, None, 0)}
    for cause, (end, stop, div) in cases.items():
        out = _cycle_end(end, stop, div, 0.4, req, "p")
        assert out["branch_end"] == cause and out["stopped_at"] == 0.4 and "0.4" in out["incomplete"]


def test_failed_auto_runs_are_reported():
    from dynalyst.analysis.bifurcation_auto import _auto_problems
    assert _auto_problems({"diagnostics": {"run_records": [{"tag": "main-forward"}]}}) == {}
    out = _auto_problems({"errors": ["main-backward: boom"], "diagnostics": {
        "run_records": [{"tag": "main-backward", "error": "boom"}], "hit_switch_cap": True}})["auto_problems"]
    assert len(out) == 2 and "only one direction" in out[0] and "branch-switch limit" in out[1]


def test_a_no_point_note_is_not_added_when_another_run_found_points():
    import json as _json
    from dynalyst.orchestration.final_check import qualify_empty_continuation
    step = {"tool": "run_bifurcation_analysis", "status": "ok"}
    empty = _json.dumps({"swept_parameter": "Vm3", "swept_range": [0, 30], "bifurcation_points": []})
    hopf = _json.dumps({"swept_parameter": "Vm3", "swept_range": [0, 30],
                        "bifurcation_points": [{"type": "HB", "Vm3": 6.9}]})
    answer = "No bifurcation was found on that branch."
    assert qualify_empty_continuation(answer, [{**step, "output": empty}]) != answer
    assert qualify_empty_continuation(answer, [{**step, "output": hopf}, {**step, "output": empty}]) == answer


def test_sign_rules_voltage_and_fraction():
    """A voltage may be negative (FitzHugh-Nagumo); a fraction above 1 runs (1 - M) backwards."""
    import json
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.analysis.steady_states import kept_nonnegative, find_steady_states
    fhn = resolve_model("fhn_check", None, None, None, None)
    assert kept_nonnegative(fhn, fhn["param_defaults"]) == set()
    r = json.loads(find_steady_states("BIOMD0000000008", fixed_params={"reaction1_vi": 0.033619433053},
                                      n_random_starts=20))
    stable = [s["state"] for s in r["steady_states"] if s["stability"].startswith("stable")]
    assert stable and all(s["M"] <= 1 and s["X"] <= 1 for s in stable)


def test_conservation_law_mixed_with_known_law():
    """BIOMD39's total Ca (Ca_cyt + 4 CaER + 4 CaM + CaPr) is found exactly, not dropped."""
    from dynalyst.analysis.ode_model import resolve_model
    from dynalyst.analysis.steady_states import exact_conservation_laws, law_is_exact
    laws = exact_conservation_laws(resolve_model("BIOMD0000000039", None, None, None, None))
    assert [1, 4, 4, 1, 0] in [[int(x) for x in c] for c in laws]
    assert all(law_is_exact(c) for c in laws)


def test_oscillation_pattern_simple_or_complex():
    """Main peaks going up AND down mark a complex oscillation; a damped one, a sampled one and
    one that settles after an overshoot are simple."""
    import numpy as np
    import pandas as pd
    from dynalyst.analysis.jaxkm_tools import time_course_behaviour
    t = np.linspace(0, 200, 4000)        # ~20 samples per cycle of sin(t), not a whole number
    cases = {"sine": (np.sin(t), "simple"),
             "damped": (np.exp(-t / 300) * np.sin(t), "simple"),
             "overshoot": ((1 + 0.5 * np.exp(-t / 5)) * np.sin(t), "simple"),
             "period_two": (np.sin(2 * t) + 0.6 * np.sin(t), "complex")}
    for name, (y, want) in cases.items():
        b = time_course_behaviour(pd.DataFrame({"time": t, "x": y + 2.0}), ["x"])
        assert b["oscillating_species"]["x"]["pattern"] == want, (name, b["oscillating_species"]["x"])
