"""When AUTO-07p itself cannot build models on a machine (object files compiled
elsewhere, no gfortran / make, never built), every tool must say so and stop --
not report "failed to converge", "no branches" or "no bistable set", and not
retry with other parameters. Reproduces the linker error
'lib/ae.o: file not recognized: file format not recognized' (the bundled
lib/*.o were built on a Mac) with a fake AUTO, so it runs anywhere."""
import json


import pytest

from dynalyst.auto_env import AUTO_UNAVAILABLE, build_failure, unavailable_message
from tests.fakes import via_teams

LINKER = ("/E-drive/x/vendor/auto-07p/lib/ae.o: file not recognized: file format not recognized\n"
          "collect2: error: ld returned 1 exit status\n")


def test_build_failures_are_told_apart_from_numerical_failures():
    assert "file format not recognized" in build_failure("gfortran -c m.f90\n" + LINKER)
    assert build_failure("[WinError 2] The system cannot find the file specified")
    assert build_failure("[Errno 2] No such file or directory: 'gfortran'")
    assert build_failure("gfortran: command not found")
    for numerical in ("MX: no convergence", "Error running AUTO", "AUTO-07p produced no branches.",
                      "Branch 1 ended at step 200: maximum number of steps"):
        assert build_failure(numerical) is None


@pytest.fixture
def auto_cannot_build(monkeypatch):
    """The main AUTO runner reports that AUTO cannot build a model on this machine."""
    from dynalyst.analysis import bifurcation_auto
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        return json.dumps({"error": unavailable_message(LINKER), "auto_unavailable": True})
    monkeypatch.setattr(bifurcation_auto, "_run_single_bifurcation_analysis", unavailable)
    return calls


def test_an_lhs_batch_stops_at_the_first_sample(auto_cannot_build):
    from dynalyst.orchestration import tools as registry
    out = registry.execute("lhs_bifurcation_scan", {"model": "Markevich2004.xml", "n_samples": 10,
                                                    "bif_param": "k2cat", "param_min": 1.5,
                                                    "param_max": 150})
    assert AUTO_UNAVAILABLE in json.loads(out)["error"]
    assert len(auto_cannot_build) == 1           # not ten failed samples


def test_the_follow_up_says_rebuild_not_retry_with_other_settings():
    from types import SimpleNamespace
    from dynalyst.orchestration.next_steps import suggest
    step = {"kind": "delegation", "sub_agent": "bifurcation", "tool": "lhs_bifurcation_scan",
            "status": "unresolved", "task": "LHS", "arguments": {"model": "m"},
            "output": json.dumps({"error": f"{AUTO_UNAVAILABLE}: collect2"})}
    q = suggest(SimpleNamespace(steps=[step]))[0]["question"]
    assert "build_auto.sh" in q and "different settings" not in q


def test_run_bifurcation_analysis_does_not_try_other_parameters(monkeypatch):
    from dynalyst.analysis import bifurcation_auto
    calls = []

    def unavailable(**kw):
        calls.append(kw)
        return json.dumps({"error": f"{AUTO_UNAVAILABLE}: collect2", "auto_unavailable": True})

    monkeypatch.setattr(bifurcation_auto, "_run_single_bifurcation_analysis", unavailable)
    out = json.loads(bifurcation_auto.run_bifurcation_analysis("Markevich2004"))
    assert out.get("auto_unavailable") and len(calls) == 1


def test_crnt_does_not_need_auto(monkeypatch):
    from dynalyst.analysis import bifurcation_auto
    from dynalyst.analysis.crnt_analysis import crnt_bistability

    def no_auto(**kw):
        raise AssertionError("CRNT ran AUTO")
    monkeypatch.setattr(bifurcation_auto, "_run_single_bifurcation_analysis", no_auto)
    d = json.loads(crnt_bistability("Full_model_for_SN", iterations=5))
    assert "error" not in d and d["bistable"] is not True


def test_through_the_agent_one_call_then_a_clear_answer(auto_cannot_build):
    import contextlib
    import io
    from dynalyst.orchestration.agent import Agent
    from tests.fakes import call, say, scripted, tools
    sub = scripted(tools(call("lhs_bifurcation_scan", model="Markevich2004.xml", n_samples=10,
                              bif_param="k2cat", param_min=1.5, param_max=150)))
    orch = scripted(tools(call("delegate_to_bifurcation", task="LHS")),
                    say("AUTO-07p cannot run on this machine; rebuild it."))
    agent = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                  verbose=False, save_records=False)
    with contextlib.redirect_stdout(io.StringIO()):
        rec = agent.run("LHS sample 10 points of Markevich2004.xml continuing in k2cat over [1.5, 150]")
    calls = [s for s in rec.steps if s.get("tool") == "lhs_bifurcation_scan"]
    assert len(calls) == 1 and calls[0]["status"] == "error"   # the error came back, no retries
    assert "build_auto.sh" in rec.final_answer


def test_the_reviewer_stops_instead_of_replanning():
    from dynalyst.orchestration.reviewer import Reviewer
    out = json.dumps({"error": f"AutoUnavailable: {AUTO_UNAVAILABLE}: collect2: error: ld returned 1"})
    v = Reviewer().review("lhs_bifurcation_scan", {"model": "m"}, out, attempt=0)
    assert v.verdict == "unresolved" and "ld returned" in v.suggestion
