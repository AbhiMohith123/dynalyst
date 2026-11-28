"""custom_plot series mode (plot numbers earlier results gave) and the check that refuses
numbers no result gave; the simulation mode must behave as before."""
import json

import pandas as pd
import pytest

from dynalyst.analysis import custom_plot as cp
from dynalyst.orchestration.final_check import unsupported_points

FOLDS = [{"label": "lower fold", "x": [300, 500], "y": [49.21, 39.25]},
         {"label": "upper fold", "x": [300, 500], "y": [59.12, 57.38]}]


def test_series_are_plotted_without_simulating(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "PLOTS_DIR", tmp_path)
    monkeypatch.setattr(cp, "_get_trajectory", lambda *a, **k: pytest.fail("series mode simulated"))
    out = json.loads(cp.custom_plot("Markevich2004", series=FOLDS, x_label="total_M", y_label="MAPKK"))
    assert out["series"][0]["y"] == [49.21, 39.25] and out["x_axis"] == "total_M"
    assert list(tmp_path.glob("Markevich2004_series_*.png"))


@pytest.mark.parametrize("bad", ["not a list", [{"x": [1, 2], "y": [1]}], [{"x": ["a"], "y": [1]}],
                                 [{"x": [1], "y": [float("nan")]}], [{"y": [1]}]])
def test_malformed_series_is_refused(monkeypatch, tmp_path, bad):
    monkeypatch.setattr(cp, "PLOTS_DIR", tmp_path)
    assert "error" in json.loads(cp._plot_series("m", bad, None, None, False, False, None))


def test_simulation_mode_is_unchanged_when_series_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "PLOTS_DIR", tmp_path)
    df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "X": [0.0, 1.0, 1.0]})
    calls = []
    monkeypatch.setattr(cp, "_get_trajectory", lambda *a, **k: calls.append(a) or (df, ["X"], "toy", None))
    for empty in (None, []):
        out = json.loads(cp.custom_plot("toy", species=["X"], t_end=2, series=empty))
        assert out["plotted_species"] == ["X"] and out["conditions"][0]["final"]["X"] == 1.0
    assert len(calls) == 2


def test_plotted_numbers_must_come_from_earlier_results():
    evidence = ['{"bifurcation_points": [{"type": "LP", "MAPKK": 49.2104}, {"type": "LP", "MAPKK": 59.1188},'
                ' {"type": "LP", "MAPKK": 39.24602}, {"type": "LP", "MAPKK": 57.37975}]}',
                '{"fixed_params": {"total_M": 300}}', '{"fixed_params": {"total_M": 500}}']
    assert unsupported_points(FOLDS, evidence) == []
    made_up = FOLDS + [{"label": "cusp", "x": [143.3], "y": [65.13]}]
    assert unsupported_points(made_up, evidence) == [143.3, 65.13]
    assert unsupported_points("not a list", evidence) == []


def test_agent_refuses_made_up_points_and_passes_results_and_simulations(tmp_path, monkeypatch):
    """In the agent: a series with a number no result gave is refused before the tool runs; a
    series copied from an earlier result and an ordinary simulation plot run as usual."""
    import contextlib
    import io

    from dynalyst.orchestration.agent import Agent
    from tests.fakes import call, say, scripted, tools, via_teams
    monkeypatch.setattr(cp, "PLOTS_DIR", tmp_path)
    sub = scripted(
        tools(call("custom_plot", model_name_or_path="pitchfork_model.xml", species=["APP"], t_end=50)),
        tools(call("custom_plot", model_name_or_path="pitchfork_model.xml", x_label="t", y_label="APP",
                   series=[{"label": "made up", "x": [1, 2], "y": [123.456, 7891.5]}])),
        say("Ran: custom_plot. Result: plot. Problems: one call refused"),
    )
    orch = scripted(
        tools(call("write_plan", steps=["plotting: plot APP of pitchfork_model.xml"])),
        tools(call("delegate_to_plotting", task="Plot APP of pitchfork_model.xml to t = 50")),
        say("Plotted APP."),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        rec = Agent(llms=via_teams({"orchestrator": orch, "subagent": sub}),
                    save_records=False).run("Plot APP of pitchfork_model.xml to t = 50")
    sim, made_up = [s for s in rec.steps if s["kind"] == "delegation" and s["tool"] == "custom_plot"]
    assert sim["status"] == "ok" and '"plotted_species"' in sim["output"]
    assert made_up["status"] == "rejected" and "123.456" in made_up["error"] and made_up["output"] is None
    # a series built from the simulation's own result passes the same check
    final = json.loads(sim["output"])["conditions"][0]["final"]["APP"]
    evidence = [sim["output"]]
    assert unsupported_points([{"x": [50], "y": [round(final, 4)]}], evidence) == []
