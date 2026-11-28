"""Switching events in stochastic runs and propensity values at a state (ported from the
fresh_agent_v3 step-7 tests, 2026-11-20)."""
import json

import numpy as np


def test_propensity_table_gives_values_at_the_initial_state():
    from dynalyst.analysis.stochastic import propensity_table
    out = json.loads(propensity_table("Markevich2004"))
    w = out["values_at_state"]["propensities"]
    assert abs(max(w.values()) - 0.4545454) < 1e-5 and sorted(w.values())[:3] == [0.0, 0.0, 0.0]


def test_switching_events_are_found_between_peak_cores():
    from dynalyst.analysis.stochastic_switching import find_switches
    grid = np.arange(12.0)
    # low (around 2) -> high (around 25) -> low, with a transit through the middle
    run = np.array([2, 3, 2, 8, 15, 22, 25, 24, 18, 9, 2, 1], dtype=float)
    md = {"modes": [{"at": 2.0}, {"at": 25.0}], "valleys": [14.0], "n_modes": 2}
    ev = find_switches(run[None, :], md, grid)
    assert [(e["from_peak"], e["to_peak"]) for e in ev] == [(0, 1), (1, 0)]
    assert ev[0]["t_leave"] == 3.0 and ev[0]["t_arrive"] == 5.0
    assert find_switches(run[None, :], {"modes": [{"at": 2.0}], "valleys": []}, grid) == []


def test_a_one_way_escape_is_found_from_the_ode_stable_states():
    """Near a fold every run leaves the low state and none returns, so the pooled distribution
    has one peak; the switch is then found between the ODE's stable states."""
    from dynalyst.analysis.stochastic_switching import _ode_modes, find_switches
    entry = {"omega": 1.0, "modes": {"modes": [{"at": 487.0}], "valleys": []},
             "attractors": [{"kind": "steady state", "stability": "stable", "readout": 106.4},
                            {"kind": "steady state", "stability": "unstable", "readout": 155.3},
                            {"kind": "steady state", "stability": "stable", "readout": 486.8}]}
    md = _ode_modes(entry)
    assert [m["at"] for m in md["modes"]] == [106.4, 486.8] and md["valleys"] == [155.3]
    grid = np.linspace(0, 100, 101)
    series = np.array([np.where(grid < 40, 106.0, 487.0), np.where(grid < 70, 106.0, 487.0)])
    ev = find_switches(series, md, grid)
    assert [(e["run"], e["from_peak"], e["to_peak"]) for e in ev] == [(0, 0, 1), (1, 0, 1)]


def test_detected_switches_replace_a_no_switching_line():
    from dynalyst.analysis.stochastic import _note_switching
    bad = "The ODE has 2 stable states, but ... within t_end no run left the first state (or ...)"
    out = {"interpretation": ["Ω=1, SSA: bistable -- " + bad, "other line"],
           "stability_verdicts": [{"omega": 1.0, "method": "SSA", "verdict": "bistable"}],
           "per_omega": [{"distribution": {"by_method": {"SSA": {
               "verdict": "bistable (deterministic only at this system size)", "explanation": [bad]}}}}]}
    info = {"omega": 1.0, "readout": "Mpp",
            "jumps": [{"run": 0, "from_peak": 0, "to_peak": 1, "t_leave": 9000.0},
                      {"run": 1, "from_peak": 0, "to_peak": 1, "t_leave": 11000.0}]}
    _note_switching(out, info)
    assert "no run left" not in str(out) and "deterministic only" not in str(out)
    assert "2 to the higher state" in out["switching_summary"] and "median t = 1e+04" in out["switching_summary"]


def test_switching_counts_cover_every_jump_not_only_the_listed_ones(tmp_path):
    """The plot keeps 50 jumps in its list; the summary must still count all of them."""
    from dynalyst.analysis.stochastic import _note_switching
    from dynalyst.analysis.stochastic_switching import plot
    grid = np.arange(400.0)
    run = np.where((np.arange(400) // 5) % 2 == 0, 2.0, 25.0)        # 79 jumps in one run
    entry = {"series": run[None, :], "omega": 1.0, "readout": "X", "method": "SSA",
             "modes": {"modes": [{"at": 2.0}, {"at": 25.0}], "valleys": [14.0], "n_modes": 2},
             "attractors": []}
    got = plot(entry, grid, "toy", tmp_path / "s.png")
    assert got["jumps_in_all_runs"] == 79 and len(got["jumps"]) == 50
    out = {}
    _note_switching(out, {"omega": 1.0, "readout": "X", **got})
    assert out["switching_summary"].startswith("Switching (omega = 1, X): 79 jump(s)")
    assert "40 to the higher state, 39 to the lower" in out["switching_summary"]
