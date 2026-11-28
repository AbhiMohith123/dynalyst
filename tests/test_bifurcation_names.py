"""Long parameter names (AUTO-07p cuts names to 13 characters), windows for a
parameter whose value is 0. No AUTO run needed."""
from dynalyst.analysis.bifurcation_auto import (_full_coordnames, _past_unstable_edges, _unstable_edges,
                                               _window_around)


def test_names_auto_cut_to_13_characters_are_mapped_back():
    full = ["Reaction12_kcat_f", "Reaction3_k2", "Reaction3_k2prime", "S1", "S2"]
    got = _full_coordnames(["Reaction12_kc", "L2-NORM", "S1", "S2"], full)
    assert got == ["Reaction12_kcat_f", "L2-NORM", "S1", "S2"]
    # a cut name that fits two full names stays as it is (no guessing)
    assert _full_coordnames(["Reaction3_k2p"], ["Reaction3_k2prime", "Reaction3_k2pX"]) == ["Reaction3_k2p"]
    # short names are never treated as cut
    assert _full_coordnames(["k"], ["k1", "k2"]) == ["k"]


def test_a_zero_valued_parameter_gets_both_signs_and_negatives_stay():
    lo, hi = _window_around(0.0, [0.4])           # model value 0, crossing at 0.4
    assert lo == -hi and hi > 0.4                 # the other sign is explored as far
    lo, hi = _window_around(0.0, [-0.0333])       # BIOMD0000000180, i2: a crossing below 0
    assert lo < -0.0333 and hi == -lo
    lo, hi = _window_around(1.0, [1.5])
    assert 0 < lo < 1.0 and hi > 1.5
    lo, hi = _window_around(-1.0, [0.5])          # a genuinely negative parameter keeps its margin
    assert lo < -1.0


def _probe(swept, params, stable, physical=None, points=()):
    return {"swept_range": list(swept), "branches": [{"parameter": params, "stable": stable,
                                                       "physical": physical or [True] * len(params),
                                                       "points": [{"index": i} for i in points]}]}


def test_a_special_point_found_past_the_window_end_moves_the_window():
    """Borghans in beta: the default window [0.1124, 11.24]; the probe's run reaches 0.083,
    the second Hopf point, past the lower end."""
    first = _probe((0.1124, 11.24), [1.124, 0.5, 0.0834, 1.166, 11.24],
                   [False, False, False, True, True], points=(2, 3))
    window, points = _past_unstable_edges(first, 1.124, [1.166], lambda w: 1 / 0)   # no extra probe
    assert points == [0.0834, 1.166] and window[0] < 0.0834 and 1.166 < window[1] < 11.24


def test_an_unstable_steady_state_at_a_window_end_widens_that_side():
    first = _probe((0.1124, 11.24), [0.1124, 0.5, 1.166, 5.0, 11.24], [False, False, True, True, True],
                   points=(2,))
    assert _unstable_edges(first) == ["lo"]
    asked = []

    def probe(wide):
        asked.append(wide)
        return _probe(wide, [0.01, 0.0834, 1.166], [True, False, True], points=(1, 2))
    window, points = _past_unstable_edges(first, 1.124, [1.166], probe)
    assert asked[0][0] == 0.1124 / 10 and asked[0][1] == 11.24     # only the unstable side moved
    assert points == [0.0834, 1.166] and window[0] < 0.0834 and 1.166 < window[1] < 11.24


def test_a_window_stays_when_both_ends_are_stable_or_the_wider_probe_finds_nothing_new():
    stable_ends = _probe((0.1, 10), [0.1, 1.0, 10], [True, False, True], points=(1,))
    assert _past_unstable_edges(stable_ends, 1.0, [1.0], lambda w: 1 / 0) == (None, [1.0])
    unstable_end = _probe((0.1, 10), [0.1, 0.5, 10], [False, True, True], points=(1,))
    assert _past_unstable_edges(unstable_end, 1.0, [0.5],
                                lambda w: _probe(w, [0.5], [True], points=(0,))) == (None, [0.5])
    # a non-physical (negative) branch at the end is no reason to widen
    assert _unstable_edges(_probe((0.1, 10), [0.1, 10], [False, True], [False, True])) == []


def test_folds_of_cycles_run_only_when_the_user_asks():
    from dynalyst.orchestration.agent import ASKS_CYCLE_FOLDS
    for asked in ("show the folds of cycles", "fold of limit cycles", "where is the cycle fold",
                  "find LPC points", "saddle-node of limit cycles", "limit point of cycles"):
        assert ASKS_CYCLE_FOLDS.search(asked), asked
    for not_asked in ("Draw bifurcation diagram of Borghans model with beta_par from 0 to 2",
                      "show folds and limit cycles", "find the fold", "continue the limit cycles"):
        assert not ASKS_CYCLE_FOLDS.search(not_asked), not_asked


def test_failed_cycle_runs_do_not_contradict_the_followed_branch():
    # Borghans beta: the branch from HB 1.166 reaches 0.0936, next to HB 0.0834, whose own run
    # stalled; a second run from 1.166 broke down. The JSON must match the diagram.
    from dynalyst.analysis.bifurcation_auto import _settle_cycle_entries
    good = {"from_hopf_at": 1.16633, "cycles_over": [0.0936, 1.16633]}
    pd_fail = {"from_period_doubling_at": 1.15, "from_hopf_at": 1.16633, "error": "x"}
    out = _settle_cycle_entries([good, {"from_hopf_at": 1.16633, "error": "broke down"},
                                 {"from_hopf_at": 0.0833819, "error": "stalled"},
                                 {"from_hopf_at": 0.0833819, "error": "stalled"},
                                {"from_hopf_at": 0.0833819, "same_cycles_as": "the branch from 1.16633"}, pd_fail],
                                "beta_par", 0.02)
    assert out[0] is good and out[2] is pd_fail and len(out) == 3
    assert "error" not in out[1] and "1.16633" in out[1]["same_cycles_as"]
    lone = {"from_hopf_at": 0.5, "error": "stalled"}
    assert _settle_cycle_entries([lone], "p", 0.02) == [lone]


def test_the_parameter_question_follows_the_users_words():
    # the model may pick a parameter itself (after a screen): only the user's words count
    from dynalyst.orchestration.approval import needed_for_user, user_named_parameter
    m = "Borghans_BiophysChem1997.xml"
    assert not user_named_parameter("Draw bifurcation diagram of Borghans model", m)
    assert user_named_parameter("Draw bifurcation diagram of Borghans model in beta_par", m)
    assert user_named_parameter("bifurcation of borghans, vary K_PAR please", m)
    assert not user_named_parameter("plot Z_state of Borghans", m)          # a species, not a parameter
    call = {"model_name_or_path": m, "param_name": "K_par"}                 # filled in by the model
    assert needed_for_user("run_bifurcation_analysis", call, "Draw bifurcation diagram of Borghans model", "ask")
    assert not needed_for_user("run_bifurcation_analysis", call, "diagram of Borghans in K_par", "ask")
    assert not needed_for_user("run_bifurcation_analysis", call, "Draw bifurcation diagram of Borghans model", "auto")
    assert not needed_for_user("lhs_bifurcation_scan", {"model": m}, "LHS of Borghans", "ask")
    # another parameter in the request is not this one: still the model's choice
    assert needed_for_user("run_bifurcation_analysis", call,
                           "Diagram of Borghans with beta_par = 0.5", "ask")
    # an LHS scan along a parameter the model picked asks as well
    assert needed_for_user("lhs_bifurcation_scan", {"model": m, "bif_param": "K_par"},
                           "LHS of Borghans", "ask")
    assert not needed_for_user("lhs_bifurcation_scan", {"model": m, "bif_param": "K_par"},
                               "LHS of Borghans along K_par", "ask")
    # the user's choice of candidate replaces the model's pick, with the candidate's range
    from dynalyst.orchestration.approval import inject
    allowed = {"model_name_or_path", "param_name", "param_min", "param_max", "plot_state"}
    cand = {"parameter": "beta_par", "range": [0.1, 1.0], "y_species": "Z_state"}
    out = inject("run_bifurcation_analysis", {**call, "param_min": 5, "param_max": 50}, cand, allowed)
    assert (out["param_name"], out["param_min"], out["param_max"]) == ("beta_par", 0.1, 1.0)
    same = {**call, "param_name": "beta_par", "param_min": 0.2, "param_max": 0.4}
    assert inject("run_bifurcation_analysis", same, cand, allowed) == same


def test_the_y_axis_is_the_species_that_keeps_the_branches_apart():
    import numpy as np
    from dynalyst.analysis.bifurcation_auto import _monotone_pieces, _separating_species
    p = np.linspace(0, 1, 30)
    fold = np.concatenate([p[::-1], p[1:]])           # one branch that turns back at p = 0
    curve = lambda x, y: {"coordnames": ["k", "X", "Y"], "p_idx": 0,
                          "arr": np.array([fold, x, y])}
    # X is the same on both sides of the fold (a fold drawn as one line); Y separates them
    c = curve(np.full(fold.size, 2.0), np.concatenate([1 - p[::-1], 1 + p[1:]]))
    assert [len(i) for i in _monotone_pieces(fold)] == [30, 30]
    assert _separating_species([c], ["X", "Y"], (0.0, 1.0)) == "Y"
    single = {"coordnames": ["k", "X", "Y"], "p_idx": 0, "arr": np.array([p, p, 2 * p])}
    assert _separating_species([single], ["X", "Y"], (0.0, 1.0)) is None   # nothing coexists
    assert _separating_species([single, single], ["X", "Y"], (0.0, 1.0)) is None   # same curve twice


def test_only_sbml_species_are_kept_non_negative():
    from dynalyst.analysis.model_prep import prepare, sbml_species
    ids = sbml_species("data/sbml_models/BIOMD0000000507.xml")
    assert ids == {"species_1", "species_2", "species_3"}
    assert prepare("BIOMD0000000507")["nonneg"] <= ids      # never a parameter or a rule
    assert sbml_species(None) is None and sbml_species("no_such_file.xml") is None


def test_the_approval_line_names_no_species_for_a_diagram():
    from dynalyst.orchestration.approval import directive
    cand = {"parameter": "ATOT", "range": [0.05, 284.772], "y_species": "AP"}
    assert "AP." not in directive(cand, "run_bifurcation_analysis")
    assert "plotting AP" in directive(cand, "lhs_bifurcation_scan")      # the LHS scan keeps it


def test_coexistence_windows_come_from_the_stable_branches():
    import numpy as np
    from dynalyst.regime.taxonomy import coexistence_windows, max_coexisting_stable
    p = np.linspace(0, 10, 101)
    low = (p, 0.1 * p, np.where(p <= 6, 1, -1))           # stable up to 6
    high = (p[30:], 5 + p[30:], np.ones(p[30:].size))               # stable from 3 on
    w = coexistence_windows([low, high], within=(0, 10))
    assert len(w) == 1 and w[0]["stable_states"] == 2
    assert abs(w[0]["from"] - 3) < 0.1 and abs(w[0]["to"] - 6) < 0.1
    assert max_coexisting_stable([low, high], within=(0, 10))[0] == 2      # unchanged
    assert coexistence_windows([low], within=(0, 10)) == []


def test_one_state_followed_twice_is_not_two_states():
    import numpy as np
    from dynalyst.regime.taxonomy import coexistence_windows, max_coexisting_stable
    p1, p2 = np.linspace(0, 10, 101), np.linspace(0.05, 10, 77)        # interpolated at other points
    state = lambda p: np.vstack([0.8 - 0.03 * p, np.zeros_like(p), 0.3 + 0.02 * p]).T
    a, b = (p1, state(p1), np.ones(p1.size)), (p2, state(p2) * (1 + 1e-5), np.ones(p2.size))
    assert max_coexisting_stable([a, b], within=(0, 10), state_tol=1e-3)[0] == 1
    assert coexistence_windows([a, b], within=(0, 10), state_tol=1e-3) == []
    other = (p1, state(p1) + np.array([0, 0.5, 0]), np.ones(p1.size))    # same x, other y: two states
    assert max_coexisting_stable([a, other], within=(0, 10), state_tol=1e-3)[0] == 2
