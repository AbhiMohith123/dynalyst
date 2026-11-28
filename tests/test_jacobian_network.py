"""Jacobian at the steady states -> signed interaction network."""
import json

from dynalyst.analysis.jacobian_network import jacobian_network

TOGGLE = dict(species=["u", "v"], odes={"u": "a/(1+v**2) - u", "v": "a/(1+u**2) - v"},
              parameters={"a": 3})


def test_toggle_switch_mutual_inhibition_is_a_positive_loop():
    r = json.loads(jacobian_network(**TOGGLE))
    assert r["n_steady_states"] == 3
    assert sorted(s["stability"] for s in r["steady_states"]) == ["stable", "stable", "unstable"]
    s = r["steady_states"][0]
    assert {(e["source"], e["target"], e["sign"]) for e in s["edges"]} == {("u", "v", "-"), ("v", "u", "-")}
    assert s["loops"]["n_positive"] == 1 and s["loops"]["n_negative"] == 2      # u -> u, v -> v: removal
    assert s["self_regulation"] == {"u": -1.0, "v": -1.0}
    # J[i][j] = d f_i / d x_j : row u, column v
    assert s["jacobian"][0][1] == next(e["J"] for e in s["edges"] if e["source"] == "v")


def test_a_given_state_is_used_and_checked():
    r = json.loads(jacobian_network(**TOGGLE, state={"u": 1.0, "v": 1.0}))
    assert r["n_steady_states"] == 1 and "warning" in r["steady_states"][0]


def test_parameter_jacobian_and_sensitivities():
    r = json.loads(jacobian_network(**TOGGLE, which="stable"))
    pj = r["steady_states"][0]["parameter_jacobian"]
    assert pj["parameters"] == ["a"] and pj["sensitivities_dxdp_top"]


def test_ferrell_oscillator_has_a_negative_loop():
    r = json.loads(jacobian_network(model_name_or_path="ferrell_cell_cycle_3d.ode"))
    loops = r["steady_states"][0]["loops"]["list"]
    assert any(l["sign"] == "negative" and l["length"] == 3 for l in loops)


def test_mapk_sn_has_two_stable_states_and_files():
    r = json.loads(jacobian_network(model_name_or_path="Full_model_for_SN.txt",
                                    parameters={"ATOT": 430}))
    stab = [s["stability"] for s in r["steady_states"]]
    assert stab.count("stable") == 2 and stab.count("unstable") == 1
    assert r["excel"] and r["steady_states"][0]["network_plot"].endswith(".png")


def test_the_steady_state_agent_owns_the_tool():
    from dynalyst.orchestration.subagents import owner_of
    assert owner_of("jacobian_network") == "steady_state"
