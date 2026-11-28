"""Forgiving argument forms for range requests, and the zero-width allowed-range fallback."""
import pytest


@pytest.mark.parametrize("text, window", [
    ("x3", {"fold": 3.0}), ("3x", {"fold": 3.0}), ("3-fold", {"fold": 3.0}),
    ("factor of 3", {"fold": 3.0}), ("+-10%", {"rel": 0.1}), ("±10%", {"rel": 0.1}),
    ("10 percent", {"rel": 0.1}), ("10 to 30", {"abs": [10.0, 30.0]}),
    ([10, 30], {"abs": [10.0, 30.0]}), ({"percent": 10}, {"rel": 0.1}), ({"factor": 3}, {"fold": 3}),
])
def test_param_windows_in_any_form(text, window):
    from dynalyst.regime.range_spec import window_spec
    assert window_spec("p", text) == window


def test_fixed_and_baseline_values_in_plain_form():
    from dynalyst.regime.range_spec import extras_from_tool_args
    x = extras_from_tool_args("k4=12, km7: 0.2", "km1, km3", None)
    assert x.baseline == {"k4": 12.0, "km7": 0.2} and x.fixed == {"km1": None, "km3": None}
    assert extras_from_tool_args(None, "Km5=100", None).fixed == {"Km5": 100.0}
    with pytest.raises(ValueError, match="window"):
        extras_from_tool_args(None, None, '{"a": "around three"}')


def test_a_zero_width_reference_range_falls_back_to_the_baseline():
    from dynalyst.regime.range_spec import RangeSpec, resolve_bounds
    r = resolve_bounds({"k2": 0.01}, RangeSpec(rel=0.01, rel_reference={"k2": (0.01, 0.01)}))
    assert r.bounds["k2"][:2] == pytest.approx((0.0099, 0.0101))
    assert "single value" in r.notes["k2"]
    held = resolve_bounds({"a": 1.0}, RangeSpec(rel=0.01, allowed={"a": (1.0, 1.0)}))
    assert held.bounds["a"][:2] == (1.0, 1.0)     # an explicit zero-width range is respected
