"""The LHS tools on any model: +-rel_window bounds, the generic model loader
(conservation laws eliminated), the parameter guard, and the tool plumbing with
AUTO stubbed out. One real-AUTO run is marked `auto`."""
import pytest

from dynalyst.regime import model_source
from dynalyst.regime.lhs_sampling import resolve_model_bounds
from dynalyst.regime.range_spec import rel_window_bounds


def test_rel_window_is_a_fraction_of_the_allowed_range():
    assert rel_window_bounds(5.0, 0.01, (0.0, 100.0)) == (4.0, 6.0, False)
    assert rel_window_bounds(10.0, 0.01) == (9.9, 10.1, False)          # of the baseline
    lo, hi, _ = rel_window_bounds(0.2, 0.01, (0.2, 100.0))              # never leaves the range
    assert lo == 0.2 and hi == pytest.approx(0.2 + 0.998)
    assert rel_window_bounds(500.0, 0.01, (0.0, 100.0)) == (100.0, 100.0, False)  # pinned


def test_generic_model_loader_eliminates_conservation_laws():
    src = model_source.load_model("Markevich2004")      # M + Mp + Mpp conserved
    assert src["parsed"]["species"] == ["M", "Mp"] and src["eliminated"] == ["Mpp"]
    assert {"k1cat", "k2cat", "Km1"} <= set(src["parsed"]["params"])
    bif, y = model_source.resolve_target("Markevich2004", "k2cat")
    assert (bif, y) == ("k2cat", "M")       # the shared default: the first state
    lo, hi = model_source.default_window(src["parsed"], "k2cat")
    assert lo == pytest.approx(src["parsed"]["param_values"]["k2cat"] / 10)
    assert model_source.resolve_target("Markevich2004", "k2cat", "Mpp") == ("k2cat", "Mpp")
    observed = model_source.load_model("Markevich2004", keep="Mpp")
    assert "Mpp" in observed["parsed"]["species"] and len(observed["parsed"]["species"]) == 2
    with pytest.raises(ValueError, match="not a parameter"):
        model_source.resolve_target("Markevich2004", "ATOT")


def test_generic_model_bounds():
    parsed = model_source.load_model("Markevich2004")["parsed"]
    b = resolve_model_bounds(parsed, exclude_params=["k2cat"], vary_only="k1cat").bounds
    assert "k2cat" not in b and b["k1cat"][0] < b["k1cat"][1] and b["Km1"][0] == b["Km1"][1]
    b = resolve_model_bounds(parsed, rel_window=0.01, allowed_bounds={"Km1": (0.0, 1000.0)}).bounds
    v = parsed["param_values"]["Km1"]
    assert b["Km1"][:2] == pytest.approx((v - 10, v + 10))             # 1% of a 1000-wide range
    assert b["Km2"][1] == pytest.approx(parsed["param_values"]["Km2"] * 1.01)   # 1% of baseline
    with pytest.raises(ValueError, match="vary_only"):
        resolve_model_bounds(parsed, vary_only="nope")


def test_a_scan_needs_a_model():
    for missing in (None, ""):
        with pytest.raises(ValueError, match="model is required"):
            model_source.resolve_target(missing, "k2cat")


def test_lhs_tool_on_another_model_reports_every_sampled_set(stubbed_auto):
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    out = lhs_bifurcation_scan.func(n_samples=5, bif_param="k2cat", param_min=1.5, param_max=150.0,
                                    model="Markevich2004", rel_window=0.01, seed=1)
    assert {k: stubbed_auto[k] for k in ("bif", "rng", "n")} == {
        "bif": "k2cat", "rng": (1.5, 150.0), "n": 5}
    assert "Sampled parameter sets (5" in out and "within +-1%" in out
    rows = [ln for ln in out.splitlines() if ln.startswith("   ") and "no_bifurcation" in ln]
    assert len(rows) == 5
    assert "k2cat" not in out.split("Sampled parameter sets")[1].split("\n")[1]   # not sampled


TOOLS_WITH_REQUIRED_MODEL = ["lhs_bifurcation_scan", "categorized_lhs_scan"]


@pytest.mark.parametrize("name", TOOLS_WITH_REQUIRED_MODEL)
def test_model_is_a_required_argument(name):
    from dynalyst.orchestration import tools as registry
    schema = registry.schema_of(registry.all_tools()[name])
    assert "model" in schema["required"]
    from dynalyst.orchestration.guard import ExecutionGuard
    d = ExecutionGuard().check("bifurcation", name, {"n_samples": 3})
    assert not d.allowed and "missing required" in d.reason and "model" in d.reason


def test_an_xpp_file_is_an_ordinary_model():
    """An XPP file with several parameter blocks is read like any other model: its
    active `par` lines, its conservation law already written into the equations."""
    src = model_source.load_model("pdp_mm_bistable.txt")
    assert src["parsed"]["species"] == ["A", "APP"]
    assert "ATOT" in src["parsed"]["params"]
    assert model_source.resolve_target("pdp_mm_bistable.txt", "ATOT", "APP") == ("ATOT", "APP")


def test_tool_argument_errors():
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    with pytest.raises(ValueError, match="not a parameter"):
        lhs_bifurcation_scan.func(model="pdp_mm_bistable.txt", n_samples=2, bif_param="k2cat")


@pytest.mark.auto
def test_real_lhs_scan_of_another_model():
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    out = lhs_bifurcation_scan.func(n_samples=2, bif_param="k2cat", model="Markevich2004",
                                    rel_window=0.01, seed=0)
    assert "LHS batch scan complete" in out and "Sampled parameter sets (2" in out
    assert "'FAILED': 2" not in out, out      # at least one sample must have converged
