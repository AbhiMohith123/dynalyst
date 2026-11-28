"""Smoke tests: every tool the agents can call, invoked the way the agents
invoke it (LangChain .invoke for the LHS / AUTO tools, the research agent's
_execute_tool dispatch for research tools). Small sample counts -- these
check the merged wiring works end to end, not the science."""
import io
import json
import contextlib

import pytest

from dynalyst import config

SMOKE_MODELS = ["smoke_xpp_model", "smoke_odes_model"]


@pytest.fixture(autouse=True, scope="module")
def _cleanup_generated_sbml():
    yield
    for stem in SMOKE_MODELS:
        (config.LOCAL_SBML_MODELS_DIR / f"{stem}.xml").unlink(missing_ok=True)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# --- LHS / AUTO tools (need AUTO) ------------------------------------------------

@pytest.mark.auto
@pytest.mark.parametrize("category,variant", [
    ("saddle-node", "generic"), ("transcritical", "generic"), ("pitchfork", "generic"),
    ("hopf", "generic"), ("subcritical hopf", "generic"),
])
def test_generic_bifurcation_model(category, variant):
    from dynalyst.regime.generic_tool import generic_bifurcation_model
    out = _quiet(generic_bifurcation_model.invoke, {"category": category, "variant": variant})
    assert json.loads(out)["bifurcation_points"], out


@pytest.mark.auto
@pytest.mark.parametrize("category,expected", [
    ("transcritical", "BP"),
    ("pitchfork", "BP"),
    ("saddle-node", "LP"),
])
def test_generic_bifurcation_model_pdp(category, expected):
    from dynalyst.regime.generic_tool import generic_bifurcation_model
    out = _quiet(generic_bifurcation_model.invoke, {"category": category, "variant": "pdp"})
    assert expected in [p["type"] for p in json.loads(out)["bifurcation_points"]]


@pytest.mark.auto
def test_pdp_saddle_node_via_research_pipeline():
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    res = json.loads(run_bifurcation_analysis(str(config.MODELS_DIR / "saddle_node.xml"),
                                              param_name="alpha", param_min=0.1, param_max=5,
                                              show_auto_log=False))
    assert [(p["type"], round(p["alpha"], 3)) for p in res["bifurcation_points"]] == [("LP", 2.0)]


def test_generic_bifurcation_model_unsupported_category_raises():
    from dynalyst.regime.generic_models import build_model_file
    with pytest.raises(ValueError):
        build_model_file("bautin", "generic", "/tmp")


XPP_SCAN = {"model": "pdp_mm_bistable.txt", "bif_param": "ATOT", "y_species": "APP", "n_samples": 3,
            "vary_only": "km5", "manual_bounds": '{"km5": [4.0, 6.0]}', "seed": 0}


@pytest.mark.auto
def test_lhs_scan_of_an_xpp_model():
    from dynalyst.regime.lhs_tool import lhs_bifurcation_scan
    out = _quiet(lhs_bifurcation_scan.invoke, XPP_SCAN)
    assert "LHS batch scan complete" in out and "Continuation window:" in out


@pytest.mark.auto
def test_categorized_scan():
    from dynalyst.regime import plot_registry
    from dynalyst.regime.categorize_tool import categorized_lhs_scan
    out = _quiet(categorized_lhs_scan.invoke, XPP_SCAN)
    assert "Category counts:" in out and ".xlsx" in out
    assert plot_registry.pop_all()  # interactive figures were pushed


# --- research tools ----------------------------------------------------------

def _research(name, **kwargs):
    from dynalyst.agents.research_agent import _execute_tool
    return json.loads(_quiet(_execute_tool, name, kwargs))


def test_list_available_models_includes_merged_model_dirs():
    res = _research("list_available_models")
    dirs = list(res["sbml_models_by_directory"])
    assert str(config.MODELS_DIR) in dirs
    assert any("Garde2020" in f for fs in res["sbml_models_by_directory"].values() for f in fs)


def test_odes_and_steady_states():
    # SBML model from data/sbml_models (XPP-derived full Srirammark model)
    res = _research("get_odes", model_name_or_path="srirammarkfullALL")
    assert res["n_state_variables"] >= 2
    res = _research("find_steady_states", model_name_or_path="srirammarkfullALL")
    # this model has a continuum of steady states: a sample of it, no isolated state to count
    assert res["n_distinct_steady_states"] >= 1 or res["continuum"]


def test_research_sbml_reader_reads_assignment_rules():
    """The generic SBML reader used to reject assignment rules, which every
    A/APP file uses; it substitutes them now, and its equations are the regime
    parser's (checked numerically)."""
    import numpy as np
    import sympy as sp
    from dynalyst.analysis.ode_model import model_from_sbml
    from dynalyst.regime.parser import parse_sbml
    res = _research("get_odes", model_name_or_path="pitchfork_model.xml")
    assert "error" not in res and res["odes"] == {"dA/dt": "-r1 + r3", "dAPP/dt": "r2 - r4"}
    path = str(config.MODELS_DIR / "pitchfork_model.xml")
    g, r = model_from_sbml(path), parse_sbml(path)
    values = {sp.Symbol(k): v for k, v in r["param_values"].items()}
    for x in np.random.default_rng(0).uniform(0.001, 0.1, (5, 2)):
        point = {sp.Symbol("A"): x[0], sp.Symbol("APP"): x[1], **values}
        for s, e in zip(g["state_names"], g["rhs"]):
            assert float(sp.sympify(e).xreplace(point)) == \
                pytest.approx(float(sp.sympify(r["ode_rhs"][s]).xreplace(point)), rel=1e-12)


def test_jaxkineticmodel_simulation_and_models_jaxkm_cannot_load():
    res = _research("simulate_model", model_name_or_path="Garde2020", t_end=5.0, n_points=20)
    assert "error" not in res, res
    assert "_simulation_" in res["plot_path"] and res["plot_path"].endswith(".png")
    res = _research("compare_initial_conditions", model_name_or_path="p_dynamics",
                    initial_conditions_list=[{"p": 0.1}, {"p": 2.0}], t_end=5.0)
    assert res["n_conditions"] == 2


def test_xpp_and_ode_conversion_to_sbml():
    res = _research("xpp_to_sbml", path=str(config.MODELS_DIR / "pdp_mm_bistable.txt"),
                    model_name="smoke_xpp_model")
    assert "error" not in res, res
    res = _research("odes_to_sbml", species=["x"], odes={"x": "mu + x**2"},
                    parameters={"mu": -1.0}, initial_conditions={"x": -1.0},
                    model_name="smoke_odes_model")
    assert "error" not in res, res


def test_pdf_tools():
    from tests.conftest import need_pdf
    pytest.importorskip("pymupdf")
    need_pdf("Markevich.pdf")
    res = _research("list_available_pdfs")
    assert any(p.endswith("Markevich.pdf") for p in res["pdfs"])
    res = _research("read_pdf", path_or_name="Markevich", page_range=[1, 1])
    assert res["pages"] and res["pages"][0]["text"]


def test_retrieve_knowledge_without_index_returns_error_json(tmp_path, monkeypatch):
    # point the index at an empty temp dir (chromadb creates its directory on
    # first use, so never let this test touch the real chroma_db/)
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(config, "BM25_INDEX_PATH", tmp_path / "chroma_db" / "bm25_index.pkl")
    from dynalyst.rag import retrieve
    for cached in (retrieve._get_chroma_client, retrieve._get_bm25):
        cached.cache_clear()
    res = _research("retrieve_knowledge", query="what solver does jaxkineticmodel use")
    assert "error" in res and "retrieve_knowledge failed" in res["error"]
    for cached in (retrieve._get_chroma_client, retrieve._get_bm25):
        cached.cache_clear()


@pytest.mark.auto
def test_bifurcation_analysis_screens_for_a_parameter_when_none_is_named():
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    d = json.loads(run_bifurcation_analysis("pitchfork_model.xml", show_auto_log=False, _cycles=False))
    assert d["automatic_parameter_search"]["selected_parameter"] == "ATOT"
    assert "BP" in [p["type"] for p in d["bifurcation_points"]]


@pytest.mark.auto
def test_relative_fixed_params_in_a_bifurcation_run():
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    d = json.loads(run_bifurcation_analysis("Markevich2004", param_name="MAPKK",
                                            fixed_params={"MKP3": "*2"}, _cycles=False))
    folds = sorted(round(p["MAPKK"], 1) for p in d["bifurcation_points"] if p["type"] == "LP")
    assert folds == pytest.approx([78.5, 114.8], abs=0.2)
