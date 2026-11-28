"""dynalyst -- merged bifurcation-analysis agent.

Subpackages:
  regime    the LHS / AUTO-07p scan pipeline: SBML -> AUTO-07p -> diagrams, LHS
            scans, categorization, generic/PDP reference models
            (formerly agent/; the package keeps its old name)
  analysis  generic any-dimension ODE/SBML/XPP analysis, AUTO-07p
            bifurcation analysis, jaxkineticmodel simulation, PDF reading
            (formerly rag_jaxkm_bifurcation_workflow/)
  rag       knowledge-base ingestion, indexing and hybrid retrieval
  agents    research_agent, unified_agent

Nothing heavy is imported here; import the submodule you need.
"""
import logging

__version__ = "1.0.1"


def _quiet_library_logging():
    """jaxkineticmodel's logging config (applied on its first import) sets
    the jax and matplotlib loggers to INFO, which prints backend-probing
    noise ("Unable to initialize backend 'tpu'") into every agent run.
    Apply its config now, then lower those two loggers to WARNING."""
    try:
        from jaxkineticmodel import utils
        utils.get_logger(__name__)
    except Exception:
        return
    for name in ("jax", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


_quiet_library_logging()

# a 0/0 inside sympy-generated code (a Jacobian at a state with zeros) is handled where it is
# used; its RuntimeWarning would print the whole generated expression into the answer
import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"<lambdifygenerated")

# every PNG a tool saves also gets an interactive Plotly twin (<name>.html, <name>.plotly.json);
# DYNALYST_INTERACTIVE=0 switches it off
try:
    from dynalyst.interactive_plots import install_savefig_hook  # noqa: E402
    install_savefig_hook()
except Exception:       # interactive twins are an extra: never block the package
    pass
