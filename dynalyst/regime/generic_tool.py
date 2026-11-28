"""
LangChain tool wrapper around generic_models.py, so a plain query like "plot
the generic model of saddle node" or "run the PDP pitchfork model" runs a
reference model for a named bifurcation type. Defaults to the textbook
normal-form model; variant='pdp' gives the phosphorylation-dephosphorylation
reference model instead.
"""
import json
import os
import uuid

from langchain.tools import tool

from dynalyst import config
from dynalyst.regime.generic_models import build_model_file


@tool
def generic_bifurcation_model(category: str, variant: str = 'generic') -> str:
    """Use when the user asks for a textbook example of a bifurcation type and names no model
    file: this tool builds the reference model itself and runs its bifurcation analysis.

    Args:
        category: which bifurcation type -- loosely matched, e.g.
            'saddle-node'/'fold', 'transcritical'/'TC', 'pitchfork'/'PF',
            'Hopf'/'HB' (supercritical), 'subcritical Hopf'. 'Generalized
            Hopf'/'Bautin' and 'limit-point cycle' are recognized, with a clear
            explanation that no reference model exists for them.
        variant: 'generic' (default) uses the bare textbook normal-form
            model (e.g. dx/dt = mu*x - x^3 for pitchfork). 'pdp' uses the
            phosphorylation-dephosphorylation biological reference model
            instead (only available for saddle_node/transcritical/pitchfork,
            not for the Hopf categories).
    """
    from dynalyst.analysis.bifurcation_auto import run_bifurcation_analysis
    workdir = os.path.abspath(os.path.join(str(config.WORKDIR), uuid.uuid4().hex[:8]))
    os.makedirs(workdir, exist_ok=True)

    key, sbml_path, spec = build_model_file(category, variant, workdir)
    lo, hi = spec['param_range']
    result = json.loads(run_bifurcation_analysis(
        model_name_or_path=sbml_path, param_name=spec['bif_param'], param_min=lo, param_max=hi,
        plot_state=spec['y_species']))
    return json.dumps({"category": key, "variant": variant, "model_file": sbml_path, **result},
                      indent=2, default=float)
