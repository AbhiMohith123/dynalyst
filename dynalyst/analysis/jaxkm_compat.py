"""
jaxkineticmodel-ready copies of SBML files that use assignment rules.

jaxkineticmodel's SBML importer loads rate-rule models whose fluxes are
defined by assignment rules (fluxes assigned, species as rate rules), but it
treats the assigned fluxes as constant parameters
(value 0) -- so the model silently simulates as a flat line. Converting the
model to jaxkineticmodel's own representation (the pipeline's SBML parser
substitutes the assignment rules; jaxkm_ode_to_sbml builds a NeuralODEBuild
and exports it with jaxkineticmodel's SBMLExporter) gives a file it
simulates correctly, with the same species and parameter names.

Copies are cached in outputs/converted/ under the original file name, so
model labels in plots and results stay the same.
"""
import contextlib
import io
from pathlib import Path

import jax.numpy as jnp
import libsbml
import sympy as sp

from dynalyst import config


def has_assignment_rules(sbml_path) -> bool:
    """Whether the file has rules jaxkineticmodel does not simulate as written: assignment
    rules (it takes the assigned values as constants) and rate rules (it keeps the
    variable constant). Either way it simulates a flat line without saying so."""
    model = libsbml.readSBMLFromFile(str(sbml_path)).getModel()
    return model is not None and any(
        r.getTypeCode() in (libsbml.SBML_ASSIGNMENT_RULE, libsbml.SBML_RATE_RULE)
        for r in model.getListOfRules())


def _converter_key(sbml_path: Path) -> str:
    """What a converted copy was made from: the source file and the code that converts it.
    A copy made by an earlier version of the converter (e.g. before a fix in how SBML is
    read) is not reused."""
    import hashlib
    from dynalyst.analysis import generic_sbml, jaxkm_ode_to_sbml, ode_model
    h = hashlib.sha1(Path(sbml_path).read_bytes())
    for mod in (generic_sbml, ode_model, jaxkm_ode_to_sbml):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def jaxkm_ready_sbml(sbml_path) -> Path:
    """The file itself if jaxkineticmodel handles it, else a converted copy."""
    sbml_path = Path(sbml_path)
    if not has_assignment_rules(sbml_path):
        return sbml_path
    out_dir = config.OUTPUT_DIR / "converted"
    out_path = out_dir / sbml_path.name
    key_path = out_path.with_suffix(".key")
    key = _converter_key(sbml_path)
    if out_path.exists() and key_path.exists() and key_path.read_text().strip() == key:
        return out_path

    from jaxkineticmodel.load_sbml.export_sbml import SBMLExporter
    from dynalyst.analysis.jaxkm_ode_to_sbml import _build_jkm_model
    from dynalyst.regime.parser import parse_sbml

    parsed = parse_sbml(str(sbml_path))
    species = list(parsed["species"])
    odes = {s: sp.sstr(parsed["ode_rhs"][s]) for s in species}
    with contextlib.redirect_stdout(io.StringIO()):
        jkm_model = _build_jkm_model(species, odes, compartment_size=1.0)
        y0 = jnp.array([float(parsed["initial_guess"].get(s, 0.0)) for s in jkm_model.species_names])
        params = {p: float(parsed["param_values"].get(p, 1.0)) for p in jkm_model.parameter_names}
        out_dir.mkdir(parents=True, exist_ok=True)
        SBMLExporter(jkm_model).export(initial_conditions=y0, parameters=params,
                                       output_file=str(out_path))
    key_path.write_text(key)     # written last: a failed export leaves no valid key
    return out_path
