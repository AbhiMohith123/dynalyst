"""save_sbml: write a changed copy of a model as a new SBML file.

Changes: parameter values (global, or reaction-local by name / '<reaction>_<name>'),
starting values of species, or every parameter of one LHS sample (the scan's
summary CSV + sample id). The copy goes to data/sbml_models/<name>.xml, where
every tool finds it by name, and into the knowledge base (registry hook). The
original's BioModels identity becomes bqmodel:isDerivedFrom, so the copy is
never labelled as the published model.
"""
import csv
import json
import re
from pathlib import Path

import libsbml

from dynalyst import config
from dynalyst.analysis.generic_sbml import resolve_sbml_path


def _sample_values(scan_csv: str, sample_id) -> dict:
    path = Path(scan_csv)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if str(r["sample_id"]) == str(sample_id)), None)
    if row is None:
        raise ValueError(f"No sample '{sample_id}' in {path.name} "
                         f"(ids: {[r['sample_id'] for r in rows][:30]}).")
    meta = {"sample_id", "category", "features", "signature", "error"}
    return {k: float(v) for k, v in row.items() if k not in meta and v not in ("", None)}


def _set_parameter(model, name, value):
    """Set a parameter (or species / compartment) value; returns where it was set."""
    p = model.getParameter(name)
    if p is not None:
        p.setValue(value)
        return f"parameter {name}"
    matches = []
    for i in range(model.getNumReactions()):
        r = model.getReaction(i)
        kl = r.getKineticLaw()
        if kl is None:
            continue
        for j in range(kl.getNumLocalParameters()):
            lp = kl.getLocalParameter(j)
            if lp.getId() == name or f"{r.getId()}_{lp.getId()}" == name:
                matches.append((r.getId(), lp))
        for j in range(kl.getNumParameters()):
            lp = kl.getParameter(j)
            if lp.getId() == name or f"{r.getId()}_{lp.getId()}" == name:
                matches.append((r.getId(), lp))
    # libSBML can expose L3 local parameters through both compatibility APIs.
    matches = list({(rid, lp.getId()): (rid, lp) for rid, lp in matches}.values())
    if len(matches) > 1:
        choices = [f"{r}_{lp.getId()}" for r, lp in matches]
        raise ValueError(f"Ambiguous local parameter '{name}'; use one of {choices}.")
    if matches:
        reaction, lp = matches[0]
        lp.setValue(value)
        return f"local parameter {lp.getId()} of reaction {reaction}"
    c = model.getCompartment(name)
    if c is not None:
        c.setSize(value)
        return f"compartment {name}"
    return None


def _set_species(model, name, value):
    s = model.getSpecies(name)
    if s is None:
        return None
    if s.isSetInitialAmount() and not s.isSetInitialConcentration():
        s.setInitialAmount(value)
    else:
        s.setInitialConcentration(value)
    return f"species {name}"


def _mark_derived(model):
    """bqmodel:is -> bqmodel:isDerivedFrom: the copy is not the published model."""
    for i in range(model.getNumCVTerms()):
        cv = model.getCVTerm(i)
        if cv.getQualifierType() == libsbml.MODEL_QUALIFIER and \
                cv.getModelQualifierType() == libsbml.BQM_IS:
            cv.setModelQualifierType(libsbml.BQM_IS_DERIVED_FROM)


def save_sbml(model_name_or_path: str, name: str, parameters: dict | None = None,
              initial_conditions: dict | None = None, scan_csv: str | None = None,
              sample_id: str | None = None, overwrite: bool = False) -> str:
    """Tool: save a changed copy of an SBML model as data/sbml_models/<name>.xml."""
    from dynalyst.analysis.ode_model import xpp_file
    xpp = xpp_file(model_name_or_path)
    if xpp is not None:
        return json.dumps({"error": f"{xpp.name} is an XPP file, not SBML; convert it first "
                                    f"(xpp_to_sbml), then save a changed copy of the SBML."})
    try:
        src = resolve_sbml_path(str(model_name_or_path))
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    if src.suffix.lower() != ".xml":
        return json.dumps({"error": f"{src.name} is not SBML; convert it first (xpp_to_sbml)."})
    # a name given with its extension ('copy.xml') is the same name: never 'copy.xml.xml'
    safe = re.sub(r"[^\w.-]+", "_", re.sub(r"\.(xml|sbml)$", "", name.strip(), flags=re.I)).strip("._")
    if not safe:
        return json.dumps({"error": "Give the new model a name."})
    out = config.LOCAL_SBML_MODELS_DIR / f"{safe}.xml"
    if out.exists() and not overwrite:
        return json.dumps({"error": f"{out.name} already exists and is never replaced; choose "
                                    "another name."})
    if out.resolve() == src.resolve():
        return json.dumps({"error": "The new name equals the original file; choose another."})

    values = {}
    try:
        if scan_csv is not None or sample_id is not None:
            if scan_csv is None or sample_id is None:
                return json.dumps({"error": "scan_csv and sample_id go together (the scan's "
                                            "'Summary CSV' path and the sample's id)."})
            values.update(_sample_values(scan_csv, sample_id))
    except (OSError, ValueError, KeyError) as e:
        return json.dumps({"error": f"Could not read the sample: {e}"})
    values.update({k: float(v) for k, v in (parameters or {}).items()})

    doc = libsbml.readSBMLFromFile(str(src))
    model = doc.getModel()
    if model is None:
        return json.dumps({"error": f"{src.name} is not readable SBML."})
    changed, unknown, overridden = {}, [], []
    assigned = {model.getInitialAssignment(i).getSymbol()
                for i in range(model.getNumInitialAssignments())}
    ruled = {model.getRule(i).getVariable() for i in range(model.getNumRules())}
    for k, v in values.items():
        try:
            where = _set_parameter(model, k, v) or _set_species(model, k, v)
        except ValueError as e:
            return json.dumps({"error": f"{e} Nothing was saved."})
        if where is None:
            unknown.append(k)
            continue
        changed[k] = {"value": v, "set_as": where}
        if k in assigned or k in ruled:
            overridden.append(k)
    for k, v in (initial_conditions or {}).items():
        where = _set_species(model, k, float(v))
        if where is None:
            unknown.append(k)
        else:
            changed[k] = {"value": float(v), "set_as": where + " (start value)"}
    if unknown:
        return json.dumps({"error": f"Not in {src.name}: {unknown}. Nothing was saved.",
                           "parameters": [model.getParameter(i).getId()
                                          for i in range(model.getNumParameters())][:80],
                           "species": [model.getSpecies(i).getId()
                                       for i in range(model.getNumSpecies())]})
    _mark_derived(model)
    model.setId(re.sub(r"\W", "_", safe) if re.match(r"[A-Za-z_]", safe) else f"m_{safe}")
    model.setName(f"{model.getName() or src.stem} ({'modified' if changed else 'copy'}: {safe})")
    config.LOCAL_SBML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not libsbml.writeSBMLToFile(doc, str(out)):
        return json.dumps({"error": f"Could not write {out}."})
    res = {"saved_as": config.display_path(out), "path": str(out), "use_as": out.stem,
           "derived_from": src.name, "changed": changed or "nothing: a plain copy under a new name",
           **({"sample": {"scan_csv": scan_csv, "sample_id": str(sample_id)}} if scan_csv else {}),
           "note": f"Saved; every tool now finds it as '{out.stem}'. Its BioModels identity is "
                   f"recorded as derived from {src.name}'s, not as the published model."}
    if overridden:
        res["warning"] = (f"{overridden} are also set by an initial assignment / rule in the "
                          "model, which may override the saved value when the model is loaded.")
    return json.dumps(res, indent=2)


TOOL_DEFINITIONS = [
    {"name": "save_sbml",
     "description": "Save a copy of an SBML model under a new name (data/sbml_models/<name>.xml) "
                    "that every tool can use, unchanged or with new parameter values, new "
                    "starting values of species, or every parameter value of one LHS sample "
                    "(scan_csv = the scan's 'Summary CSV' path, sample_id = its id). The "
                    "original file is never changed.",
     "input_schema": {"type": "object", "properties": {
         "model_name_or_path": {"type": "string"},
         "name": {"type": "string", "description": "Name of the new model file."},
         "parameters": {"type": "object", "description": "{parameter: value}"},
         "initial_conditions": {"type": "object", "description": "{species: start value}"},
         "scan_csv": {"type": "string", "description": "An LHS scan's summary CSV path."},
         "sample_id": {"type": "string", "description": "The sample's id in that CSV."}},
         "required": ["model_name_or_path", "name"]}},
]
TOOL_FUNCTIONS = {"save_sbml": save_sbml}
