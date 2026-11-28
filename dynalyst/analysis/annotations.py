"""
Get annotations: the metadata an SBML file carries about itself and its
components -- model name, notes, authors/dates, SBO terms, and MIRIAM
cross-references (UniProt, ChEBI, GO, KEGG, BioModels, PubMed, ...).

Reads the file directly with libsbml; nothing is looked up online.
"""
import json
import re

import libsbml

from dynalyst.analysis.generic_sbml import resolve_sbml_path

_COMPONENT_LISTS = {
    "compartments": "getListOfCompartments",
    "species": "getListOfSpecies",
    "reactions": "getListOfReactions",
    "parameters": "getListOfParameters",
}


def _notes_text(sbase) -> str | None:
    if not sbase.isSetNotes():
        return None
    text = re.sub(r"<[^>]+>", " ", sbase.getNotesString())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _cv_terms(sbase) -> list[dict]:
    terms = []
    for i in range(sbase.getNumCVTerms()):
        cv = sbase.getCVTerm(i)
        if cv.getQualifierType() == libsbml.BIOLOGICAL_QUALIFIER:
            qualifier = "bqbiol:" + libsbml.BiolQualifierType_toString(cv.getBiologicalQualifierType())
        else:
            qualifier = "bqmodel:" + libsbml.ModelQualifierType_toString(cv.getModelQualifierType())
        resources = [cv.getResourceURI(j) for j in range(cv.getNumResources())]
        terms.append({"qualifier": qualifier, "resources": resources})
    return terms


def _element(sbase) -> dict:
    entry = {"id": sbase.getId(), "name": sbase.getName() or None}
    if sbase.isSetSBOTerm():
        entry["sbo_term"] = sbase.getSBOTermID()
    terms = _cv_terms(sbase)
    if terms:
        entry["cross_references"] = terms
    notes = _notes_text(sbase)
    if notes:
        entry["notes"] = notes[:500]
    return entry


def _history(model) -> dict | None:
    if not model.isSetModelHistory():
        return None
    h = model.getModelHistory()
    creators = []
    for i in range(h.getNumCreators()):
        c = h.getCreator(i)
        creators.append({k: v for k, v in {
            "name": " ".join(x for x in (c.getGivenName(), c.getFamilyName()) if x) or None,
            "email": c.getEmail() or None, "organisation": c.getOrganisation() or None,
        }.items() if v})
    out = {"creators": creators}
    if h.isSetCreatedDate():
        out["created"] = h.getCreatedDate().getDateAsString()
    modified = [h.getModifiedDate(i).getDateAsString() for i in range(h.getNumModifiedDates())]
    if modified:
        out["modified"] = modified
    return out


BIOMODELS_ID = re.compile(r"\b(BIOMD\d{10}|MODEL\d{8,12})\b")
_IDENTITY_CACHE = {}


def model_identity(model_name_or_path) -> dict | None:
    """Who a model file is: {'file', 'model_name', 'biomodels_id'}, read from the
    file itself -- the BioModels ID it declares with bqmodel:is (a curated BIOMD
    ID first, else a MODEL ID), else the ID it was saved under by
    fetch_biomodels_model (<BIOMD ID>.xml). biomodels_id is None when the
    file names none: then no BioModels ID may be attached to it. None for
    anything that is not an SBML file."""
    try:
        path = resolve_sbml_path(str(model_name_or_path))
    except (FileNotFoundError, ValueError):
        return None
    if path.suffix.lower() != ".xml":
        return None
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _IDENTITY_CACHE:
        model = libsbml.readSBMLFromFile(str(path)).getModel()
        ids = []
        if model is not None:
            for term in _cv_terms(model):
                if term["qualifier"] == "bqmodel:is":
                    ids += [m.group(1) for r in term["resources"] for m in BIOMODELS_ID.finditer(r)]
        from_name = BIOMODELS_ID.fullmatch(path.stem)
        if from_name:
            ids.append(from_name.group(1))
        ids.sort(key=lambda i: not i.startswith("BIOMD"))
        _IDENTITY_CACHE[key] = {
            "file": path.name,
            "model_name": (model.getName() or model.getId()) if model is not None else None,
            "biomodels_id": ids[0] if ids else None,
        }
    return dict(_IDENTITY_CACHE[key])


def get_annotations(model_name_or_path: str, components: list[str] | None = None,
                    only_annotated: bool = True, max_per_component: int = 50) -> str:
    """Tool: annotations of an SBML model and (optionally) its components."""
    from dynalyst.analysis.ode_model import xpp_file
    xpp = xpp_file(model_name_or_path)
    if xpp is not None:
        return json.dumps({"error": f"{xpp.name} is an XPP file: XPP files carry no annotations "
                                    "(names, authors, database references), only equations."})
    try:
        path = resolve_sbml_path(model_name_or_path)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is None:
        return json.dumps({"error": f"Could not read an SBML model from {path}."})

    components = components or list(_COMPONENT_LISTS)
    unknown = [c for c in components if c not in _COMPONENT_LISTS]
    if unknown:
        return json.dumps({"error": f"Unknown component type(s) {unknown}. "
                                    f"Valid: {list(_COMPONENT_LISTS)}"})

    result = {
        "model_file": str(path),
        "sbml_level_version": f"L{doc.getLevel()}V{doc.getVersion()}",
        "model": {**_element(model), "history": _history(model)},
        "counts": {c: getattr(model, fn)().size() for c, fn in _COMPONENT_LISTS.items()},
    }
    n_annotated_total = 0
    for comp in components:
        elements = [_element(x) for x in getattr(model, _COMPONENT_LISTS[comp])()]
        annotated = [e for e in elements if len(e) > 2 or e.get("name")]
        n_annotated_total += sum(1 for e in elements if "cross_references" in e or "sbo_term" in e)
        chosen = annotated if only_annotated else elements
        result[comp] = chosen[:max_per_component]
        if len(chosen) > max_per_component:
            result[f"{comp}_truncated"] = len(chosen) - max_per_component
    result["n_components_with_cross_references_or_sbo"] = n_annotated_total
    result["has_any_annotation"] = bool(
        n_annotated_total or result["model"].get("cross_references")
        or result["model"].get("notes") or result["model"].get("history"))
    return json.dumps(result, indent=2)


TOOL_DEFINITIONS = [{
    "name": "get_annotations",
    "description": (
        "Read the annotations stored in an SBML model file: model name, notes, "
        "authors and dates, SBO terms, and MIRIAM cross-references (UniProt, ChEBI, GO, "
        "KEGG, PubMed, BioModels ...) for the model and its compartments, species, "
        "reactions and parameters. Reads the file only; does not look anything up online."),
    "input_schema": {"type": "object", "properties": {
        "model_name_or_path": {"type": "string", "description": "SBML model name or path."},
        "components": {"type": "array", "items": {"type": "string",
                       "enum": list(_COMPONENT_LISTS)},
                       "description": "Which component types to include. Default: all."},
        "only_annotated": {"type": "boolean",
                           "description": "Default true: list only components that carry a name, notes, SBO term or cross-reference."},
        "max_per_component": {"type": "integer", "description": "Cap per component type. Default 50."},
    }, "required": ["model_name_or_path"]},
}]

TOOL_FUNCTIONS = {"get_annotations": lambda **kw: get_annotations(**kw)}
