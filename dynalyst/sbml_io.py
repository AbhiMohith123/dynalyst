"""Reading SBML for the ODE builders."""
import libsbml


def _rate_of_placeholders(model) -> list[str]:
    """SBML Level 2 has no rateOf: exporters write it as a function definition 'rateOf'
    whose body is only NaN, standing for Level 3's rateOf. Inlined, every rateOf(x) would
    become NaN; such definitions are not inlined, so the calls stay and are read as rateOf
    (generic_sbml)."""
    ids = []
    for fd in model.getListOfFunctionDefinitions():
        body = fd.getBody() if fd.isSetMath() else None
        if (fd.getId().lower() == "rateof" and body is not None
                and body.getType() == libsbml.AST_REAL and body.getReal() != body.getReal()):
            ids.append(fd.getId())
    return ids


def read_sbml_expanded(path) -> libsbml.SBMLDocument:
    """Read an SBML file with its <functionDefinition>s inlined into every
    kinetic law and rule (libsbml's own converter). Rate laws written as
    ``f(k1, S)`` then become plain expressions, which both ODE builders and
    the Fortran code generator need. A document without function
    definitions is returned unchanged."""
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    if model is not None and model.getNumFunctionDefinitions():
        props = libsbml.ConversionProperties()
        props.addOption("expandFunctionDefinitions", True)
        keep = _rate_of_placeholders(model)
        if keep:
            props.addOption("skipIds", ",".join(keep))
        if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise ValueError(f"{path}: could not expand the SBML function definitions "
                             f"({[f.getId() for f in model.getListOfFunctionDefinitions()]})")
    return doc
