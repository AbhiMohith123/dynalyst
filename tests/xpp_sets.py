"""Test data: parameter sets of the XPP model models/pdp_mm_bistable.txt.

The file holds three parameter blocks (two commented out). The tool code
reads it as a plain XPP model with its active block; these sets let the AUTO
tests run the other blocks through the same generic model path, by writing a
copy of the file whose active `par` lines carry the chosen values.
"""
import re

from dynalyst import config

XPP_FILE = config.MODELS_DIR / "pdp_mm_bistable.txt"

MARKEVICH = {"ATOT": 0.1, "KTOT": 500.0, "PTOT": 107.5, "km1": 50.5, "km3": 500.0, "km5": 20.2,
             "km7": 0.16, "k2": 0.01, "k4": 15.0, "k6": 0.01, "k8": 0.15}
PITCHFORK = {"ATOT": 0.1, "KTOT": 100.0, "PTOT": 100.0, "km1": 50.5, "km3": 500.0, "km5": 50.5,
             "km7": 500.0, "k2": 0.01, "k4": 0.2, "k6": 0.01, "k8": 0.2}
TRANSCRITICAL = {"ATOT": 0.1, "KTOT": 100.0, "PTOT": 100.0, "km1": 50.5, "km3": 500.0,
                 "km5": 5.1, "km7": 500.0, "k2": 0.01, "k4": 0.2, "k6": 0.01, "k8": 0.2}


def write_xpp(path, values) -> str:
    """A copy of the XPP file at `path` whose active parameters take `values`."""
    def sub(m):
        name = m.group(1)
        return f"par {name}={values[name]:g}" if name in values else m.group(0)
    path.write_text(re.sub(r"^par (\w+)=\S+", sub, XPP_FILE.read_text(), flags=re.M))
    return str(path)


def parsed(path):
    """The AUTO-ready parsed form of a model file (the LHS / AUTO pipeline's own reader)."""
    from dynalyst.regime import model_source
    return model_source.load_model(str(path))["parsed"]
