"""File safety (deterministic, no language model): checked on EVERY tool call, by any
role, before the tool runs (tools.execute).

  no overwrite   a tool never replaces an existing file; 'overwrite' is refused
  plain names    a name for a new file is a plain file name: no folder, no '..'
  inside folders new files go only into the model folder (data/sbml_models) or outputs/
  no deletion    the only id that reaches a delete (a CRNT job id) must be one the tool
                 itself made: 12 hex characters, so it cannot point outside its folder

A refused call is returned as that tool's error, so the agent reads why and can choose
another name; it never runs.
"""
import re
from pathlib import Path

from dynalyst import config

# arguments that name a NEW file a tool writes into the model folder
NEW_FILE_ARGS = {"name", "model_name"}
# a plain file name: letters, digits, '_', '-', '.', and not a hidden or relative name
_PLAIN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,120}$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")


def _new_model_file(tool: str, args: dict) -> Path | None:
    """The model file this call would create, when the name is given in the call."""
    if tool == "fetch_biomodels_model" and args.get("filename"):
        # a named source file is downloaded over the entry's local copy
        return config.LOCAL_SBML_MODELS_DIR / f"{str(args.get('model_id', '')).strip()}.xml"
    for key in NEW_FILE_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            if tool == "reduce_model" and args.get("export") is False:
                return None
            stem = re.sub(r"\.(xml|sbml)$", "", value.strip(), flags=re.I)
            return config.LOCAL_SBML_MODELS_DIR / f"{stem}.xml"
    return None


def inside(path: Path, *roots: Path) -> bool:
    path = Path(path).resolve()
    return any(path == Path(r).resolve() or Path(r).resolve() in path.parents for r in roots)


def problem(tool: str, args: dict) -> str | None:
    """Why this call may not run, or None."""
    if not isinstance(args, dict):
        return None
    if args.get("overwrite"):
        return ("Existing files are never replaced. Leave out 'overwrite' and give the new file a "
                "name that is not taken yet.")
    for key in NEW_FILE_ARGS & set(args):
        value = args.get(key)
        if isinstance(value, str) and value.strip() and not _PLAIN.match(value.strip()):
            return (f"'{value}' is not a plain file name ({key}): use letters, digits, '_', '-' "
                    "and '.' only, with no folder and no '..'.")
    target = _new_model_file(tool, args)
    if target is not None:
        if not inside(target, config.LOCAL_SBML_MODELS_DIR):
            return f"{tool} may only write into the model folder, not to {target}."
        if target.exists():
            return (f"{target.name} already exists and is never replaced. Use the existing file, "
                    "or give the new one another name.")
    job = args.get("job_id")
    if job is not None and not _JOB_ID.match(str(job)):
        return (f"'{job}' is not a job id this tool returned (12 hex characters). Pass the job_id "
                "exactly as the earlier result gave it.")
    return None


def checked_write(path: Path) -> Path:
    """For code that writes a model file whose name it chose itself: the path, if it is
    inside the model folder or outputs/ and does not exist yet; else ValueError."""
    path = Path(path)
    if not inside(path, config.LOCAL_SBML_MODELS_DIR, config.OUTPUT_DIR):
        raise ValueError(f"refusing to write outside the model folder and outputs/: {path}")
    if path.exists():
        raise FileExistsError(f"{path.name} already exists and is never replaced")
    return path
