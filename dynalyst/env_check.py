"""Is the Python environment complete? Every package requirements.txt names, plus the ones
installed apart from it (crnt4sbml), checked against what is installed -- so a missing
package is reported once at start-up instead of breaking a run later (a figure that cannot
be drawn, a knowledge base that cannot be searched)."""
import re
from importlib import metadata
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"
# installed apart from requirements.txt (see the comments there)
SEPARATE = {"crnt4sbml": 'pip install --no-deps "crnt4sbml==0.0.15"'}

_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^#;]*)")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(path: Path = REQUIREMENTS) -> list[tuple[str, str]]:
    """(distribution name, version specifier) for each requirement line."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = _LINE.match(line) if line else None
        if m:
            out.append((m.group(1), m.group(3).strip()))
    return out


def problems(path: Path = REQUIREMENTS) -> list[dict]:
    """Each requirement that is missing or installed at a version it does not allow:
    {'name', 'want', 'have', 'install'}."""
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:          # without packaging only presence is checked
        SpecifierSet = None
    installed = {_norm(d.metadata["Name"] or ""): d.version for d in metadata.distributions()}
    out = []
    reqs = requirements(path) + [(n, "") for n in SEPARATE]
    for name, spec in reqs:
        have = installed.get(_norm(name))
        ok = have is not None
        if ok and spec and SpecifierSet is not None:
            try:
                ok = SpecifierSet(spec).contains(have, prereleases=True)
            except Exception:
                ok = True        # a specifier this check cannot read is not a problem
        if not ok:
            install = SEPARATE.get(name) or f'pip install "{name}{spec}"'
            out.append({"name": name, "want": spec or "any", "have": have, "install": install})
    return out


def check_env(path: Path = REQUIREMENTS, quiet_if_ok: bool = True) -> list[dict]:
    """Print one line per missing or mismatched package (with the command that installs
    it), or nothing when the environment is complete. Returns the problems."""
    found = problems(path)
    if found:
        print(f"Environment: {len(found)} package(s) missing or at the wrong version:")
        for p in found:
            state = "not installed" if p["have"] is None else f"installed {p['have']}, need {p['want']}"
            print(f"  {p['name']}: {state}  ->  {p['install']}")
    elif not quiet_if_ok:
        print("Environment: every required package is installed.")
    return found
