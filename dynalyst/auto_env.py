"""Make AUTO-07p's Python interface importable in the current process.

The LHS / AUTO pipeline calls AUTO in-process (`from auto import run`), which
needs three things set up first: the AUTO_DIR environment variable (AUTO's
makefiles use it to find cmds/cmds.make and lib/*.o), AUTO's bin/ and cmds/
on PATH, and AUTO's python/ directory on sys.path. The bifurcation agent's
notebook used to do this by hand in its first cell with a hard-coded
/E-drive path; ensure_auto_env() does it from config.AUTO_DIR instead.

The research pipeline (analysis/bifurcation_auto.py) runs AUTO in a
subprocess and builds that subprocess's environment itself, so it does not
depend on this module.
"""
import functools
import os
import re
import sys
from pathlib import Path

from dynalyst import config

# ---------------------------------------------------------------------------
# AUTO itself not working (a setup problem) vs a continuation that failed
# ---------------------------------------------------------------------------
# AUTO compiles every model with gfortran and links it against AUTO's own
# lib/*.o. When that build fails -- objects compiled on another machine (the
# bundled lib/*.o were built on a Mac: "file format not recognized"), no
# gfortran / make, AUTO never built -- EVERY run fails before any
# continuation starts. Reported as "samples failed to converge" or "no
# branches", that looks like a property of the model and sends the agent
# retrying with other parameters, which cannot help. It is raised as
# AutoUnavailable instead, with the compiler's own words.

AUTO_UNAVAILABLE = "AUTO-07p cannot run on this machine"
_BUILD_FAILURE = re.compile(
    r"file format not recognized|file not recognized|unknown file type|collect2|ld returned"
    r"|cannot find -l|undefined reference to|relocation R_|recompile with -fPIC"
    r"|gfortran: (?:command )?not found|command not found|No such file or directory: '(?:make|gfortran)"
    r"|not recognized as an internal or external command|has not been built|AUTO-07p not found"
    r"|cannot be linked here"
    # a program AUTO runs (make, gfortran, the compiled model) is missing
    r"|WinError 2\]|The system cannot find the file specified|\[Errno 2\] No such file or directory: '[^']*'$",
    re.I)


class AutoUnavailable(RuntimeError):
    """AUTO-07p cannot build or run models on this machine: a setup problem,
    not a property of the model being analysed."""


def build_failure(text: str) -> str | None:
    """The line of compiler / linker output that shows AUTO cannot build
    models here, or None."""
    for line in (text or "").splitlines():
        if _BUILD_FAILURE.search(line):
            return line.strip()[:300]
    return None


BUILD_CMD = "python scripts/build_auto.py"     # or: bash scripts/build_auto.sh
BUILD_STAMP = "BUILD_INFO.json"                 # written last by a complete build


def unavailable_message(reason: str) -> str:
    return (f"{AUTO_UNAVAILABLE}: {reason}. This is a setup problem, not a property of the "
            f"model -- no choice of parameters or window can fix it. Build AUTO for this "
            f"machine with `{BUILD_CMD}` (or `bash scripts/build_auto.sh`; needs gfortran), "
            f"then run the analysis again.")


def auto_dir() -> Path:
    """AUTO install dir: env AUTO_DIR wins, else config.AUTO_DIR."""
    env_dir = os.environ.get("AUTO_DIR")
    return Path(env_dir) if env_dir else Path(config.AUTO_DIR)


# ---------------------------------------------------------------------------
# One compiled library per OS / CPU / compiler
# ---------------------------------------------------------------------------
# scripts/build_auto.py puts AUTO's objects in lib/<platform>-<compiler>/ so that one
# checkout, copied or synced between a Mac, Linux machines and a Windows PC, works on
# each machine it has been built on. AUTO compiles every model with the local gfortran
# and links it against these objects, so they must come from the same toolchain: the
# compiler's major version is part of the folder name, and each build records the
# machine that made it (a library built elsewhere is not used). lib/*.o directly in
# lib/ is the older in-place build (configure + make): used only when it was compiled
# for this platform.

def platform_tag() -> str:
    """e.g. 'win32-amd64', 'linux-x86_64', 'darwin-arm64'."""
    import platform
    return f"{sys.platform}-{platform.machine().lower() or 'unknown'}"


@functools.lru_cache(maxsize=None)
def compiler_tag(fc: str | None = None) -> str | None:
    """'gfortran16' for the Fortran compiler AUTO uses here (FC, else gfortran), or
    None when there is none."""
    import shutil
    import subprocess
    fc = fc or os.environ.get("FC") or "gfortran"
    exe = shutil.which(fc)
    if not exe:
        return None
    try:
        version = subprocess.run([exe, "-dumpversion"], capture_output=True, text=True,
                                 timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    major = version.split(".")[0] if version else "unknown"
    return f"{Path(fc).stem.split('-')[0]}{major}"


def this_host() -> str:
    import platform
    return platform.node() or "unknown"


def built_on(folder) -> str | None:
    """The machine a library folder was built on (its BUILD_INFO.json), if recorded."""
    try:
        import json
        return json.loads((Path(folder) / BUILD_STAMP).read_text()).get("host")
    except (OSError, ValueError):
        return None


def lib_dir(root=None) -> Path:
    return Path(root or auto_dir()) / "lib" / f"{platform_tag()}-{compiler_tag() or 'nofortran'}"


def _cpu(machine: str) -> str:
    m = machine.lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(m, m)


def object_platform(path) -> tuple[str | None, str | None, str]:
    """(sys.platform prefix, CPU, description) of the machine an object file was
    built for, read from its header (Mach-O cputype, ELF e_machine, COFF machine)."""
    try:
        head = Path(path).read_bytes()[:20]
    except OSError:
        return None, None, "unreadable"
    if head[:4] == b"\xcf\xfa\xed\xfe":
        cpu = {7: "x86_64", 12: "arm64"}.get(int.from_bytes(head[4:8], "little") & 0xFFFFFF)
        return "darwin", cpu, f"macOS {cpu or ''} (Mach-O)"
    if head[:4] == b"\x7fELF":
        cpu = {0x3E: "x86_64", 0xB7: "arm64"}.get(int.from_bytes(head[18:20], "little"))
        return "linux", cpu, f"Linux {cpu or ''} (ELF)"
    cpu = {0x8664: "x86_64", 0xAA64: "arm64"}.get(int.from_bytes(head[:2], "little"))
    if cpu:
        return "win32", cpu, f"Windows {cpu} (COFF)"
    return None, None, "an unknown format"


def find_lib_dir(root=None) -> Path:
    """The folder of AUTO objects usable on this machine; RuntimeError (with what to
    do) when there is none."""
    root = Path(root or auto_dir())
    if compiler_tag() is None:
        raise RuntimeError("gfortran: not found on PATH. AUTO-07p compiles every model "
                           "with it; install it (see scripts/build_auto.py)")
    own = lib_dir(root)
    if (own / BUILD_STAMP).exists() and any(own.glob("*.o")):
        host = built_on(own)     # None: built before hosts were recorded -- origin unknown
        if host != this_host() and os.environ.get("DYNALYST_AUTO_SHARED_BUILD") != "1":
            raise RuntimeError(
                f"AUTO-07p's library {own.name} was built on another machine "
                f"({host or 'unknown: built before machines were recorded'}); "
                f"this is {this_host()}. A library compiled elsewhere cannot be linked here "
                f"safely (a mismatched build can crash with a segmentation fault). Run "
                f"`{BUILD_CMD} --force` on this machine (or set DYNALYST_AUTO_SHARED_BUILD=1 "
                f"for machines that share one toolchain)")
        return own
    other = sorted(p.name for p in (root / "lib").glob(f"{platform_tag()}*")
                   if (p / BUILD_STAMP).exists())
    if other:
        raise RuntimeError(
            f"AUTO-07p at {root} was built for this platform with another compiler "
            f"({', '.join(other)}); this machine's is {compiler_tag()}. Objects from another "
            f"gfortran version cannot be linked here (models crash). Run `{BUILD_CMD}`")
    legacy = sorted((root / "lib").glob("*.o"))
    if legacy:
        import platform
        plat, cpu, what = object_platform(legacy[0])
        if plat and sys.platform.startswith(plat) and cpu in (None, _cpu(platform.machine())):
            return root / "lib"
        raise RuntimeError(
            f"AUTO-07p at {root} was compiled for {what}, not for this machine "
            f"({platform_tag()}): its lib/*.o cannot be linked here. Run `{BUILD_CMD}`")
    raise RuntimeError(f"AUTO-07p at {root} has not been built for {platform_tag()}. "
                       f"Run `{BUILD_CMD}` first.")


def posix_path(path) -> str:
    """AUTO's runner splits its compile / link / run commands POSIX-style, which reads a
    Windows backslash as an escape: hand it forward slashes (gfortran accepts both)."""
    return str(path).replace("\\", "/") if os.name == "nt" else str(path)


def ensure_auto_env() -> Path:
    """Idempotently configure env vars / sys.path for `import auto`."""
    root = auto_dir()
    if not (root / "python" / "auto").is_dir():
        raise RuntimeError(
            f"AUTO-07p not found at {root}. Build the bundled copy with "
            f"`{BUILD_CMD}`, or set AUTO_DIR to an existing install."
        )
    os.environ["AUTO_LIB_DIR"] = posix_path(find_lib_dir(root))
    os.environ["AUTO_DIR"] = posix_path(root)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for extra in (root / "cmds", root / "bin"):
        if str(extra) not in path_parts:
            path_parts.insert(0, str(extra))
    os.environ["PATH"] = os.pathsep.join(path_parts)
    py_dir = str(root / "python")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    return root
