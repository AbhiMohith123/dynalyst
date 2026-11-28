"""Build the bundled AUTO-07p library for THIS machine: Windows, Linux or macOS.

    python scripts/build_auto.py            # build, then run a 1-D continuation as a check
    python scripts/build_auto.py --check    # only the check
    python scripts/build_auto.py --force    # rebuild even if this platform is built

The objects go to vendor/auto-07p/lib/<platform>-<compiler>/ (e.g.
linux-x86_64-gfortran11, win32-amd64-gfortran16, darwin-arm64-gfortran14), one folder
per OS / CPU / gfortran major version, so a checkout copied or synced between machines
works on each of them once it has been built there. Objects compiled on another
machine, or by another gfortran version, are never linked: a macOS lib/*.o cannot link
on Linux, and gfortran-16 objects linked into a model compiled by another gfortran
crash with a segmentation fault.

Needs gfortran (and ar, which comes with it). AUTO's own `configure` / `make` are
not used: the agent builds and runs models through AUTO's Python interface, which
only needs the library objects, include/ and cmds/cmds.make.

  Windows   a MinGW-w64 gfortran on PATH, e.g. `winget install BrechtSanders.WinLibs.POSIX.UCRT`
            or `conda install -c conda-forge m2w64-gcc-fortran`
  Linux     sudo apt-get install -y gfortran      (or: conda install -c conda-forge gfortran)
  macOS     brew install gcc                       (or: conda install -c conda-forge gfortran)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dynalyst import auto_env  # noqa: E402

# src/Makefile's objects in dependency order (a module must be compiled before its users)
SOURCES = [("auto_constants.f90", "auto_constants.o"), ("nompi.f90", "mpi.o"),
           ("mesh.f90", "mesh.o"), ("support.f90", "support.o"), ("f2003.f90", "compat.o"),
           ("io.f90", "io.o"), ("interfaces.f90", "interfaces.o"), ("solvebv.f90", "solvebv.o"),
           ("ae.f90", "ae.o"), ("bvp.f90", "bvp.o"), ("toolboxae.f90", "toolboxae.o"),
           ("toolboxbv.f90", "toolboxbv.o"), ("floquet.f90", "floquet.o"),
           ("equilibrium.f90", "equilibrium.o"), ("timeint.f90", "timeint.o"),
           ("periodic.f90", "periodic.o"), ("maps.f90", "maps.o"), ("homcont.f90", "homcont.o"),
           ("optimization.f90", "optimization.o"), ("parabolic.f90", "parabolic.o"),
           ("main.f90", "main.o"), ("blas.f", "blas.o"), ("lapack.f", "lapack.o")]


def _tool(name, near=None):
    """A program on PATH, preferring the one next to `near` (same toolchain)."""
    if near:
        for cand in (name, name + ".exe"):
            p = Path(near).parent / cand
            if p.exists():
                return str(p)
    return shutil.which(name)


def _flags(auto_dir: Path) -> tuple[list, list]:
    """FFLAGS and OPT from cmds/cmds.make, the flags AUTO compiles each model with."""
    var = {"FFLAGS": "-g -fopenmp", "OPT": "-O"}
    for line in (auto_dir / "cmds" / "cmds.make").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "=" and parts[0] in var:
            var[parts[0]] = " ".join(parts[2:])
    return var["FFLAGS"].split(), var["OPT"].split()


def _run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")


def build(auto_dir: Path, force: bool = False) -> Path:
    out = auto_env.lib_dir(auto_dir)
    stamp = out / auto_env.BUILD_STAMP
    host = auto_env.built_on(out) if stamp.exists() else None
    if stamp.exists() and not force and host == auto_env.this_host():
        print(f"AUTO-07p is already built for {out.name} in {out} (--force rebuilds)")
        return out
    if stamp.exists() and not force:
        print(f"{out.name} was built on another machine ({host or 'unknown'}): rebuilding it here")
    fc = os.environ.get("FC") or _tool("gfortran")
    if not fc:
        sys.exit("ERROR: gfortran not found on PATH. " + __doc__.split("Needs gfortran")[1])
    ar = _tool("ar", near=fc) or _tool("gcc-ar", near=fc)
    if not ar:
        sys.exit(f"ERROR: 'ar' not found next to {fc} or on PATH.")
    fflags, opt = _flags(auto_dir)
    src, inc = auto_dir / "src", auto_dir / "include"
    out.mkdir(parents=True, exist_ok=True)
    if stamp.exists():
        stamp.unlink()                  # an interrupted rebuild must not look complete
    version = subprocess.run([fc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"Building AUTO-07p for {out.name} with {version}")
    t0 = time.time()
    # compile in a scratch folder: the .mod files in src/ were written by another
    # machine's compiler and must not be picked up
    with tempfile.TemporaryDirectory(prefix="auto07p_build_") as tmp:
        for source, obj in SOURCES:
            print(f"  {source}")
            extra = ["-I", str(inc)] if source == "auto_constants.f90" else []
            _run([fc, *fflags, *opt, *extra, "-J", tmp, "-I", tmp, "-c", str(src / source),
                  "-o", str(out / obj)], cwd=tmp)
        _run([fc, *fflags, *opt, "-J", tmp, "-I", tmp, "-c", str(src / "user_c.f90"),
              "-o", "user_c.o"], cwd=tmp)
        lib_c = out / "libauto_c.a"
        if lib_c.exists():
            lib_c.unlink()
        _run([ar, "cr", str(lib_c), "user_c.o"], cwd=tmp)
    stamp.write_text(json.dumps({"platform": auto_env.platform_tag(), "folder": out.name,
                                 "host": auto_env.this_host(), "compiler": version,
                                 "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"Built {len(SOURCES)} objects in {out} ({time.time() - t0:.0f} s)")
    return out


_CHECK_F90 = """SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
      F(1) = PAR(1) + U(1) - U(1)**3
      END SUBROUTINE FUNC
      SUBROUTINE STPNT(NDIM,U,PAR,T)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      PAR(1) = -1.0D0
      U(1) = -1.3247179572447460D0
      END SUBROUTINE STPNT
      SUBROUTINE BCND
      END SUBROUTINE BCND
      SUBROUTINE ICND
      END SUBROUTINE ICND
      SUBROUTINE FOPT
      END SUBROUTINE FOPT
      SUBROUTINE PVLS
      END SUBROUTINE PVLS
"""
_CHECK_C = """NDIM=   1, IPS =   1, IRS =   0, ILP =   1
ICP = ['p']
NTST=  20, NCOL=   4, IAD =   3, ISP =   2, ISW = 1, IPLT= 0
NMX=  2000, NPR=  100, MXBF=  0, IID =   2, ITMX=  8
ITNW=   7, NWTN=   3, JAC= 0
EPSL= 1e-07, EPSU = 1e-07, EPSS = 1e-05
DS  =   0.01, DSMIN=   0.0001, DSMAX=   0.05, IADS=   1
NPAR = 1
UZSTOP = {'p': 1.0}
unames = {1:'x'}
parnames = {1:'p'}
"""


def check() -> int:
    """Continue dx/dt = p + x - x^3 in p from -1 to 1 with AUTO: the S-shaped curve
    has two folds at p = -/+ 2/(3*sqrt(3)) = -/+ 0.3849. Runs in a subprocess the way
    the agent runs AUTO (analysis/bifurcation_auto.py)."""
    code = (
        "import contextlib, io, sys\n"
        "from dynalyst.auto_env import ensure_auto_env\n"
        "ensure_auto_env()\n"
        "from auto import run\n"
        "log = io.StringIO()\n"
        "with contextlib.redirect_stdout(log):\n"
        "    r = run(e='fold', c='fold')\n"
        "b = r[0]\n"
        "lp = sorted(round(float(b['p'][i]), 4) for i in b.labels.getIndices()\n"
        "            if 'LP' in b.labels[i])\n"
        "print('LP', lp)\n"
        "sys.exit(0 if len(lp) == 2 and abs(lp[0] + 0.3849) < 2e-3 and abs(lp[1] - 0.3849) < 2e-3 else 1)\n")
    with tempfile.TemporaryDirectory(prefix="auto07p_check_") as tmp:
        Path(tmp, "fold.f90").write_text(_CHECK_F90)
        Path(tmp, "c.fold").write_text(_CHECK_C)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(ROOT)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
        try:
            p = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, capture_output=True,
                               text=True, errors="replace", timeout=300)
        except subprocess.TimeoutExpired:
            print("CHECK FAILED: AUTO did not finish within 300 s")
            return 1
    if p.returncode == 0:
        print(f"Check passed: AUTO found the two folds ({p.stdout.strip().splitlines()[-1]}, "
              "expected -/+0.3849)")
    else:
        print(f"CHECK FAILED (exit {p.returncode})\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p.returncode


def unpack_vendor() -> None:
    """vendor/ ships AUTO-07p and jaxkineticmodel as zip files (the web upload limits the
    number of files). Unpack any that are not unpacked yet, so a manual install works
    without setup_env.sh."""
    import zipfile
    vendor = ROOT / "vendor"
    for name in ("auto-07p", "jaxkineticmodel"):
        z = vendor / f"{name}.zip"
        if z.exists() and not (vendor / name).is_dir():
            print(f"unpacking {z.relative_to(ROOT)}")
            with zipfile.ZipFile(z) as f:
                f.extractall(vendor)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only run the check")
    ap.add_argument("--force", action="store_true", help="rebuild even if already built")
    args = ap.parse_args()
    unpack_vendor()
    auto_dir = auto_env.auto_dir()
    if not args.check:
        build(auto_dir, force=args.force)
    sys.exit(check())


if __name__ == "__main__":
    main()
