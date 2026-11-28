import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before dynalyst.config is imported: test runs write their
# plots / workdirs / scan memory here instead of into outputs/.
TEST_OUTPUT_DIR = ROOT / "outputs" / "_test_runs"
os.environ["DYNALYST_OUTPUT_DIR"] = str(TEST_OUTPUT_DIR)
os.environ.setdefault("DYNALYST_AUTO_INDEX_MODELS", "0")   # no RAG writes from the suite
# CRNT in-process, so tests can stub what it calls; the worker with its time limit has
# its own tests (test_robustness.py)
os.environ.setdefault("DYNALYST_CRNT_MAX_SECONDS", "0")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)


def _auto_ready() -> bool:
    try:
        from dynalyst.auto_env import ensure_auto_env
        ensure_auto_env()
        return True
    except Exception:   # not built, not unpacked, or no gfortran
        return False


@pytest.fixture
def stubbed_auto(monkeypatch):
    """The LHS batch runner with AUTO and plotting stubbed out. Returns a dict
    recording the last call; its "specials" list, when a test fills it, gives the
    special points (idx, type, label) each sample (cycled) comes back with, and
    "calls" lists the out_dir of every _run_samples call (a control run ends in
    'control')."""
    from dynalyst.regime import batch_lhs_runner
    seen = {}
    specials = []

    def fake(samples, bif_param, param_range, out_dir, y_species, parsed_base, unavailable=None,
             log_probes=False):
        seen.update(bif=bif_param, rng=tuple(param_range), n=len(samples))
        seen.setdefault("calls", []).append(out_dir)
        out = batch_lhs_runner.BatchResults()
        for i, s in enumerate(samples):
            sp = specials[i % len(specials)] if specials else []
            out.append({"sample_id": i, "params": s, "all_branches": [], "special": sp,
                        "signature": batch_lhs_runner.classify_signature(sp), "error": None})
        return out

    monkeypatch.setattr(batch_lhs_runner, "_run_samples", fake)
    monkeypatch.setattr(batch_lhs_runner, "mark_special_points", lambda *a, **k: set())
    monkeypatch.setattr(batch_lhs_runner, "_plot_branch_with_stability", lambda *a, **k: None)
    seen["specials"] = specials
    return seen


def pytest_configure(config):
    config.addinivalue_line("markers", "auto: needs a built AUTO-07p (python scripts/build_auto.py)")


def need_pdf(*names):
    """Skip unless the named papers are in data/papers (the PDFs used in development
    are not distributed, as they belong to their publishers)."""
    from dynalyst import config as _cfg
    missing = [n for n in names if not (_cfg.PAPERS_DIR / n).exists()]
    if missing:
        pytest.skip(f"paper(s) not in data/papers (not distributed): {', '.join(missing)}")


def pytest_collection_modifyitems(config, items):
    if _auto_ready():
        return
    skip = pytest.mark.skip(reason="AUTO-07p not built for this machine -- run "
                                   "python scripts/build_auto.py or set AUTO_DIR")
    for item in items:
        if "auto" in item.keywords:
            item.add_marker(skip)
