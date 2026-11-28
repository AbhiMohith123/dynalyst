"""Prune old AUTO-07p run folders and CRNT job folders once per process.
LHS work folders are retained because they also contain final results and plots."""
import shutil
import time

from dynalyst import config

_DONE = False


def scratch_dirs():
    # WORKDIR contains durable LHS summaries and plots, including older runs.
    return [config.AUTO_WORK_DIR, config.OUTPUT_DIR / "crnt" / "jobs"]


def prune_scratch(days: float | None = None) -> int:
    """Remove scratch folders not touched for `days` (config.SCRATCH_KEEP_DAYS); returns how
    many. A folder still in use is touched by its run and is kept. 0 or less: keep all."""
    days = config.SCRATCH_KEEP_DAYS if days is None else days
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for parent in scratch_dirs():
        if not parent.is_dir():
            continue
        for entry in parent.iterdir():
            try:
                if entry.is_dir() and _newest_mtime(entry) < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    return removed


def prune_once() -> None:
    """prune_scratch, at most once per process; never lets a failure stop the caller."""
    global _DONE
    if _DONE:
        return
    _DONE = True
    try:
        prune_scratch()
    except Exception:
        pass


def _newest_mtime(folder) -> float:
    newest = folder.stat().st_mtime
    for p in folder.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    return newest
