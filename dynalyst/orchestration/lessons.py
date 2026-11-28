"""Lessons learned from runs: a call that was rejected, failed or sent back by the reviewer,
then succeeded once its arguments changed, leaves a lesson. The next sub-agent that works with that tool is
shown the matching lessons as hints.

A lesson is data in a file (config.LESSONS_PATH), never code or prompt text. It holds
only the tool, the kind of problem and which ARGUMENT NAMES changed: no model name,
parameter name, value or path is stored, so a lesson applies to any model.

Only a success teaches: a lesson is recorded when the retry succeeded, never from a failed
run. Lessons expire (config.LESSONS_MAX_AGE_DAYS) and the
file is capped (config.LESSONS_MAX_ENTRIES). DYNALYST_LESSONS=0 turns it all off.
"""
import json
import re
import threading
import time

from dynalyst import config

_LOCK = threading.Lock()
# arguments that name the model or a file: never part of a lesson
_IDENTITY_KEYS = {"model", "model_name_or_path", "sbml_path", "path", "path_or_name", "pdf",
                  "data_path", "scan_csv", "name", "filename", "model_id"}
_ACCEPTED = ("ok",)


def _problem(text: str) -> str:
    """The kind of problem, without the names, numbers and paths of this run."""
    text = re.sub(r"(?<!\w)'[^']*'(?!\w)|\"[^\"]*\"", "'X'", text or "")   # not an apostrophe
    text = re.sub(r"\S+\.(?:xml|ode|txt|csv|sbml)\b", "FILE", text)
    text = re.sub(r"\b[A-Z]{2,}\d{3,}\w*\b", "ID", text)
    text = re.sub(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", "#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(". ")[0][:160]


# a problem of the machine or the world, not of the arguments: never a lesson
_ENVIRONMENT = re.compile(r"no module named|not installed|cannot run on this machine|timed? ?out|"
                          r"connection|unavailable|not found|no pdf matching|403|404|rate limit|"
                          r"could not download|server is having|credentials", re.I)


def _failure(attempt: dict) -> str | None:
    """Why an attempt did not stand: the guard's reason, the tool's error, or the issues
    the reviewer sent it back for."""
    guard = attempt.get("guard") or {}
    if guard and not guard.get("allowed", True):
        return _problem(guard.get("reason") or "")
    if attempt.get("status") == "error":
        return _problem(attempt.get("error") or "")
    verdict = attempt.get("reviewer") or {}
    if verdict.get("verdict") in ("replan", "unresolved"):
        return _problem("; ".join(verdict.get("issues") or []))
    return None


def _diff(before: dict, after: dict) -> dict:
    keys = (set(before) | set(after)) - _IDENTITY_KEYS
    return {"changed": sorted(k for k in keys if k in before and k in after
                              and before[k] != after[k]),
            "added": sorted(k for k in keys if k in after and k not in before),
            "removed": sorted(k for k in keys if k in before and k not in after)}


def _enabled() -> bool:
    return config.LESSONS_ENABLED


def _read() -> list:
    path = config.LESSONS_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _write(entries: list) -> None:
    path = config.LESSONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries),
                    encoding="utf-8")


def record(attempts: list) -> list:
    """Save the lessons of one successful analysis from its attempts (failed ones first,
    the successful one last). Returns the lessons saved."""
    if not _enabled() or len(attempts) < 2:
        return []
    final = attempts[-1]
    if final.get("status") not in _ACCEPTED:
        return []
    saved = []
    for earlier in attempts[:-1]:
        why = _failure(earlier)
        if not why:
            continue
        if earlier.get("tool") != final.get("tool") or _ENVIRONMENT.search(why):
            continue        # a tool switch or a machine problem teaches nothing general
        diff = _diff(earlier.get("arguments") or {}, final.get("arguments") or {})
        if not any(diff.values()):
            continue                          # nothing changed: nothing to learn
        saved.append({"tool": earlier.get("tool"), "problem": why, **diff})
    if not saved:
        return []
    now = time.time()
    with _LOCK:
        entries = _read()
        for lesson in saved:
            key = {k: lesson.get(k) for k in ("tool", "problem", "changed", "added", "removed")}
            for e in entries:
                if {k: e.get(k) for k in key} == key:
                    e["count"] = e.get("count", 1) + 1
                    e["updated"] = now
                    break
            else:
                entries.append({**lesson, "count": 1, "created": now, "updated": now})
        entries.sort(key=lambda e: (e.get("count", 1), e.get("updated", 0)), reverse=True)
        _write(entries[:config.LESSONS_MAX_ENTRIES])
    return saved


def _sentence(e: dict) -> str:
    parts = []
    for label, key in (("changing", "changed"), ("adding", "added"), ("removing", "removed")):
        if e.get(key):
            parts.append(f"{label} the argument(s) {', '.join(e[key])}")
    seen = f" (seen {e['count']} times)" if e.get("count", 1) > 1 else ""
    return f"- {e['tool']}: after \"{e['problem']}\" the call succeeded by " \
           f"{' and '.join(parts)}{seen}."


def for_tools(tools, limit: int | None = None) -> str:
    """The lessons for these tools as hints for a sub-agent's task, or '' (none, or off)."""
    if not _enabled():
        return ""
    cutoff = time.time() - 86400 * config.LESSONS_MAX_AGE_DAYS
    with _LOCK:
        entries = [e for e in _read() if e.get("tool") in set(tools)
                   and e.get("updated", 0) >= cutoff]
    entries.sort(key=lambda e: (e.get("count", 1), e.get("updated", 0)), reverse=True)
    entries = entries[:limit or config.LESSONS_PER_TASK]
    if not entries:
        return ""
    return ("\n\nLessons from earlier runs (hints from calls that were corrected and then "
            "succeeded; the user's request and the tool's own limits come first, and a hint "
            "that does not fit this task is ignored):\n" + "\n".join(_sentence(e) for e in entries))
