"""Summary over many run records: how often the agent revised its plan and recovered from
tool errors, and what it cost (model calls, tokens, time).

    python scripts/summarize_runs.py [folder ...]     # default: outputs/runs

Each run record is the JSON the agent writes per question (outputs/runs/<time>.json).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(folders):
    recs = []
    for folder in folders:
        for p in sorted(Path(folder).rglob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict) and "query" in rec and "steps" in rec:
                rec["_path"] = str(p)
                recs.append(rec)
    return recs


def row(rec) -> dict:
    tokens = rec.get("tokens") or {}
    return {
        "query": " ".join(str(rec.get("query")).split())[:60],
        "revisions": max(len(rec.get("plan_versions") or []) - 1, 0),
        "errors": sum(1 for s in rec.get("steps") or [] if s.get("status") == "error"),
        "exceptions": len(rec.get("interventions") or []),
        "calls": sum((rec.get("model_calls") or {}).values()),
        "input": sum(t.get("input", 0) for t in tokens.values()),
        "cached": sum(t.get("cached_input", 0) for t in tokens.values()),
        "output": sum(t.get("output", 0) for t in tokens.values()),
        "seconds": rec.get("seconds") or 0,
    }


def summarize(recs) -> str:
    if not recs:
        return "No run records found."
    rows = [row(r) for r in recs]
    lines = [f"{'query':60}  rev  err  exc  calls  tokens in/cached/out   s"]
    for r in rows:
        lines.append(f"{r['query']:60}  {r['revisions']:>3}  {r['errors']:>3}  "
                     f"{r['exceptions']:>3}  {r['calls']:>5}  "
                     f"{r['input']:>7}/{r['cached']:>6}/{r['output']:>6}  {r['seconds']:>5.0f}")
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    cached_share = sum(r["cached"] for r in rows) / max(sum(r["input"] for r in rows), 1)
    lines += ["",
              f"runs {n} | plan revised {sum(r['revisions'] > 0 for r in rows)} | "
              f"with tool errors {sum(r['errors'] > 0 for r in rows)} | "
              f"code exceptions stepped in {sum(r['exceptions'] > 0 for r in rows)}",
              f"mean per run: {mean('calls'):.1f} model calls, {mean('input'):.0f} input tokens "
              f"({cached_share:.0%} from the prompt cache), {mean('output'):.0f} output tokens, "
              f"{mean('seconds'):.0f} s"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(load(sys.argv[1:] or [ROOT / "outputs" / "runs"])))
