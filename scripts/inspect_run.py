"""What did the agent actually do in a run? Reads a run record (outputs/runs/<time>.json,
written after every query) and prints every plan version the orchestrator wrote, every
tool call on a timeline (which ran in parallel), the team supervisors, what the language
models decided, and where one of the code exceptions stepped in.

    python scripts/inspect_run.py              # the latest run
    python scripts/inspect_run.py 3            # the 3rd latest
    python scripts/inspect_run.py outputs/runs/20261013-201500.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _record(arg):
    from dynalyst import config
    if arg and Path(arg).exists():
        return Path(arg)
    runs = sorted((config.OUTPUT_DIR / "runs").glob("*.json"))
    if not runs:
        sys.exit(f"No run records in {config.OUTPUT_DIR / 'runs'} (they are written after every query).")
    n = int(arg) if arg and arg.isdigit() else 1
    return runs[-n]


def main():
    path = _record(sys.argv[1] if len(sys.argv) > 1 else None)
    rec = json.loads(path.read_text(encoding="utf-8"))
    arch = rec.get("architecture") or {}
    print(f"Run   {path.name}   {rec.get('seconds')} s")
    print(f"Query {rec['query'][:200]}")
    print(f"Mode  {arch.get('mode')}, parallel {'on' if arch.get('parallel') else 'off'}, "
          f"approval {arch.get('approval')}")

    print("\nPLAN (written by the orchestrator)")
    for v in rec.get("plan_versions") or []:
        why = f" -- revised: {v['reason']}" if v.get("reason") else ""
        print(f"  v{v['version']} (after {v.get('after_tool_calls', 0)} tool call(s)){why}")
        for i, s in enumerate(v.get("steps") or [], start=1):
            print(f"     {i}. {s[:120]}")
    if not rec.get("plan_versions"):
        print("  (none: answered directly or with utility tools)")

    calls = [s for s in rec.get("steps") or [] if s.get("tool")]
    t0 = min([s["started"] for s in calls if s.get("started")] or [0])
    print("\nTOOL CALLS  (seconds from the first call)")
    spans = []
    for s in calls:
        start = s.get("started") or 0
        end = start + (s.get("seconds") or 0)
        who = s.get("sub_agent") or "orchestrator"
        spans.append((start, end, who))
        timing = f"{start - t0:6.1f} +{s.get('seconds') or 0:5.1f}" if start else "   (utility)  "
        print(f"  {timing}  {who}: {s['tool']} -> {s.get('status')}"
              + (f"\n                 {str(s.get('error'))[:150]}" if s.get("error") else ""))
    overlaps = sorted({(a[2], b[2]) for i, a in enumerate(spans) for b in spans[i + 1:]
                       if a[0] and b[0] and a[0] < b[1] and b[0] < a[1]})

    for tr in rec.get("team_runs") or []:
        if tr is rec["team_runs"][0]:
            print("\nTEAM SUPERVISORS")
        print(f"  {tr['team']} ({tr.get('model_calls')} calls): {(tr.get('task') or '')[:100]}")
        for d in tr.get("delegations") or []:
            print(f"     -> {d.get('sub_agent')}: {(d.get('task') or '')[:90]}")

    errors = [s for s in calls if s.get("status") == "error"]
    recovered = [s for s in errors if any(o.get("tool") == s["tool"] and o.get("status") == "ok"
                                          and calls.index(o) > calls.index(s) for o in calls)]
    print("\nDECIDED BY LANGUAGE MODELS")
    print(f"  plan versions     {len(rec.get('plan_versions') or [])}")
    print(f"  tool calls        {len(calls)}" + (f", in parallel: {overlaps}" if overlaps else ""))
    print(f"  tool errors       {len(errors)}, {len(recovered)} recovered by a corrected call")
    print(f"  model calls       {rec.get('model_calls')}")
    print("\nCODE EXCEPTIONS THAT STEPPED IN")
    for i in rec.get("interventions") or []:
        print(f"  {i['kind']}: " + (f"{i.get('tool')} ({str(i.get('reason'))[:120]})"
                                    if i.get("tool") else ""))
    if rec.get("awaiting_approval") or rec.get("approval"):
        print(f"  approval: {rec.get('approval') or 'asked (run paused)'}")
    fc = rec.get("final_check") or {}
    print("  final check: " + ("answer rewritten once" if fc.get("rewritten") else "answer unchanged")
          + (f"; removed IDs {fc['model_ids_removed']}" if fc.get("model_ids_removed") else "")
          + (f"; untraced values {fc['unsupported_after_rewrite']}"
             if fc.get("unsupported_after_rewrite") else ""))


if __name__ == "__main__":
    main()
