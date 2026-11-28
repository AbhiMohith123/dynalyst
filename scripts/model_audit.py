"""Audit every model the agent can load (no AUTO, no model calls).

    python scripts/model_audit.py              # sanity only (fast)
    python scripts/model_audit.py --screen     # + the eigenvalue screen (candidates; slower)
    python scripts/model_audit.py Markevich    # only models whose name contains this

Per model: does it load, how many tunable parameters (by kind), the scale of its
equations at the settled state (model_check.sanity), and with --screen the ranked
candidates. A model that fails here would fail inside AUTO too, for a reason AUTO
cannot report; fix the loader, not the settings. Saves outputs/model_audit.json.
"""
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def audit(path: Path, screen: bool) -> dict:
    from dynalyst.analysis.model_check import sanity
    from dynalyst.analysis.ode_model import resolve_model
    row = {"model": path.name}
    t0 = time.time()
    try:
        m = resolve_model(str(path))
    except Exception as e:
        return {**row, "status": "LOAD FAILED", "detail": f"{type(e).__name__}: {str(e)[:140]}"}
    kinds = {}
    for k in (m.get("param_kinds") or {}).values():
        kinds[k] = kinds.get(k, 0) + 1
    row.update(states=len(m["state_names"]), parameters=len(m["param_names"]), kinds=kinds)
    chk = sanity(m)
    row.update(max_eig=chk["max_eig"], residual=chk["residual"], notes=chk["notes"])
    row["status"] = "ok" if chk["ok"] else "PROBLEM"
    if not chk["ok"]:
        row["detail"] = "; ".join(chk["problems"])[:200]
    if screen and chk["ok"]:
        try:
            from dynalyst.analysis.candidates import propose
            c = propose(str(path), 3)["candidates"]
            row["candidates"] = [f"{x['parameter']} ({x['expected'][:24]})" for x in c]
        except Exception as e:
            row.update(status="SCREEN FAILED", detail=f"{type(e).__name__}: {str(e)[:140]}")
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main(argv):
    from dynalyst import config
    screen = "--screen" in argv
    filters = [a for a in argv if not a.startswith("--")]
    files = sorted(p for d in config.SBML_SEARCH_DIRS if Path(d).exists()
                   for p in Path(d).rglob("*.xml"))
    files = [p for p in files if not filters or any(f.lower() in p.name.lower() for f in filters)]
    rows = []
    for p in files:
        r = audit(p, screen)
        rows.append(r)
        extra = r.get("detail") or ", ".join(r.get("candidates", []))
        print(f"{r['status']:13s} {r['model'][:38]:38s} states={r.get('states', '-'):>3} "
              f"params={r.get('parameters', '-'):>4} max|eig|={r.get('max_eig') or 0:9.2e}  {extra[:90]}")
    bad = [r for r in rows if r["status"] != "ok"]
    out = Path(config.OUTPUT_DIR) / "model_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n{len(rows) - len(bad)}/{len(rows)} models ok; details: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
