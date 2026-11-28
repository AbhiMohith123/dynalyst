"""Session facts: what earlier questions established, for the orchestrator only.

Tools write it (through the agent, after each tool call), so it holds the REAL
values -- the model in use, the continuation parameter and plotted species
actually used, the last diagram / scan (bounds, categories, plots, CSV), the
ranked candidates, a pending approval, saved models and papers -- instead of
whatever survives in the last chat lines. The orchestrator gets them as a
compact block (as_text), to use when the user refers back; teams and
sub-agents never see them. Persisted to outputs/session_state.json so a
restart keeps it; "Clear chat" clears it.
"""
import json
import re
import time
from pathlib import Path

from dynalyst import config

STATE_PATH = config.OUTPUT_DIR / "session_state.json"
_MAX_LIST = 8


def _json(output):
    try:
        d = json.loads(output)
        return d if isinstance(d, dict) else None
    except (TypeError, ValueError):
        return None


def _push(lst, item, key):
    lst[:] = [x for x in lst if x.get(key) != item.get(key)] + [item]
    del lst[:-_MAX_LIST]


def diagram_range_key(model, param) -> str:
    """Markevich2004, Markevich2004.xml and data/sbml_models/Markevich2004.xml are one model."""
    return f"{Path(str(model)).name.removesuffix('.xml')}|{param}"


class SessionState:
    def __init__(self, path=None, persist=True):
        self.path = path or STATE_PATH
        self.persist = persist
        self.data = {"model": None, "bif_param": None, "y_species": None,
                     "last_diagram": None, "last_scan": None, "candidates": None,
                     "pending_approval": None, "saved_models": [], "papers": [],
                     "last_stochastic": None, "network_check": None, "updated": None}
        if persist and self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def __getitem__(self, k):
        return self.data.get(k)

    def set(self, **kw):
        self.data.update(kw)
        self.save()

    def save(self):
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.persist:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
            except OSError:
                pass

    def clear(self):
        keep = {k: ([] if isinstance(v, list) else None) for k, v in self.data.items()}
        self.data.update(keep)
        self.save()

    # -- written after every tool call ---------------------------------------------
    def update_from_tool(self, tool, args, output):
        args = args or {}
        d = _json(output)
        if d is not None and d.get("error"):
            return
        if isinstance(output, str) and output.lstrip().startswith('{"error"'):
            return
        text = output if isinstance(output, str) else ""
        model = args.get("model") or args.get("model_name_or_path")
        if d is not None and d.get("use_as") and tool in ("fetch_biomodels_model", "save_sbml"):
            model = d["use_as"]          # a newly downloaded / saved model is the one in focus
        if model:
            if model != self.data["model"]:      # a new model: its parameters no longer apply
                self.data.update(bif_param=None, y_species=None)
            self.data["model"] = model
        m = re.search(r"bif_param=(\w+), y_species=(\w+)", text)
        bif = (args.get("bif_param") or args.get("param_name")
               or (d or {}).get("swept_parameter") or (m.group(1) if m else None))
        ys = (args.get("y_species") or args.get("plot_state")
              or (d or {}).get("plot_state") or (m.group(2) if m else None))
        if bif:
            self.data["bif_param"] = bif
        if ys:
            self.data["y_species"] = ys

        if d is not None and "bifurcation_points" in d:
            rng = d.get("swept_range")
            if model and d.get("swept_parameter") and isinstance(rng, (list, tuple)) and len(rng) == 2:
                # the widest range this session's diagrams swept, per model and parameter: an
                # LHS scan with no range of its own is run over it (agent.py, _diagram_range)
                key = diagram_range_key(model, d["swept_parameter"])
                seen = (self.data.get("diagram_ranges") or {}).get(key)
                lo, hi = float(rng[0]), float(rng[1])
                if seen:
                    lo, hi = min(lo, seen[0]), max(hi, seen[1])
                self.data["diagram_ranges"] = {**(self.data.get("diagram_ranges") or {}), key: [lo, hi]}
            cls = d.get("classification") or {}
            self.data["last_diagram"] = {
                "tool": tool, "model": model, "bif_param": d.get("swept_parameter"),
                "range": d.get("swept_range"), "plot_state": d.get("plot_state"),
                "points": [f"{p.get('type')} at {p.get(d.get('swept_parameter'))}"
                           for p in d.get("bifurcation_points", [])][:10],
                "category": cls.get("category"), "plot": d.get("plot_path")}
        elif "Bounds (lo, hi) per parameter" in text or "Summary CSV:" in text:
            grab = lambda pat: (re.search(pat, text).group(1).strip()
                                if re.search(pat, text) else None)
            self.data["last_scan"] = {
                "tool": tool, "model": model,
                "bif_param": bif, "y_species": ys, "n_samples": args.get("n_samples"),
                "categories": grab(r"Categories \(primary, per sample\): (.+)"),
                "csv": grab(r"Summary CSV: (.+)"), "plot": grab(r"Combined plot: (.+)"),
                "arguments": {k: v for k, v in args.items() if k != "model"}}
        if d is not None and tool in ("stochastic_simulation", "compare_simulations"):
            self.data["last_stochastic"] = {
                "model": model or d.get("model"), "omega": d.get("omega"),
                "methods": d.get("executed_methods", d.get("methods")),
                "requested_methods": d.get("requested_methods", d.get("methods")),
                "status": d.get("status"), "ssa_method": d.get("ssa_method"),
                "n_runs": d.get("n_runs"), "t_end": d.get("t_end"),
                "plot": d.get("plot_path"), "csv": d.get("csv"),
                "distribution_plot": d.get("distribution_plot_path"),
                "verdicts": d.get("stability_verdicts"),
                "matlab": d.get("matlab_script")}
        if d is not None and d.get("candidates") and tool == "propose_bifurcation_candidates":
            self.data["candidates"] = {"model": model, "list": d["candidates"][:6]}
        if d is not None and (d.get("use_as") and (d.get("saved_as") or d.get("path"))):
            _push(self.data["saved_models"], {"name": d["use_as"], "tool": tool,
                                              "from": d.get("derived_from") or d.get("model_id")},
                  "name")
        if d is not None and tool == "omnipath_network_check":
            self.data["network_check"] = {
                "model": model, "organism": d.get("organism"),
                "supported": sum(1 for e in d.get("edges", [])
                                 if str(e.get("status", "")).startswith("supported")),
                "n_edges": len(d.get("edges", [])),
                "missing": [f"{m['source']}->{m['target']}"
                            for m in (d.get("missing_internal_edges") or [])[:4]],
                "report": (d.get("excel_paths") or {}).get("report")}
        if d is not None and d.get("pmid") and d.get("title"):
            _push(self.data["papers"], {"pmid": d["pmid"], "title": d["title"][:120]}, "pmid")
        self.save()

    # -- read by the orchestrator ------------------------------------------------------------
    def as_text(self, limit=1500) -> str:
        d = self.data
        lines = []
        if d["model"]:
            lines.append(f"current model: {d['model']}")
        if d["bif_param"]:
            lines.append(f"for {d['model']} ONLY: continuation parameter {d['bif_param']}"
                         + (f", plotted species {d['y_species']}" if d["y_species"] else "")
                         + " (never apply these names to another model)")
        if d["last_diagram"]:
            ld = d["last_diagram"]
            lines.append(f"last diagram: {ld['model']} in {ld['bif_param']} over {ld['range']}, "
                         f"{ld['category']}, points {ld['points']}, plot {ld['plot']}")
        if d["last_scan"]:
            ls = d["last_scan"]
            lines.append(f"last scan: {ls['tool']} of {ls['model']}"
                         f", arguments {json.dumps(ls['arguments'], default=str)}, "
                         f"categories {ls['categories']}, CSV {ls['csv']}")
        if d.get("last_stochastic"):
            ls = d["last_stochastic"]
            lines.append(f"last stochastic run: {ls['model']} with {ls['methods']} at omega "
                         f"{ls['omega']}, t_end {ls['t_end']}, {ls['n_runs']} runs, plot {ls['plot']}"
                         + (f", verdicts {ls.get('verdicts')}" if ls.get("verdicts") else ""))
        if d["candidates"]:
            c = d["candidates"]
            lines.append(f"ranked candidates for {c['model']}: " + "; ".join(
                f"{i + 1}. {x['parameter']} {x.get('range')} -> {x.get('expected')}"
                for i, x in enumerate(c["list"])))
        if d["pending_approval"]:
            p = d["pending_approval"]
            lines.append(f"AWAITING APPROVAL: {p['tool']} on {p['model']} (candidates above)")
        if d["saved_models"]:
            lines.append("saved models: " + ", ".join(m["name"] for m in d["saved_models"]))
        if d.get("network_check"):
            n = d["network_check"]
            lines.append(f"OmniPath check of {n['model']}: {n['supported']}/{n['n_edges']} model "
                         f"edges supported; in OmniPath but not in the model: {n['missing']}")
        if d.get("last_run"):
            lines.append("last question's steps (what actually happened): " + "; ".join(
                f"{x['tool']} -> {x['status']}" + (f" ({x['issues'][:120]})" if x.get("issues") else "")
                for x in d["last_run"]))
        if d["papers"]:
            lines.append("papers: " + "; ".join(f"PMID {p['pmid']} {p['title']}"
                                               for p in d["papers"][-3:]))
        text = "\n".join(f"- {ln}" for ln in lines)
        return text[:limit] if text else ""
