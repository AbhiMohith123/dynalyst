"""Write one check notebook per capability (bifurcation, CRNT, simulation, ...) into
notebooks/tools/, with a section per main tool of that capability.

Each section runs ONE tool directly, without the agent, on inputs YOU give (no model or
value is filled in), so a problem seen in a chat can be traced: run the tool with the same
arguments the agent used (paste them from the log). If the tool misbehaves here, the tool
is the cause; if it works here but not in the chat, the agent is.

Rerun after adding or changing tools:  python scripts/make_tool_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "notebooks" / "tools"

# capability -> its main tools
FAMILIES = {
    "bifurcation": ["run_bifurcation_analysis", "propose_bifurcation_candidates",
                    "lhs_bifurcation_scan"],
    "crnt": ["crnt_bistability", "draw_reaction_network"],
    "simulation": ["simulate_model", "compare_initial_conditions", "plot_phase_plane", "custom_plot"],
    "steady_state": ["find_steady_states", "jacobian_network"],
    "stochastic": ["stochastic_simulation", "propensity_table"],
    "model_structure": ["classify_model", "reduce_model"],
    "fitting": ["inspect_dataset", "estimate_parameters"],
    "biomodels": ["search_biomodels", "get_biomodels_info", "fetch_biomodels_model",
                  "find_biomodels_by_paper"],
    "retrieval": ["search_literature", "get_pubmed_article", "retrieve_knowledge", "read_pdf"],
    "omnipath": ["omnipath_network_check", "model_proteins"],
}

SETUP = '''import os, sys, time, json
ROOT = os.path.abspath(os.path.join("..", ".."))      # notebooks/tools/ -> project root
sys.path.insert(0, ROOT)
from dynalyst.orchestration import tools as registry
from IPython.display import Image, display


def arguments(tool):
    """The tool's arguments, read from the tool itself."""
    schema = registry.schema_of(registry.all_tools()[tool])
    print(tool, "-- required:", schema.get("required", []))
    for name, spec in schema.get("properties", {}).items():
        print(f"  {name} ({spec.get('type', 'any')}): {(spec.get('description') or '')[:150]}")


def run_tool(tool, args, pasted=""):
    """Run one tool alone: your ARGS, or the agent's call pasted from its log."""
    if pasted.strip():
        args = json.loads(pasted)
    print(tool, args)
    t = time.time()
    raw = registry.execute(tool, args)
    return show(tool, raw, time.time() - t)


def show(tool, raw, seconds):
    """The result: error first, then each field (long ones shortened), then plots."""
    print(f"{tool} ran in {seconds:.1f} s")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        print(str(raw)[:3000])
        return None
    if isinstance(data, dict):
        if data.get("error"):
            print("ERROR:", data["error"])
        for key, value in data.items():
            text = json.dumps(value, default=str)
            print(f"- {key}: {text[:400]}{' ...' if len(text) > 400 else ''}")
        for key, value in data.items():
            if isinstance(value, str) and value.lower().endswith(".png"):
                path = value if os.path.isabs(value) else os.path.join(ROOT, value)
                if os.path.exists(path):
                    print(key)
                    display(Image(filename=path))
    else:
        print(json.dumps(data, default=str)[:3000])
    return data'''

SECTION = '''arguments("{tool}")
# Fill in what you need (required: {required}), OR paste the agent's call from its log
# (the {{...}} after "SELECT {tool}(") into PASTED.
ARGS = {{
{lines}
}}
PASTED = ""
result = run_tool("{tool}", ARGS, PASTED)'''

AGENT = '''# Optional: ask the agent the same thing and compare (needs the API key).
QUERY = ""                       # e.g. the request that went wrong
if QUERY.strip():
    from dynalyst.agents.unified_agent import run
    t = time.time()
    rec = run(QUERY)
    print(f"agent: {time.time() - t:.1f} s\\n")
    calls = [s for s in rec.steps if s.get("tool") in TOOLS]
    print(f"the agent made {len(calls)} call(s) to these tools:")
    for a in calls:
        print(" ", a.get("tool"), json.dumps(a.get("arguments"), default=str)[:400],
              "|", a.get("status"), (a.get("error") or "")[:200])
    print("\\nanswer:\\n", rec.final_answer[:2000])'''

HOW = '''**How to read it**
- The tool fails or gives a wrong result here, with the agent's arguments: the **tool** is the cause.
- The tool works here, but the chat went wrong: the **agent** is the cause (wrong arguments,
  wrong tool, or a wrong reading of the result). Compare with the agent's calls above.
- The tool works here only with different arguments: the agent passed wrong arguments.'''


def notebook(family: str, tools: list, registry) -> dict:
    cells = [("markdown", f"# Tool check: {family.replace('_', ' ')}\n\n"
                          "Runs each tool of this capability **directly, without the agent**, on the "
                          "inputs you give. One section per tool: run the setup, then only the "
                          "section you need."),
             ("code", SETUP + f"\n\nTOOLS = {tools!r}")]
    for tool in tools:
        schema = registry.schema_of(registry.all_tools()[tool])
        props, required = schema.get("properties", {}), schema.get("required", [])
        lines = []
        for name, p in props.items():
            desc = (p.get("description") or "").replace("\n", " ")[:100]
            mark = "  REQUIRED" if name in required else ""
            lines.append(f'    # "{name}": ...,   # {p.get("type", "any")}{mark}: {desc}')
        cells.append(("markdown", f"## `{tool}`\n{(schema.get('description') or '').strip()[:400]}"))
        cells.append(("code", SECTION.format(tool=tool, required=required, lines="\n".join(lines))))
    cells += [("markdown", "## The agent on the same request (optional)"), ("code", AGENT),
              ("markdown", HOW)]
    return {"cells": [{"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
                       **({"execution_count": None, "outputs": []} if kind == "code" else {})}
                      for kind, src in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    from dynalyst.orchestration import tools as registry
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        if old.stem not in FAMILIES and old.stem != "crnt_bistability_stages":
            old.unlink()
    index = ["# Tool check notebooks", "",
             "One notebook per capability: run a tool alone on your inputs (or the agent's pasted "
             "call) to see whether a problem comes from the tool or from the agent.", ""]
    for family, tools in FAMILIES.items():
        tools = [t for t in tools if t in registry.all_tools()]
        (OUT / f"{family}.ipynb").write_text(json.dumps(notebook(family, tools, registry), indent=1),
                                             encoding="utf-8")
        index.append(f"- [{family}]({family}.ipynb): {', '.join(tools)}")
    index.append("- [crnt_bistability_stages](crnt_bistability_stages.ipynb): time each crnt4sbml stage")
    (OUT / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"{len(FAMILIES)} notebooks in {OUT}")


if __name__ == "__main__":
    main()
