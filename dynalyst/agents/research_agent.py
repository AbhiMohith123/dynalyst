"""
The agent: an LLM tool-use loop that has access to BOTH
  1. retrieve_knowledge  -> hybrid RAG search over papers + code (explanatory answers)
  2. list_available_models / get_model_odes / simulate_model -> real jaxkineticmodel
     execution (numeric answers)

The LLM decides per-query which tool(s) to call, possibly chaining them (e.g.
retrieve the API docs for a function, then call it).

Backend is OPENAI by default (config.LLM_BACKEND = "openai"). Set
OPENAI_API_KEY in your environment. An Anthropic backend is also included
(set config.LLM_BACKEND = "anthropic" and ANTHROPIC_API_KEY instead) —
useful if you ever want to compare the two.

Usage (mostly called from dynalyst/cli.py, but can be used standalone):
    from dynalyst.agents.research_agent import ask
    print(ask("Simulate the Garde2020 model and summarize the dynamics"))

(Formerly rag_jaxkm_bifurcation_workflow/agent.py. The tool-use loop below
is also reused by dynalyst/agents/unified_agent.py with a larger tool set;
its defaults are this agent's own prompt and tools.)
"""
import json
import os
import traceback

from dynalyst import config
from dynalyst.analysis import bifurcation_auto
from dynalyst.analysis.jaxkm_tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from dynalyst.analysis.xpp_to_sbml import (
    TOOL_DEFINITIONS as XPP_TOOL_DEFINITIONS,
    TOOL_FUNCTIONS as XPP_TOOL_FUNCTIONS,
)

SYSTEM_PROMPT = """You are a research assistant specialized in jaxkineticmodel
(a JAX/Diffrax-based package for building, simulating, and training kinetic
models of biological systems), its accompanying paper, the SBML benchmark
models it uses, and bifurcation/dynamical-systems analysis of those models.

You have three kinds of tools:
1. `retrieve_knowledge` — searches a knowledge base of the paper, its
   supplementary text, referenced papers, and the jaxkineticmodel source code.
   Use this for explanatory/conceptual questions (e.g. "why does it use a
   mean-centered loss", "what does AdaBelief do", "how does the SBML converter work").
2. `list_available_models`, `get_model_odes`, `simulate_model`,
   `plot_phase_plane`, `plot_rate_of_change`, `compare_initial_conditions` —
   call the REAL, installed jaxkineticmodel package (or a hand-coded
   fallback for models it can't import) and return real numeric simulation
   results. Use these whenever the user wants to actually simulate a model
   or see concentration dynamics over time.
3. The analysis pipeline — `get_odes`, `odes_to_sbml`, `find_steady_states`,
   `run_bifurcation_analysis`, `read_pdf` /
   `list_available_pdfs` — work generically for a model of ANY dimension
   (any SBML file, or a directly-specified system of ODEs), not just
   hand-coded models. This is the intended end-to-end workflow:

     get_odes  ->  find_steady_states  ->  run_bifurcation_analysis
       (extract/           (locate + classify         (real AUTO-07p
        derive the          stability of each          continuation:
        ODE system,         steady state via           detects/classifies
        any dimension)      Jacobian eigenvalues)       LP/BP/HB bifurcations,
                                                          plots the diagram)

   odes_to_sbml converts a directly-specified ODE system (species + odes you
   already have as plain Python data) into a real SBML file (usable by every
   other tool afterward, by name); get_odes already covers SBML->ODE
   extraction in the other direction. read_pdf / list_available_pdfs extract
   text from any paper (by path or name) for you to summarize or answer
   questions about — the tool only extracts; you do the actual summarizing.

   xpp_to_sbml converts an XPPAUT .ode file directly into SBML. If the user
   gives you a .ode file (or pastes XPP-format text: 'par ...' lines,
   NAME'=... derivative lines, '@' AUTO settings) and wants it converted,
   analyzed, or run through this pipeline, call xpp_to_sbml on it — do NOT
   read the file yourself and manually retype species/odes as arguments to
   odes_to_sbml. That manual step is unreliable (it has previously produced
   mismatched placeholder models) and is exactly what xpp_to_sbml replaces.
   Only fall back to typing out species/odes by hand for a system the user
   describes in prose rather than an actual .ode file.

Never fabricate numeric results, ODEs, steady states, or bifurcation
diagrams yourself — always call the appropriate tool. Never fabricate the
species/odes of a .ode file by "reading and retyping" it either — that's
what xpp_to_sbml is for.

TOOL SELECTION for the analysis pipeline:
- Always call `get_odes` first when the user asks about "the ODEs", "the
  equations", or a model's structure — for an SBML file OR a system typed
  in directly (species + odes). This also surfaces the model's actual free
  parameters, which run_bifurcation_analysis needs by exact
  name (never guess a parameter name; check get_odes or the error message).
- `get_odes` and `find_steady_states` are useful diagnostics, but they are
  NOT mandatory prerequisites for `run_bifurcation_analysis`. If the user
  directly asks for a bifurcation diagram from an SBML model, call
  `run_bifurcation_analysis` directly; it is self-contained and performs
  model resolution, conservation reduction, starting-equilibrium search,
  AUTO continuation, branch switching, and plotting internally.
- Use `get_odes`/`find_steady_states` first when they materially help identify
  the intended parameter/range or diagnose a failed/ambiguous run, but do not
  force a multi-tool chain for every direct diagram request.
- If the user asks for a bifurcation analysis/diagram of an uploaded `.txt` or `.ode` XPP model, call `run_bifurcation_analysis` with that exact filename directly. Do NOT first convert it with `xpp_to_sbml`; the bifurcation tool accepts XPP text input and carries the exact file through conversion itself.
- Never invent, reuse, or substitute a model name from a previous turn. The filename in the user's current request is authoritative.

- `run_bifurcation_analysis` is the authoritative tool: real AUTO-07p
  pseudo-arclength continuation, works for any dimension, detects and
  classifies LP (saddle-node/fold), BP (transcritical/pitchfork), and HB
  (Hopf) bifurcations from the continuation itself. Use this whenever the
  user asks to "find the bifurcations", "classify the bifurcation", or
  wants the actual bifurcation diagram.
- If the user gives you a system of ODEs directly (not an SBML file), pass
  species/odes/parameters/initial_conditions to get_odes, find_steady_states,
  or run_bifurcation_analysis directly — they all accept
  this the same way SBML models are accepted. Use `odes_to_sbml` if the user
  additionally wants a reusable SBML file out of it.
- If the user uploads or references a paper, use `read_pdf` (or
  `list_available_pdfs` if the filename is unclear) to read it, then
  summarize or answer from the extracted text yourself. Don't guess at a
  paper's content without reading it.

TOOL SELECTION for simulation/plotting:
- User names specific species to plot (e.g. "just plot S1") -> pass `species`
  to simulate_model (or compare_initial_conditions). Do not plot everything
  and call it close enough.
- User gives their OWN starting concentrations, or wants MULTIPLE starting
  conditions compared -> `compare_initial_conditions`, not repeated
  simulate_model calls.
- User wants ONE simulation from a custom starting point -> simulate_model
  with `initial_conditions`.
- User wants one species plotted against another (not against time) ->
  `plot_phase_plane`. Works for ANY model.
- User wants a "rate plot", "flux plot", "rate balance", or "production vs
  consumption" -> `plot_rate_of_change`. Works for ANY model and gives NET
  rate per state/species, not individual per-reaction fluxes.

GENERAL:
- Before calling any tool with a species name or parameter name, check the
  model's actual species/parameters (via get_odes/get_model_odes or the
  tool's error message) rather than guessing — SBML files and hand-coded
  models don't all share the same names.
- If a tool call fails because of a wrong argument (unknown species, wrong
  parameter name), read the error message, correct the call, and retry
  automatically rather than giving up or asking the user to rephrase —
  the error messages are written to tell you exactly what's valid.

Cite which source (paper section, code file, or "live simulation output") each
part of your answer comes from. If a query needs both explanation and execution,
use both tool types and synthesize.

CRITICAL: When an XPP/TXT/ODE source is supplied, report the actual XPP AUTO
metadata used by the tool (including parmin/parmax/autovar) rather than guessing.

CRITICAL: For every run_bifurcation_analysis call, set show_auto_log=True.
The complete AUTO-07p terminal output is part of the normal result. Surface it
automatically; do not require the user to ask for it. The terminal output must
include AUTO's continuation records (BR/PT/TY/LAB/L2-NORM/U(...)) whenever
provided by the tool. If the result has "auto_terminal_log" or
"auto_stdout_log", reproduce that output in the final response after the
bifurcation summary. Never replace the AUTO terminal output with a short
summary when the user asked to see the output. If the result has "plot_path",
report that exact plot path as a Markdown image.

CRITICAL BIFURCATION TOPOLOGY: A special point (LP/BP/HB) is a point ON a
continuation trajectory, not automatically the endpoint of that trajectory. A
single model may contain multiple bifurcations on the same overall continuation,
e.g. a saddle-node/LP and a transcritical/branch-point BP. Do not infer that a
trajectory ends at the first special point. The complete AUTO continuation data
are authoritative for trajectory geometry; the "bifurcation_points" list only
identifies special points.

CRITICAL: If run_bifurcation_analysis returns a tool/implementation error,
report the actual error so the AUTO pipeline can be fixed.

CRITICAL: Never describe "starting_equilibrium" as a bifurcation point.
The "bifurcation_points" array is the ONLY authoritative source for detected
bifurcations. If it is empty, say that no bifurcation point was detected.
"starting_equilibrium" is only the equilibrium used to initialize continuation
and must never be relabeled as a special point.

CRITICAL: If a tool call returns an error (e.g. a FileNotFoundError, missing
model, or any JSON containing an "error" key), you MUST report that error
plainly to the user. Never paper over a failed or missing tool result with
plausible-sounding invented content — if you don't have real data, say so
explicitly rather than describing equations, parameters, or values you did
not actually retrieve.

CRITICAL: You cannot generate, draw, or embed images/plots yourself. The ONLY
way a plot exists is if a tool result's JSON contains a "plot_path" field
(returned by simulate_model, plot_phase_plane, plot_rate_of_change,
compare_initial_conditions, or run_bifurcation_analysis).
If a bifurcation tool result contains "plot_path", ALWAYS include that exact
path as a Markdown image in the final answer. Do not merely say that a plot
was saved. If a tool result has no "plot_path", do NOT invent a plot or image.
"""

RETRIEVE_TOOL_DEF = {
    "name": "retrieve_knowledge",
    "description": (
        "Search the knowledge base (paper text + supplementary info + referenced "
        "papers + jaxkineticmodel source code) for information relevant to a query. "
        "Use for conceptual/explanatory questions, not for running simulations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "prefer_code": {
                "type": "boolean",
                "description": "Set true if the question is specifically about implementation/API details.",
            },
        },
        "required": ["query"],
    },
}

# Canonical tool definitions, written in Anthropic's schema shape
# ({name, description, input_schema}). Converted to OpenAI's shape below.
ALL_TOOL_DEFINITIONS = TOOL_DEFINITIONS + XPP_TOOL_DEFINITIONS
ALL_TOOL_FUNCTIONS = {**TOOL_FUNCTIONS, **XPP_TOOL_FUNCTIONS}

ALL_TOOLS_ANTHROPIC = [RETRIEVE_TOOL_DEF] + ALL_TOOL_DEFINITIONS


def _to_openai_tool(tool_def: dict) -> dict:
    """Convert an Anthropic-style tool def to OpenAI's function-calling shape."""
    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "parameters": tool_def["input_schema"],
        },
    }


ALL_TOOLS_OPENAI = [_to_openai_tool(t) for t in ALL_TOOLS_ANTHROPIC]


def _run_retrieve_tool(query: str, prefer_code: bool = False) -> str:
    # Imported here so the agent (and every non-RAG tool) still works when
    # the RAG stack (chromadb, sentence-transformers) is not installed.
    from dynalyst.rag.retrieve import retrieve
    if config.AUTO_INDEX_MODELS:      # PDFs uploaded since the last search: index them first
        try:
            from dynalyst.rag.model_index import sync_papers
            sync_papers()
        except Exception:
            pass
    collections = [config.COLLECTION_CODE] if prefer_code else None
    hits = retrieve(query, collections=collections)
    formatted = [
        {
            "source": hit["metadata"].get("source_name"),
            "source_type": hit["metadata"].get("source_type"),
            "section_or_unit": hit["metadata"].get("section") or hit["metadata"].get("unit_name"),
            "text": hit["text"],
        }
        for hit in hits
    ]
    return json.dumps(formatted, indent=2)


def _print_bifurcation_auto_log(result_text: str) -> None:
    """Print the complete AUTO terminal transcript for interactive CLI runs."""
    try:
        result = json.loads(result_text)
    except Exception:
        return

    log = result.get("auto_terminal_log") or result.get("auto_stdout_log")
    if not log:
        return

    print("\n" + "=" * 80)
    print("AUTO-07p TERMINAL OUTPUT")
    print("=" * 80)
    print(log.rstrip())
    print("=" * 80 + "\n")


def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "retrieve_knowledge":
        # Same error contract as every other tool (a JSON "error" the model
        # reports), instead of an exception that ends the whole conversation
        # -- e.g. when the index hasn't been built on this machine yet.
        try:
            return _run_retrieve_tool(**tool_input)
        except Exception as e:
            return json.dumps({
                "error": f"retrieve_knowledge failed: {e}",
                "hint": "Build the knowledge-base index first (README: ingest + build_index).",
                "tool": name,
                "traceback": traceback.format_exc(),
            })

    # IMPORTANT: dispatch bifurcation analysis directly through the currently
    # imported bifurcation_auto module. This prevents the Ask path from using
    # a stale function object cached by jaxkm_tools after bifurcation_auto.py
    # was replaced/reloaded.
    if name == "run_bifurcation_analysis":
        try:
            return bifurcation_auto.run_bifurcation_analysis(**tool_input)
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tool": "run_bifurcation_analysis",
                "implementation": getattr(
                    bifurcation_auto.run_bifurcation_analysis,
                    "__code__", None
                ).co_filename if getattr(
                    bifurcation_auto.run_bifurcation_analysis, "__code__", None
                ) else getattr(bifurcation_auto, "__file__", None),
            })

    if name in ALL_TOOL_FUNCTIONS:
        try:
            return ALL_TOOL_FUNCTIONS[name](**tool_input)
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "tool": name,
                "traceback": traceback.format_exc(),
            })
    return json.dumps({"error": f"Unknown tool: {name}"})


MAX_TOOL_TURNS = 12


def _ask_openai(user_message: str, conversation_history: list[dict] | None, verbose: bool,
                system_prompt: str = SYSTEM_PROMPT, tools_anthropic: list | None = None,
                execute=None) -> str:
    from openai import OpenAI

    tools_openai = ALL_TOOLS_OPENAI if tools_anthropic is None else [
        _to_openai_tool(t) for t in tools_anthropic]
    execute = execute or _execute_tool
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    for _tool_turn in range(MAX_TOOL_TURNS):
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=config.MAX_TOKENS,
            tools=tools_openai,
            messages=messages,
        )
        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content or ""

        # Model wants to use one or more tools
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_input = json.loads(tc.function.arguments or "{}")

            if verbose:
                print(f"  [tool call] {tool_name}({json.dumps(tool_input)})")
            result_text = execute(tool_name, tool_input)
            if verbose:
                if tool_name == "run_bifurcation_analysis":
                    _print_bifurcation_auto_log(result_text)
                preview = result_text[:400] + ("..." if len(result_text) > 400 else "")
                print(f"  [tool result] {preview}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })
        # loop continues: send tool results back to the model for the next step

    return (
        "Tool-use limit reached before the model produced a final answer. "
        f"Stopped after {MAX_TOOL_TURNS} tool rounds to avoid an unbounded retry loop."
    )


def _ask_anthropic(user_message: str, conversation_history: list[dict] | None, verbose: bool,
                   system_prompt: str = SYSTEM_PROMPT, tools_anthropic: list | None = None,
                   execute=None) -> str:
    import anthropic

    tools_anthropic = ALL_TOOLS_ANTHROPIC if tools_anthropic is None else tools_anthropic
    execute = execute or _execute_tool
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages = list(conversation_history) if conversation_history else []
    messages.append({"role": "user", "content": user_message})

    for _tool_turn in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            tools=tools_anthropic,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if verbose:
                print(f"  [tool call] {block.name}({json.dumps(block.input)})")
            result_text = execute(block.name, block.input)
            if verbose:
                if block.name == "run_bifurcation_analysis":
                    _print_bifurcation_auto_log(result_text)
                preview = result_text[:400] + ("..." if len(result_text) > 400 else "")
                print(f"  [tool result] {preview}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })
        messages.append({"role": "user", "content": tool_results})
        # loop continues: send tool results back to Claude for the next step

    return (
        "Tool-use limit reached before the model produced a final answer. "
        f"Stopped after {MAX_TOOL_TURNS} tool rounds to avoid an unbounded retry loop."
    )


def ask(user_message: str, conversation_history: list[dict] | None = None, verbose: bool = True) -> str:
    """Dispatches to the configured backend (config.LLM_BACKEND)."""
    if config.LLM_BACKEND == "openai":
        return _ask_openai(user_message, conversation_history, verbose)
    elif config.LLM_BACKEND == "anthropic":
        return _ask_anthropic(user_message, conversation_history, verbose)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r} (use 'openai' or 'anthropic')")


# ---------------------------------------------------------------------------
# OPTIONAL: running a local open-source LLM on your GPU instead of the API.
# You lose Claude's native tool-use ergonomics -- you'd need to prompt the
# model to emit a JSON tool call, parse it yourself, then feed the result back
# in the next turn. Sketch (not wired up by default):
#
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
#   model = AutoModelForCausalLM.from_pretrained(
#       "Qwen/Qwen2.5-14B-Instruct", device_map="cuda", torch_dtype="auto"
#   )
#   # then: build a prompt that lists ALL_TOOLS as text, ask the model to reply
#   # with {"tool": "...", "input": {...}} or a final answer, parse with json.loads,
#   # call _execute_tool(...) yourself, and append the result to the prompt.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What SBML models are available and what does jaxkineticmodel do?"
    print(ask(q))


