"""Central configuration for the merged project.

Combines the former rag_jaxkm_bifurcation_workflow/config.py (RAG index,
LLM backend, jaxkineticmodel + AUTO-07p locations) with the paths the
bifurcation agent (agent/) used to hard-code next to its own source files
(pdp_mm_bistable.txt, the reference SBML files, workdir/, scan_memory.jsonl).

Every path is relative to the repository root, so the project works from
wherever it is checked out. AUTO_DIR can still be pointed at an existing
AUTO-07p install with the AUTO_DIR environment variable.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Device (detected lazily: importing torch is slow, and only the RAG
#     embedding/rerank code needs it) ---
def _detect_device() -> str:
    override = os.environ.get("DYNALYST_DEVICE")
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def __getattr__(name):
    # PEP 562: `config.DEVICE` is computed on first access, not at import.
    if name == "DEVICE":
        value = _detect_device()
        globals()["DEVICE"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Data locations (RAG knowledge base) ---
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CODE_DIR = DATA_DIR / "code"
DOCS_DIR = DATA_DIR / "docs_html"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = CHROMA_PERSIST_DIR / "bm25_index.pkl"

# --- Chunking ---
PDF_CHUNK_TOKENS = 400          # approx tokens per prose chunk
PDF_CHUNK_OVERLAP_TOKENS = 60
MIN_CODE_CHUNK_CHARS = 40       # skip trivial functions/snippets

# --- Embeddings ---
# General purpose text embedding model (good quality, runs fine on a single GPU)
TEXT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# Code-aware embedding model, better for matching queries like "simulate function"
# against actual function bodies
CODE_EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
EMBED_BATCH_SIZE = 32

# --- Retrieval ---
TOP_K_DENSE = 15
TOP_K_BM25 = 15
TOP_K_FINAL = 6                 # after fusion/rerank, how many chunks go to the LLM
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-large"

# --- Collections ---
COLLECTION_PAPERS = "jkm_papers"
COLLECTION_CODE = "jkm_code"
COLLECTION_DOCS = "jkm_docs"
COLLECTION_MODELS = "jkm_models"   # models the user saved (rag/model_index.py)
# Add every model the user saves (BioModels download, XPP / ODE conversion) to the
# knowledge base at once. Off in the test suite.
AUTO_INDEX_MODELS = os.environ.get("DYNALYST_AUTO_INDEX_MODELS", "1") == "1"
# Human approval before AUTO-07p when no continuation parameter was named is not a
# setting: it follows from who is using the agent. The chat entry points (cli.py,
# notebook.ask, chat_ui.py) create it with approval="ask" -- a person is there to pick a
# candidate; called from code (scripts, evaluations, tests) it is "auto" and runs the
# top candidate (orchestration/approval.py).
APPROVAL_EXPIRES_S = 3600      # an unanswered approval older than this is dropped

# --- AUTO-07p numerics bounds (analysis/bifurcation_auto.py, reviewer retries) ---
AUTO_DEFAULT_POINTS = 500       # n_pts when the caller gives none
AUTO_MAX_POINTS = 20000         # n_pts is capped here (NMX = 40 x n_pts)
AUTO_MAX_NMX = 400000           # largest step budget a run or a retry may ask for
AUTO_MIN_DSMIN = 1e-12          # smallest step a run or a retry may ask for

# --- Model sanity check before AUTO (analysis/model_check.py) ---
SANITY_MIN_EIG = 1e-9           # largest |eigenvalue| below this: the equations have no scale
SANITY_MAX_RESIDUAL = 1e-4      # relative residual above this: the settled state is not steady

# --- Parameter screen (analysis/parameter_screen.py) ---
SCREEN_SAME_BRANCH_REL = 0.3      # largest relative state change per small step on one branch
SCREEN_RUNAWAY_DECADES = 1.0      # a state changing >10x before the edge: runaway, not a fold
SCREEN_AT_VALUE_DECADES = 0.005   # an event within ~1% of the model's own value: suspect

# --- Agent / generation ---
# Backend for the research agent and the unified agent: "openai" or "anthropic"
LLM_BACKEND = "openai"

OPENAI_MODEL = "gpt-4o-mini"           # verify access with: openai.OpenAI().models.list()
ANTHROPIC_MODEL = "claude-sonnet-5"
MAX_TOKENS = 2000

# --- The agent (orchestrator -> teams -> sub-agents -> guard -> tool -> final check) ---
# Any LangChain "provider:model" string works, e.g. "openai:gpt-4o-mini",
# "anthropic:claude-sonnet-5", "google_genai:gemini-2.5-pro",
# "ollama:qwen2.5:14b" (the matching langchain-<provider> package must be
# installed). DYNALYST_MODEL overrides it; the per-role settings override it
# for one role only (None = use AGENT_MODEL).
AGENT_MODEL = os.environ.get("DYNALYST_MODEL", "openai:gpt-4o-mini")
ORCHESTRATOR_MODEL = os.environ.get("DYNALYST_ORCHESTRATOR_MODEL")
SUBAGENT_MODEL = os.environ.get("DYNALYST_SUBAGENT_MODEL")
# The orchestrator writes and revises its own plan (write_plan); at most this many versions.
MAX_PLAN_VERSIONS = 4
AGENT_TEMPERATURE = 0.0
# Some models (reasoning models) accept only their default temperature, and only the
# Responses API for tool calls: DYNALYST_TEMPERATURE=default sends no temperature,
# DYNALYST_RESPONSES_API=1 uses OpenAI's Responses API instead of Chat Completions.
_TEMPERATURE = os.environ.get("DYNALYST_TEMPERATURE")
if _TEMPERATURE is not None:
    AGENT_TEMPERATURE = (None if _TEMPERATURE.strip().lower() in ("", "default", "none")
                         else float(_TEMPERATURE))
RESPONSES_API = os.environ.get("DYNALYST_RESPONSES_API", "0") == "1"
# Reasoning effort for reasoning models (sent only with the Responses API): one value for
# every role ("low"), or per role ("orchestrator=low,team=low,planner=medium,subagent=medium").
REASONING_EFFORT = os.environ.get("DYNALYST_REASONING_EFFORT", "")
# Extra keyword arguments for every chat model, as a JSON object, passed unchanged to
# LangChain's init_chat_model. Examples: a Qwen3 model served by vLLM without thinking,
# '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'; an Ollama model
# with a longer context, '{"num_ctx": 32768}'.
MODEL_KWARGS = json.loads(os.environ.get("DYNALYST_MODEL_KWARGS") or "{}")
if not isinstance(MODEL_KWARGS, dict):
    raise ValueError("DYNALYST_MODEL_KWARGS must be a JSON object, e.g. '{\"num_ctx\": 32768}'")
# Lessons (orchestration/lessons.py): a call that failed and then succeeded with other
# arguments is saved as data (tool, kind of problem, which argument names changed;
# never a model, parameter name or value) and shown as a hint to later sub-agents using that
# tool. DYNALYST_LESSONS=0 turns it off.
LESSONS_ENABLED = os.environ.get("DYNALYST_LESSONS", "1") == "1"
LESSONS_MAX_AGE_DAYS = 60
LESSONS_MAX_ENTRIES = 300
LESSONS_PER_TASK = 4

# Budgets (each can be set with the environment variable DYNALYST_<NAME>, e.g.
# DYNALYST_ORCHESTRATOR_MAX_CALLS=20)
ORCHESTRATOR_MAX_CALLS = int(os.environ.get("DYNALYST_ORCHESTRATOR_MAX_CALLS", 14))  # orchestrator calls per query
SUBAGENT_MAX_CALLS = int(os.environ.get("DYNALYST_SUBAGENT_MAX_CALLS", 6))  # per delegation
# A tool that failed is retried at most this many times per request (the same failed call
# never), then the error goes back up to the orchestrator.
MAX_TOOL_RETRIES = int(os.environ.get("DYNALYST_MAX_TOOL_RETRIES", 2))
MAX_DIAGRAMS_PER_PARAMETER = int(os.environ.get("DYNALYST_MAX_DIAGRAMS_PER_PARAMETER", 3))
# The reviewer (orchestration/reviewer.py) re-checks every result; a result it sends back
# with a correction is re-selected at most this many times per task.
REVIEWER_MAX_REPLANS = int(os.environ.get("DYNALYST_REVIEWER_MAX_REPLANS", 3))

# --- Architecture (README.md, "How it works"): orchestrator -> team supervisors -> sub-agents ---
TEAM_MODEL = os.environ.get("DYNALYST_TEAM_MODEL")     # None = AGENT_MODEL
TEAM_MAX_CALLS = int(os.environ.get("DYNALYST_TEAM_MAX_CALLS", 6))  # team-supervisor calls per team delegation
# Tasks one sub-agent may be given in one request (a supervisor that keeps re-asking is
# stopped and reports what it has).
MAX_TASKS_PER_SUBAGENT = int(os.environ.get("DYNALYST_MAX_TASKS_PER_SUBAGENT", 3))
# Analyses one delegation may run (each through the guard).
SUBAGENT_MAX_ANALYSES = int(os.environ.get("DYNALYST_SUBAGENT_MAX_ANALYSES", 3))
MAX_RUNS_PER_OMEGA = 2      # stochastic runs at one system size in a request: one, and one retry
# past this time a person chatting is asked whether to go on
STOCHASTIC_ASK_SECONDS = float(os.environ.get("DYNALYST_STOCHASTIC_ASK_SECONDS", 12 * 60))
# Delegations a supervisor issues in the same turn run concurrently. Tool execution itself
# is serialised except for the network look-ups in tools.PARALLEL_SAFE: most analysis tools
# plot with matplotlib's global state and the regime pipeline changes the working directory.
PARALLEL = True
MAX_PARALLEL = 4                # delegations running at the same time, at most
# Stochastic simulation (analysis/stochastic.py). Wall-clock budget for one exact SSA (all
# runs together): over it the numpy engine is replaced by the compiled JAX engine when jax
# is installed, else the SSA is skipped with an estimate and the options (ssa_method='exact'
# runs numpy anyway, up to the forced limit). SSA_MAX_EVENTS_PER_RUN_JAX caps the events of
# one run (both engines); a run that hits it stops early and is reported as truncated.
SSA_MAX_SECONDS = float(os.environ.get("DYNALYST_SSA_MAX_SECONDS", 300))
# under this many molecules a non-elementary rate law is flagged: a choice, not derived
SSA_FEW_MOLECULES = float(os.environ.get("DYNALYST_SSA_FEW_MOLECULES", 50))
SSA_REFERENCE_ODE_SECONDS = float(os.environ.get("DYNALYST_SSA_REFERENCE_ODE_SECONDS", 30))
# after this long a run projects its own full time, and stops if that is STOP_FACTOR call limits or more
SSA_PILOT_SECONDS = float(os.environ.get("DYNALYST_SSA_PILOT_SECONDS", 10))
SSA_STOP_FACTOR = float(os.environ.get("DYNALYST_SSA_STOP_FACTOR", 3))
SSA_MAX_SECONDS_FORCED = float(os.environ.get("DYNALYST_SSA_MAX_SECONDS_FORCED", 3 * 3600))
SSA_MAX_EVENTS_PER_RUN_JAX = int(os.environ.get("DYNALYST_SSA_MAX_EVENTS_PER_RUN_JAX", 500_000_000))
# CRNT (analysis/crnt_analysis.py): crnt4sbml's search for rate constants with several steady
# states runs in a worker process stopped after this many seconds (its symbolic steps grow
# very fast with network size); 0 = no limit.
CRNT_MAX_SECONDS = int(os.environ.get("DYNALYST_CRNT_MAX_SECONDS", 600))
# A person chatting (approval="ask") is asked after this many seconds instead: let the search
# finish (then AUTO-07p checks its first candidate set), or stop it and draw the diagram.
CRNT_ASK_SECONDS = int(os.environ.get("DYNALYST_CRNT_ASK_SECONDS", 1200))
# ... and a search the user chose to wait for is stopped this many seconds after it started
CRNT_CONTINUE_MAX_SECONDS = int(os.environ.get("DYNALYST_CRNT_CONTINUE_MAX_SECONDS", 3 * 3600))
# Scratch folders (AUTO runs, LHS work folders, CRNT jobs; dynalyst/housekeeping.py) not
# touched for this many days are removed when an agent starts; 0 keeps everything.
SCRATCH_KEEP_DAYS = float(os.environ.get("DYNALYST_SCRATCH_KEEP_DAYS", 7))
# QSSA model reduction (analysis/model_reduction.py); 0 = only conservation laws are
# eliminated (the timescale analysis is still reported)
QSSA_ENABLED = os.environ.get("DYNALYST_QSSA", "1") == "1"
TOOL_OUTPUT_CHARS = 10000      # tool output shown to a language model (full output is kept in the run record)

# --- Vendored third-party code ---
VENDOR_DIR = PROJECT_ROOT / "vendor"
JKM_REPO_PATH = VENDOR_DIR / "jaxkineticmodel"
JKM_SBML_MODELS_DIR = JKM_REPO_PATH / "models" / "sbml_models"

# --- Model files ---
# SBML models downloaded/uploaded for the research tools (benchmark models,
# XPP uploads converted to SBML, odes_to_sbml output).
LOCAL_SBML_MODELS_DIR = DATA_DIR / "sbml_models"
# Reference model files shipped with the project (the textbook bifurcation models and
# XPP examples). Kept apart from data/sbml_models because some names clash with
# different models there.
MODELS_DIR = PROJECT_ROOT / "models"

# All directories searched (in order) when resolving a model by name.
# MODELS_DIR is last so names that already resolved in data/sbml_models keep
# resolving to the same file as before the merge.
SBML_SEARCH_DIRS = [LOCAL_SBML_MODELS_DIR, JKM_SBML_MODELS_DIR, MODELS_DIR]

# --- AUTO-07p (bifurcation continuation engine) ---
# Bundled source in vendor/auto-07p; build it with scripts/build_auto.sh.
# Set the AUTO_DIR environment variable to use an existing install instead.
AUTO_DIR = Path(os.environ.get("AUTO_DIR") or (VENDOR_DIR / "auto-07p"))

# --- Outputs ---
# DYNALYST_OUTPUT_DIR overrides the location (the test suite uses it so test
# runs never touch outputs/). Keep it inside PROJECT_ROOT: the research tools
# report plot paths relative to the project root.
OUTPUT_DIR = Path(os.environ.get("DYNALYST_OUTPUT_DIR") or (PROJECT_ROOT / "outputs"))
# Research tools: simulation / bifurcation / scan plots.
PLOTS_DIR = OUTPUT_DIR / "plots"
LESSONS_PATH = OUTPUT_DIR / "lessons.jsonl"
# Research tools: one scratch dir per AUTO run (removed after each run).
AUTO_WORK_DIR = OUTPUT_DIR / "auto_runs"
# LHS / AUTO pipeline (dynalyst.regime): one subdirectory per run / LHS batch.
WORKDIR = OUTPUT_DIR / "workdir"

def best_name_match(query: str, paths) -> "Path | None":
    """The file whose name matches `query` best: an exact name (without its
    extension) first, then names starting with it, then names containing it,
    shortest first -- so "Smith" is Smith.pdf, not Goldsmith.pdf, and
    "model_a" is not model_a_extended."""
    q = query.lower()
    paths = list(paths)
    for test in (lambda s: s == q, lambda s: s.startswith(q), lambda s: q in s):
        hits = sorted((p for p in paths if test(p.stem.lower())), key=lambda p: (len(p.stem), str(p)))
        if hits:
            return hits[0]
    return None


def existing_file(name) -> "Path | None":
    """`name` as an existing file: as given, or relative to the project root
    (so "models/pitchfork.xml" works whatever the working directory)."""
    for p in (Path(name), PROJECT_ROOT / str(name)):
        if p.is_file():
            return p
    return None


def ensure_unambiguous(query: str, dirs=None, suffixes=(".xml",)):
    """Raise ValueError when the bare name `query` (with or without its
    extension) is the exact name of DIFFERENT files in the model folders --
    e.g. pitchfork.xml, a 1D toy model in data/sbml_models and the 2D PDP
    model in models/. Silently taking whichever folder is searched first gives
    different tools different models for the same name. Identical copies are fine."""
    stem = Path(str(query)).stem.lower() if Path(str(query)).suffix.lower() in suffixes \
        else str(query).lower()
    seen = {}
    for root in (dirs or SBML_SEARCH_DIRS):
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes and p.stem.lower() == stem:
                seen.setdefault(p.read_bytes(), p)
    if len(seen) > 1:
        paths = [display_path(p) for p in seen.values()]
        raise ValueError(
            f"'{query}' is the name of {len(paths)} different models: {', '.join(paths)}. "
            f"Say which one by passing its path, e.g. '{paths[0]}'.")


def display_path(path) -> str:
    """A path relative to the project when it lies inside it, else absolute
    (DYNALYST_OUTPUT_DIR may point anywhere)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
