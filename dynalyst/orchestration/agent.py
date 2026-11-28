"""The agent: one orchestrator that plans, delegates and answers -> team supervisors ->
sub-agents -> execution guard -> analysis tool -> reviewer -> final-answer check.

Language models make every decision about the request: what the steps are (the
orchestrator writes its own plan with write_plan, and writes it again to revise it),
who does each step, which tool with which arguments, what to do when a tool reports an
error, and what the answer says. Code adds only a fixed list of exceptions, each for a
reason a prompt cannot guarantee:

  1. file safety      no tool replaces, deletes or writes outside its folders (safety.py)
  2. permission       a sub-agent calls only its own tools, with valid arguments (guard.py)
  3. retry limit      a failing tool is retried at most config.MAX_TOOL_RETRIES times
  4. budgets          model calls per role, plan versions
  5. duplicates       the same call in one request reuses its result
  6. asking the user in a chat, the run pauses before an AUTO-07p run with no
                      parameter named and when a CRNT search runs long (approval.py)
  7. final check      every number, model ID and paper title in the answer appears in a
                      tool result of this request, and its conclusions are ones an
                      analysis reached (final_check.py)
  8. reviewer         every result is re-checked from the model's own equations (reported
                      bifurcation points are steady states with the right eigenvalues,
                      simulations stay finite and non-negative, fits are not on a bound);
                      a failure goes back to the sub-agent with a correction (reviewer.py)

Context is scoped: the orchestrator sees the user's message, the recent conversation and
the session's facts; a team supervisor sees only the orchestrator's task; a sub-agent
sees only its supervisor's task.
"""
import itertools
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from dynalyst import config
from dynalyst.orchestration import tools as registry
from dynalyst.orchestration import approval, lessons
from dynalyst.orchestration.final_check import (blocks_with, claim_patterns, evidence_numbers,
                                                keeps_content, model_identities,
                                                placeholders, qualify_empty_continuation,
                                                split_blocks, stability_claim_without_check,
                                                strip_model_ids, tidy_answer,
                                                undecided_absence_claims, unfound_bifurcation_claim,
                                                unsupported_points,
                                                unfound_hopf_claim, unreported_folds,
                                                unsupported_model_ids, unsupported_reference_ids,
                                                unsupported_titles, unsupported_values)
from dynalyst.orchestration.guard import ExecutionGuard, _model_of
from dynalyst.orchestration.llm import get_chat_model, model_spec
from dynalyst.orchestration.next_steps import format_follow_ups, strip_own_next_steps, suggest
from dynalyst.orchestration.reviewer import ANSWER_REVIEW_PROMPT, Reviewer
from dynalyst.orchestration.state import SessionState, diagram_range_key
from dynalyst.orchestration.subagents import SUBAGENTS
from dynalyst.orchestration.teams import TEAMS

ANSWER_REVIEW_ROUNDS = 2      # revisions after the answer review, each reviewed again
# the limit-cycle question (_maybe_ask) as the orchestrator may copy it from the earlier turn
_STALE_CYCLE_QUESTION = re.compile(
    r"\n*[^\n]*AUTO-07p found (?:a |\*\*)?[^\n]*Hopf[^\n]*?Continue the limit cycles[^\n]*"
    r"Reply \*\*yes\*\* or \*\*no\*\*\.?[^\n]*", re.I)

# the user asks for folds of cycles (saddle-node of limit cycles): AUTO then locates them
ASKS_CYCLE_FOLDS = re.compile(
    r"folds? of (?:the )?(?:limit[- ])?cycles?|(?:limit[- ])?cycles?[- ]folds?|"
    r"limit points? of (?:the )?(?:limit[- ])?cycles?|saddle[- ]nodes? of (?:the )?(?:limit[- ])?cycles?|"
    r"\bLPCs?\b", re.I)

ORCHESTRATOR_PROMPT = """WHO I AM
I am the orchestrator of a systems-biology modelling assistant. I pass each question to the
team that can answer it, read what comes back, and answer the user from those results.

SCOPE
In scope: questions about models of biological systems that my teams can answer (each
team's tool says what it covers).
Model-independent scientific or educational questions (for example, asking what a general
mathematical concept means) may be answered from established domain knowledge.
When a question concerns a particular model's behavior, properties, dynamics, mechanisms,
parameter effects, or comparison with another condition, model-specific conclusions must be
established from appropriate analysis of that model. General knowledge may provide context,
but it is not evidence that the particular model exhibits a behavior or mechanism.
Anything else (general chat, other subjects): I answer only
"This is outside what I can help with: I analyse models of biological systems."

TOOLS (when to call each)
- delegate_to_<team>(task): hands a part of the question to that team (see SCOPE).
- write_plan(steps, reason): first, for every question a team answers: the question broken
  into its parts (one part is a one-step plan), each in the user's terms, with the team for
  it; again when a result changes what is needed, with the reason.
- list_available_models, list_available_pdfs: which model files and PDFs are local. Before
  planning a search or download for a named model or paper, check: a local file is used.
- model_parameters(model): the model's parameters, their values and their part in the rate
  laws. When the user describes a parameter instead of naming it, I check which ones fit.
- inspect_model(model), inspect_dataset(file): a model's species, equations and reactions; a
  data file's columns. Read-only, for planning and asking; the teams do the analysis.
- A diagram that finds a Hopf point also continues the limit cycles born there, in the same
  run (period, amplitude, stability).
- A diagram given no range sweeps the parameter's value x10^-3 .. x10^3 on a log axis; a range
  the user gives is passed on exactly. When its result has log_scale, every parameter value in
  the answer is written actual and log10: "HB at k = 1.642 (log10_k = 0.2155)".
- A path the user gives to a model file is passed on whole, never shortened to its file name.

TERMS (a task keeps the user's term; these are not the same analysis)
- scan / sweep: the model run at each of several values spread over the whole range asked
  for, one by one (a few chosen values are not a scan of the range).
- bifurcation diagram / continuation: steady states followed smoothly as one parameter varies,
  with their stability and the points where it changes.

INPUT
The user's message, the recent conversation, and the session's facts (models and results
of earlier questions), which matter only when the user refers back to them.

RULES
1. A task is the user's own words for that part of the question, with the model and values
   an earlier turn or result fixed ("that model" -> its name). Teams see nothing else. A
   model the user names is passed as named, never with "find" or "download" added unless the
   user asks: if no such file exists, the tool says so, and then I can propose a download.
   When the user's words fit more than one parameter or model, I ask which before delegating.
2. Parts that do not need each other's results go out in the same turn (they run in
   parallel); a part that needs a result waits for it.
3. Tool errors and reviewer warnings come back with the results: read them, then give a
   corrected task, take another route, or report them. Going beyond the question (another
   model, parameter or analysis) needs the user's yes: propose it instead of doing it.
4. Every number, ID, plot and finding in my answer comes from a result, with its step:
   "folds at k1 = 2.31 and 4.87 [step 1]". A failed analysis shows nothing either way.
5. Objectives: the plan covers each distinct scientific objective of the question with the
   evidence it needs: one step when one analysis answers it, more when it needs independent,
   sequential or comparative analyses. No redundant or unrelated work.
6. Why / how a model behaves: from that model's own evidence (equations, network, Jacobian,
   stability, perturbation, simulation, continuation: whichever fits), never a textbook
   mechanism presented as a finding about the model.
7. Comparisons and rankings: evidence for every candidate under comparable conditions, with
   a metric stated first. Distinct metrics (distance to a threshold, response size, width of
   a regime) are not interchangeable; one candidate's result, a name or intuition ranks nothing.
8. The answer speaks to the user: never about a review, a correction or an earlier draft.
   "Bistable" says where: across a range of a parameter, or at the model's own values. Each conclusion traces to the result that establishes it; a failed, missing or conflicting
   analysis is reported as unresolved, never replaced by an assumption. Every analysis that ran
   for the question is in the answer: results that disagree are both reported.
9. A simulation runs only when the user asks for one (a simulation, a time course, an
   experiment over time). Period, amplitude, where oscillations start or stop and whether cycles
   are stable come from the bifurcation diagram and its cycle continuation. When only a
   simulation could settle something, I propose it instead of running it.
10. An omega goes in a task only when the user gave it. A question comparing deterministic and stochastic behaviour (do both oscillate, similar
   periods) is one task for the stochastic sub-agent, with compare_simulations.
11. A "generic" or "textbook" model of a bifurcation type is the textbook example, with no model
   file, even when a file of that name is listed.

OUTPUT
The answer first, in plain words, with step references. Then the key results with their
numbers; every plot as ![description](path); errors, reviewer warnings and "unresolved"
verdicts as given. No raw tool output, greetings or closing offers.

EXAMPLES
1. User: Is model.xml bistable in k1?
write_plan(steps=["dynamics_team: bifurcation diagram of model.xml in k1 (bistable: two folds)"])
delegate_to_dynamics_team(task="Bifurcation diagram of model.xml in the parameter k1.")
  -> folds (LP) at k1 = 2.31 and k1 = 4.87, stable branches outside; plot outputs/plots/model_k1.png
Answer: Yes: model.xml is bistable for 2.31 < k1 < 4.87, between two folds [step 1].
![Bifurcation diagram of model.xml in k1](outputs/plots/model_k1.png)

2. (one analysis is one step) User: Sample 5 parameter sets of model.xml within 5%,
continue each in k1. Which categories appear?
write_plan(steps=["dynamics_team: sample 5 parameter sets of model.xml within 5%, continue
                  each in k1, categories"])
delegate_to_dynamics_team(task="Sample 5 parameter sets of model.xml within 5%, continue each
in k1, and say which categories appear.")
  -> 3 bistable, 2 single fold; table outputs/lhs/model/summary.csv
Answer: 3 of the 5 sets are bistable and 2 have a single fold [step 1].

3. (the result's own verdict, not a reading of its last values) User: Does model.xml
oscillate over 1000 time units?
write_plan(steps=["dynamics_team: simulate model.xml over 1000 time units; oscillating?"])
delegate_to_dynamics_team(task="Simulate model.xml over 1000 time units: does it oscillate?")
  -> verdict "oscillating", 3 peaks, period 145, amplitude decreasing, limit cycle not established
Answer: It oscillates, with a period of about 145 [step 1]. Three peaks are too few to tell
damped from sustained oscillation; a longer run would decide it.

4. (more than one step) User: Find the Smith 2004 model in BioModels and draw its
bifurcation diagram in k1.
list_available_models() -> no local Smith2004 file, so a download step is needed
write_plan(steps=["model_team: find and download the Smith 2004 model",
                  "dynamics_team: bifurcation diagram of that model in k1"])
delegate_to_model_team(task="Find the Smith 2004 model in BioModels and download it.")
  -> downloaded <ID>, use_as "<ID>"
delegate_to_dynamics_team(task="Bifurcation diagram of <ID> in k1.")

5. (no parameter named: the tool chooses or asks) User: Compute a bifurcation diagram for
model.xml.
write_plan(steps=["dynamics_team: bifurcation diagram of model.xml (no parameter named)"])
delegate_to_dynamics_team(task="Bifurcation diagram of model.xml.")
  -> the tool screens the parameters and asks the user, with ranked candidates
"""

PLAN_TOOL = StructuredTool.from_function(
    func=lambda **kw: kw, name="write_plan",
    description="Record an objective-driven scientific plan before the first delegation and "
                "revise it when results change what is needed. Break the question into distinct "
                "scientific objectives and the evidence-producing analyses needed for each. Use "
                "one step when one analysis is sufficient; use multiple justified steps when "
                "independent or sequential investigations, comparisons, or complementary "
                "evidence are needed. Do not use an arbitrary minimum step count or add redundant "
                "work. steps: one line each, 'team: what', in the user's terms. reason: why it "
                "changed (for a revision).",
    args_schema={"type": "object", "properties": {
        "steps": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"}}, "required": ["steps"]})


@dataclass
class RunRecord:
    query: str
    model: dict
    plan: list = field(default_factory=list)            # the current plan's steps
    plan_versions: list = field(default_factory=list)   # every plan written, with its reason
    steps: list = field(default_factory=list)           # every tool call, in order
    follow_ups: list = field(default_factory=list)
    final_answer: str = ""
    final_check: dict = field(default_factory=dict)
    answer_review: dict = field(default_factory=dict)   # the reviewer's reading of the answer
    model_calls: dict = field(default_factory=lambda: {"orchestrator": 0, "subagent": 0})
    architecture: dict = field(default_factory=dict)
    team_runs: list = field(default_factory=list)       # each team supervisor's trace
    interventions: list = field(default_factory=list)   # where an exception stopped a call
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    seconds: float = 0.0
    record_path: str | None = None
    awaiting_approval: bool = False
    approval: dict | None = None
    tokens: dict = field(default_factory=dict)          # per role, as the provider reports

    def plots(self) -> list:
        out = []
        for step in self.steps:
            for key in ("plot_path", "counts_plot_path"):
                try:
                    p = json.loads(step.get("output") or "{}").get(key)
                except (TypeError, ValueError, AttributeError):
                    p = None
                if p:
                    out.append(p)
        return out


def _delegate_tool(name: str, summary: str) -> StructuredTool:
    schema = {"type": "object", "required": ["task"], "properties": {"task": {
        "type": "string",
        "description": "The complete task: model name or file, every value and range the "
                       "user gave, and what to produce."}}}
    return StructuredTool.from_function(func=lambda **kw: kw, name=f"delegate_to_{name}",
                                        description=summary, args_schema=schema)


def _history_messages(history, turns=10):
    out = []
    for m in (history or [])[-turns:]:
        cls = HumanMessage if m.get("role") == "user" else AIMessage
        out.append(cls(m.get("content") or ""))
    return out


def _error_of(output) -> str | None:
    try:
        d = json.loads(output)
    except (TypeError, ValueError):
        return None
    return str(d["error"]) if isinstance(d, dict) and d.get("error") else None


def _invoke(llm, messages):
    """A model call. A provider's server error (5xx) gets two more tries; anything else is raised."""
    for wait in (20, 60, 0):
        try:
            return llm.invoke(messages)
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None),
                                                                "status_code", None)
            if not wait or not (isinstance(status, int) and 500 <= status < 600):
                raise
            time.sleep(wait)


USER_SIZED_TOOLS = ("stochastic_simulation", "compare_simulations")
LHS_TOOLS = ("lhs_bifurcation_scan", "categorized_lhs_scan")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


_NUM_LIST = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*(?:,|and|or|&)\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)*"
# words that name a setting, so the numbers after them are the user's value for it
_SETTING_NAMES = {
    "omega": r"omega|\u03a9|\u03c9|system\s+sizes?|volumes?|sizes?",
    "t_end": r"t_end|t\s*end|tend|end\s+time|final\s+time|t|time|until|up\s+to|for",
}
_TIME_UNITS = r"(?:time\s+units?|units?\s+of\s+time|s|sec|secs|seconds?|min|mins|minutes?|h|hrs?|hours?)\b"


def _numbers(text: str) -> set:
    return {float(x) for x in _NUMBER.findall(text or "")}


def _labelled_values(user_text: str) -> tuple[dict, set]:
    """The numbers the user's words give each setting ("omega 5", "t = 400", "for 500 time
    units"), and the numbers they assign to other names ("MKP3 = 100", "k1: 0.2, 0.4")."""
    text = user_text or ""
    labelled = {}
    for key, names in _SETTING_NAMES.items():
        found = set()
        for m in re.finditer(rf"(?<![\w])(?:{names})\s*(?:=|:|of|is|to|at)?\s*({_NUM_LIST})", text, re.I):
            found |= _numbers(m.group(1))
        labelled[key] = found
    labelled["t_end"] |= {float(m.group(1)) for m in
                          re.finditer(rf"({_NUMBER.pattern})\s*{_TIME_UNITS}", text, re.I)}
    setting_words = re.compile(rf"^(?:{'|'.join(_SETTING_NAMES.values())})$", re.I)
    other = set()
    for m in re.finditer(rf"([A-Za-z_][\w\-\[\]]*)\s*(?:=|:)\s*({_NUM_LIST})", text):
        if not setting_words.match(m.group(1)):
            other |= _numbers(m.group(2))
    return labelled, other


def _user_values_only(args: dict, user_text: str) -> dict:
    """omega and t_end are the user's: a value their words do not give is dropped, so the
    tool's defaults (omega 1, t_end 100) apply. A value counts when the words name it for that
    setting ("omega 5", "t = 400", "for 500 s"), or when it appears in the words without being
    assigned to something else: "MKP3 = 100" does not let t_end = 100 through."""
    said = _numbers(user_text)
    labelled, other = _labelled_values(user_text)
    out = dict(args)
    for key in ("omega", "t_end"):
        v = out.get(key)
        if v is None:
            continue
        vals = v if isinstance(v, (list, tuple)) else [v]
        try:
            ok = all(float(x) in labelled[key] or (float(x) in said and float(x) not in other)
                     for x in vals)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.pop(key)
    return out


def _exact_states(args: dict, steps) -> dict:
    """initial_conditions that match a state find_steady_states gave earlier in this request
    (every species within rounding: 0.5 %, plus 0.1 % of the state's largest value) become that
    exact state: the numbers reach the tool as computed, not as retyped by a language model. A
    start that matches no found state stays as it is."""
    given = args.get("initial_conditions")
    if not isinstance(given, dict) or not given:
        return args
    model = _model_of(args)
    for s in steps:
        if (s.get("tool") != "find_steady_states" or s.get("status") != "ok"
                or _model_of(s.get("arguments") or {}) != model):
            continue
        try:
            states = json.loads(s.get("output") or "{}").get("steady_states") or []
        except (TypeError, ValueError):
            continue
        for found in states:
            exact = found.get("state") or {}
            if not exact or not set(given) <= set(exact):
                continue
            scale = max(abs(float(v)) for v in exact.values())
            try:
                close = all(abs(float(given[k]) - exact[k]) <= 0.005 * abs(exact[k]) + 1e-3 * scale
                            for k in given)
            except (TypeError, ValueError):
                return args
            if close:
                return {**args, "initial_conditions": {k: float(exact[k]) for k in given}}
    return args


def _omega_progress(steps) -> dict:
    """Where the stochastic runs stand on the tool's omega_plan: the system sizes whose runs
    finished, those stopped before t_end (a later finished run counts) and those not run."""
    finished, stopped, series = set(), set(), []
    for step in steps:
        try:
            out = json.loads(step.get("output") or "{}")
        except (ValueError, TypeError):
            continue
        series = (out.get("omega_plan") or {}).get("series") or series
        for size in out.get("per_omega") or []:
            runs = [m.get("status") for name, m in (size.get("method_results") or {}).items()
                    if name != "ode"]
            (finished if all(r == "completed" for r in runs) else stopped).add(
                float(f"{size['omega']:.6g}"))
    return {"finished": sorted(finished), "stopped": sorted(stopped - finished),
            "not_run": [x for x in map(float, series) if x not in finished | stopped]}


class Agent:
    """One agent over all capabilities. `llms` can override the chat model per role
    ({'orchestrator'|'team'|'subagent': chat model}); otherwise config.AGENT_MODEL (or
    `model`) is used for every role. `approval`: "ask" when a person is chatting (the
    chat entry points pass it): an AUTO-07p run whose call names no continuation
    parameter pauses with the ranked candidates; "auto" (default, called from code) runs."""

    def __init__(self, model: str | None = None, llms: dict | None = None,
                 verbose: bool = True, save_records: bool = True,
                 state: SessionState | None = None, approval: str = "auto"):
        self._model_override = model
        self._llms = dict(llms or {})
        self._llms_given = llms is not None
        self.verbose = verbose
        self.save_records = save_records
        if approval not in ("ask", "auto"):
            raise ValueError(f"approval must be 'ask' or 'auto', not {approval!r}")
        self.approval = approval
        # what earlier questions established (models, last results, a pending approval);
        # shown to the orchestrator only
        self.state = state or SessionState(persist=save_records)
        from dynalyst.housekeeping import prune_once
        prune_once()     # old scratch folders (AUTO runs, LHS work, CRNT jobs), once per process
        self._approved = None      # the candidate the user approved, for this run only
        self._user_text = ""       # the user's own words this run (a parameter they named)
        # the user's answer to "continue the limit cycles from the Hopf point(s)?" for this
        # run: None (not asked), "yes" or "no"
        self._cycles_choice = None
        # "continue the limit cycles?" or "look over a wider range?", asked after this run's
        # answer (not instead of it)
        self._cycle_offer = None
        self._widen_ok = False     # the user accepted a wider range than theirs, for this run
        self._resume = None        # a paused run the user's reply continues, for this run only
        self._stochastic_go = False    # the user chose to continue the stochastic runs of this run
        self._budget_asked = False     # a budget was reached and the user answered, in this run
        self._tasks = {}           # sub-agent -> tasks given in this run (budget)
        self._lock = threading.RLock()
        self._delegation_ids = itertools.count(1)
        self._tokens = {}

    def _llm(self, role):
        with self._lock:
            if role not in self._llms:
                if role == "team" and self._llms_given:   # scripted roles: the orchestrator's
                    self._llms[role] = self._llm("orchestrator")
                else:
                    self._llms[role] = get_chat_model(role, self._model_override,
                                                      on_usage=self._add_usage)
            return self._llms[role]

    def _add_usage(self, role, usage):
        cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
        with self._lock:
            t = self._tokens.setdefault(role, {"input": 0, "output": 0, "cached_input": 0})
            t["input"] += usage.get("input_tokens") or 0
            t["output"] += usage.get("output_tokens") or 0
            t["cached_input"] += cached

    def _count(self, rec, role):
        with self._lock:
            rec.model_calls[role] = rec.model_calls.get(role, 0) + 1

    def _log(self, msg):
        # to the console captured when the run started: a tool running in another
        # thread may have swapped sys.stdout to capture its own output
        if self.verbose:
            print(msg, file=getattr(self, "_out", None) or sys.stdout, flush=True)

    # -- public ----------------------------------------------------------------
    def run(self, query: str, history: list | None = None) -> RunRecord:
        t0 = time.time()
        rec = RunRecord(query=query, model={
            r: (type(self._llms[r]).__name__ if r in self._llms
                else model_spec(r, self._model_override))
            for r in ("orchestrator", "team", "subagent")})
        self._tokens = rec.tokens
        self._out = sys.stdout
        rec.architecture = {
            "mode": "hierarchical", "parallel": config.PARALLEL, "approval": self.approval,
            "max_parallel": config.MAX_PARALLEL,
            "budgets": {"orchestrator_calls": config.ORCHESTRATOR_MAX_CALLS,
                        "team_calls": config.TEAM_MAX_CALLS,
                        "subagent_calls": config.SUBAGENT_MAX_CALLS,
                        "analyses_per_delegation": config.SUBAGENT_MAX_ANALYSES,
                        "tasks_per_subagent": config.MAX_TASKS_PER_SUBAGENT,
                        "tool_retries": config.MAX_TOOL_RETRIES,
                        "plan_versions": config.MAX_PLAN_VERSIONS}}
        query, report_only = self._resolve_pending(query, rec)
        # the user's own words: the approval line the agent appends ("(Approved by the user: ...,
        # plotting X.)") is the agent's choice, not theirs
        self._user_text = re.sub(r"^\(Approved by the user:.*$", "", "\n".join(
            [query] + [m.get("content") or "" for m in (history or []) if m.get("role") == "user"]),
            flags=re.M)
        # earlier turns (answers already checked) can supply numbers for a custom_plot series
        self._history_texts = [m.get("content") or "" for m in (history or [])]
        if report_only is not None:     # "no" to a pending approval: the candidates only
            rec.final_answer = report_only
            return self._finish(rec, t0)
        guard = ExecutionGuard()
        self._tasks = {}              # sub-agent -> tasks given in this request (budget)
        try:
            if self._resume is not None:     # the user answered a question asked mid-run
                query = f"{query}\n\n{self._continue_paused(rec, guard)}"
            rec.final_answer = self._orchestrate(query, history, rec, guard)
            self._final_check(query, history, rec)
            issues = self._review_answer(query, history, rec)
            revisions = 0
            while issues and revisions < ANSWER_REVIEW_ROUNDS:
                # correct the claims, or run what is missing; the revision is reviewed again
                revisions += 1
                rec.final_answer = (self._orchestrate_more(_review_note(issues), rec, guard)
                                    or rec.final_answer)
                self._final_check(query, history, rec)
                issues = self._review_answer(query, history, rec)
                rec.answer_review["revised"] = revisions
        except approval.AwaitApproval as wait:      # stop and ask the user
            with self._lock:
                self.state.set(pending_approval={**wait.pending, "cycle_offer": self._cycle_offer})
            rec.final_answer, rec.awaiting_approval = wait.message, True
            self._log("AWAITING APPROVAL")
            return self._finish(rec, t0)
        rec.final_answer = tidy_answer(rec.final_answer, query)
        rec.final_answer += _problem_notes(rec.steps, rec.final_answer)
        if self._cycle_offer is not None:      # the answer, then the question
            message, pending = self._cycle_offer
            with self._lock:
                self.state.set(pending_approval={**pending, "steps": self._kept_steps(rec)})
            rec.final_answer += f"\n\n{message}"
            rec.awaiting_approval = True
            self._log("AWAITING APPROVAL (after the answer)")
            return self._finish(rec, t0)
        # no question is pending: a copy of the earlier limit-cycle question is not one
        rec.final_answer = _STALE_CYCLE_QUESTION.sub("", rec.final_answer).rstrip()
        try:
            rec.follow_ups = suggest(rec)
        except Exception as e:  # suggestions must never break an answer
            rec.follow_ups = []
            self._log(f"(follow-up suggestions failed: {e})")
        if rec.follow_ups:
            rec.final_answer = strip_own_next_steps(rec.final_answer)
        rec.final_answer += format_follow_ups(rec.follow_ups)
        return self._finish(rec, t0)

    def ask(self, query: str, history: list | None = None) -> str:
        return self.run(query, history).final_answer

    def _finish(self, rec, t0):
        self._approved = None
        self._cycles_choice = None
        self._cycle_offer = None
        self._widen_ok = False
        self._resume = None
        self._stochastic_go = False
        self._budget_asked = False
        try:   # what really happened, so "how reliable was that?" is answered from the record
            self.state.set(last_run=[
                {"tool": s.get("tool"), "status": s.get("status"),
                 "issues": (s.get("error") or "")[:200]}
                for s in rec.steps if s.get("tool")][-8:])
        except Exception:
            pass
        rec.seconds = round(time.time() - t0, 1)
        if self.save_records:
            self._save(rec)
        return rec

    def _resolve_pending(self, query, rec):
        """A reply to a pending approval: (query to run, None), or (query, the report
        only) for "no". Any other message is a new request and drops the pending one."""
        pending = self.state["pending_approval"]
        if pending and time.time() - pending.get("created", 0) > config.APPROVAL_EXPIRES_S:
            if pending.get("kind") == "crnt_wait":    # nobody answered: the search stops
                from dynalyst.analysis.crnt_analysis import stop_crnt_job
                stop_crnt_job(pending["job_id"])
            pending = None            # a stale question is not answered by "yes"
        if pending and pending.get("kind") in ("cycles", "widen") and pending.get("after_answer") \
                and approval.parse(query, [])[0] == "reject":
            # the answer was given with the question: "no" leaves it as it is
            self.state.set(pending_approval=None)
            rec.approval = {"reply": query, "question": pending["kind"], "choice": "no"}
            return query, ("No limit-cycle continuation, as you chose: the answer above stands."
                           if pending["kind"] == "cycles" else
                           "No wider range, as you chose: the answer above stands.")
        if pending and pending.get("kind") in ("cycles", "widen", "crnt_wait", "stochastic_wait",
                                               "budget_wait"):
            return self._resolve_question(pending, query, rec), None
        if not pending or not pending.get("candidates"):
            if self.state["pending_approval"]:
                self.state.set(pending_approval=None)
            return query, None
        shown = list(pending["candidates"])
        kind, idx = approval.parse(query, shown)
        if kind == "other" and pending.get("other_candidates"):
            k, j = approval.parse_name(query, pending["other_candidates"])
            if k == "approve":
                kind, idx = k, len(shown) + j
        self.state.set(pending_approval=None)
        if kind == "approve":
            cand = (shown + list(pending.get("other_candidates") or []))[idx]
            self._approved = cand
            rec.approval = {"reply": query, "choice": idx + 1, "candidate": cand}
            self._log(f"APPROVED candidate {idx + 1}: {cand['parameter']}")
            rec.query = f"{pending['query']}\n{approval.directive(cand, pending.get('tool'))}"
            return rec.query, None
        if kind == "reject":
            from dynalyst.analysis.candidates import format_candidates
            rec.approval = {"reply": query, "choice": "no"}
            return query, (f"No AUTO-07p run, as you chose. The ranked candidates for "
                           f"{pending['model']} were:\n\n"
                           + format_candidates({"candidates": pending["candidates"]},
                                               species=pending.get("tool") != "run_bifurcation_analysis"))
        rec.approval = {"reply": query, "choice": "new request (the pending approval was dropped)"}
        return query, None

    def _resolve_question(self, pending, query, rec) -> str:
        """The reply to "continue the limit cycles?" (yes / no) or to "the CRNT search is
        still running" (continue / plot). Any other message is a new request: the question
        is dropped (and a waiting CRNT search stopped)."""
        from dynalyst.analysis.crnt_analysis import stop_crnt_job
        self.state.set(pending_approval=None)
        if pending["kind"] in ("cycles", "widen", "stochastic_wait", "budget_wait"):
            kind, _ = approval.parse(query, [])
            choice = {"approve": "yes", "reject": "no"}.get(kind)
        else:
            choice = {"continue": "continue", "plot": "plot"}.get(approval.parse_crnt_wait(query))
            if choice != "continue":
                stop_crnt_job(pending["job_id"])
        if choice is None:
            rec.approval = {"reply": query, "choice": f"new request (the {pending['kind']} "
                                                      "question was dropped)"}
            return query
        self._resume = {**pending, "choice": choice}
        self._cycle_offer = pending.get("cycle_offer")
        if pending["kind"] == "cycles":
            self._cycles_choice = choice
        rec.approval = {"reply": query, "question": pending["kind"], "choice": choice}
        self._log(f"REPLY to the {pending['kind']} question: {choice}")
        rec.query = pending["query"]
        return rec.query

    def _continue_paused(self, rec, guard) -> str:
        """Continue a run that stopped to ask the user: the steps it had done are kept (not
        run again), the call the user chose runs now, and the orchestrator gets a note of
        both to finish the answer with."""
        r = self._resume
        rec.steps = [dict(s) for s in r.get("steps") or []]
        call = r.get("call") or {}
        # one diagram per swept parameter whose Hopf points the question listed
        calls = [c for _, c in (r.get("calls") or {}).values()] or [call]
        # the approved limit-cycle run repeats the equilibrium call's arguments: that call
        # must not count as done, or its equilibrium-only result would be reused
        rerun = r["kind"] in ("cycles", "widen") and r["choice"] == "yes"
        self._widen_ok = r["kind"] == "widen" and r["choice"] == "yes"
        for s in rec.steps:          # done already: an identical call reuses the result
            if s.get("tool") and s.get("status") == "ok" and not (
                    rerun and s["tool"] == "run_bifurcation_analysis"):
                guard.record(s["tool"], s.get("arguments") or {}, s)
        done = "\n".join(f"- {s.get('sub_agent')}: {s['tool']} -> {s.get('status')}: "
                         f"{registry.for_model(s.get('output') or '', 4000)}"
                         for s in rec.steps if s.get("tool"))
        note = f"[Already done in this request, do not repeat:\n{done}"
        if rerun:
            guard.diagrams.clear()      # the user asked for this run
            for c in calls:
                step = self._run_tool(c["sub_agent"], c["task"], next(self._delegation_ids),
                                      {"name": c["tool"], "args": c["arguments"]}, rec, guard,
                                      [], {"reviewer": Reviewer(), "sent_back": 0})
                a = c["arguments"]
                note += (f"\nThe user asked for the limit cycles from the Hopf point(s); the "
                         f"continuation in {a.get('param_name')} ran now. Report these "
                         f"cycles in the answer, with this run's plot:\n{step['reply']}"
                         if r["kind"] == "cycles" else
                         f"\nThe user asked for the wider range {a.get('param_name')} in "
                         f"[{a.get('param_min'):g}, {a.get('param_max'):g}]; the continuation ran "
                         f"now. Answer from this run, with its plot:\n{step['reply']}")
        elif r["kind"] in ("cycles", "widen"):
            note += ("\nThe user did not want the limit-cycle continuation." if r["kind"] == "cycles"
                     else "\nThe user did not want a wider range.")
        elif r["kind"] == "budget_wait":
            self._budget_asked = True
            note += ("\nThe user allowed more analyses: run what is still needed." if r["choice"] == "yes"
                     else "\nThe user wants no more analyses: answer from what ran, and say what is "
                          "still missing.")
        elif r["kind"] == "stochastic_wait":
            self._stochastic_go = r["choice"] == "yes"
            note += ("\nThe user chose to continue the stochastic runs: run the system sizes the "
                     "task names that have not run yet. Retry a stopped size only if its projection says "
                     "it can finish." if self._stochastic_go else
                     "\nThe user stopped the stochastic runs: answer from those that finished, "
                     "and name the system sizes not finished.")
        elif r["choice"] == "continue":
            step = self._run_tool("network_theory", call["task"], next(self._delegation_ids),
                                  {"name": "crnt_bistability", "args": {"job_id": r["job_id"]}},
                                  rec, guard, [], {"reviewer": Reviewer(), "sent_back": 0})
            note += (f"\nThe user chose to let the CRNT search on {r['model']} finish; its "
                     f"result:\n{step['reply']}\nIf it found candidate sets, check the first "
                     "set's model file (multistationary_sets[0].model_file) with a bifurcation "
                     "diagram whose parameter is left out (the tool screens it), and report "
                     "whether two steady states are stable.")
        else:
            note += (f"\nThe user stopped the CRNT search on {r['model']} and asked for the "
                     "model's own bifurcation diagram instead.")
        return note + "\nFinish the plan with what is still needed, then answer.]"

    # -- the orchestrator ----------------------------------------------------------
    def _orchestrate(self, query, history, rec, guard) -> str:
        delegates = {t.name: t.summary for t in TEAMS.values()}
        tools = ([_delegate_tool(n, s) for n, s in delegates.items()]
                 + registry.get(registry.UTILITY) + [PLAN_TOOL])
        llm = self._llm("orchestrator").bind_tools(tools)
        listing = "\n".join(f"- {n}: {s}" for n, s in delegates.items())
        facts = self.state.as_text()
        system = (ORCHESTRATOR_PROMPT.format(delegates=listing)
                  + (f"\nSESSION FACTS (from earlier questions; use only when the user refers "
                     f"back to them)\n{facts}\n" if facts else ""))
        messages = [SystemMessage(system), *_history_messages(history), HumanMessage(query)]
        self._conversation = (llm, messages)
        return self._orchestrate_loop(llm, messages, rec, guard)

    def _orchestrate_more(self, note, rec, guard) -> str:
        """The same orchestrator conversation, continued with a note (the reviewer's issues)."""
        llm, messages = self._conversation
        messages.append(HumanMessage(note))
        return self._orchestrate_loop(llm, messages, rec, guard)

    def _orchestrate_loop(self, llm, messages, rec, guard) -> str:
        answer = ""
        for call in range(config.ORCHESTRATOR_MAX_CALLS):
            last = call == config.ORCHESTRATOR_MAX_CALLS - 1
            if last:
                messages.append(HumanMessage("Model-call budget reached: write the final answer "
                                             "now from the results above, without calling tools."))
            ai = _invoke(llm, messages)
            self._count(rec, "orchestrator")
            messages.append(ai)
            if not ai.tool_calls or last:
                answer = _text(ai)
                break
            replies = {}
            for tc in ai.tool_calls:
                if tc["name"] == PLAN_TOOL.name:
                    replies[tc["id"]] = self._write_plan(rec, tc.get("args") or {})
            others = [tc for tc in ai.tool_calls if tc["name"] != PLAN_TOOL.name]
            if others:
                replies.update(zip([tc["id"] for tc in others],
                                   self._act_all(others, rec, guard)))
            for tc in ai.tool_calls:
                messages.append(ToolMessage(replies[tc["id"]], tool_call_id=tc["id"]))
        return answer or self._fallback_answer(rec)

    def _write_plan(self, rec, args) -> str:
        steps = [str(s) for s in (args.get("steps") or []) if str(s).strip()]
        with self._lock:
            if len(rec.plan_versions) >= config.MAX_PLAN_VERSIONS:
                rec.interventions.append({"kind": "plan version limit"})
                return (f"Not recorded: {config.MAX_PLAN_VERSIONS} plan versions already. Answer "
                        "with the results so far and say what could not be done.")
            rec.plan = steps
            rec.plan_versions.append({"version": len(rec.plan_versions) + 1, "steps": steps,
                                      "reason": args.get("reason") or "",
                                      "after_tool_calls": len(rec.steps)})
            n = len(rec.plan_versions)
        self._log(f"PLAN v{n}" + (f" ({args['reason']})" if args.get("reason") else "") + ": "
                  + "; ".join(steps))
        return f"Plan version {n} recorded. Carry it out."

    def _act_all(self, tool_calls, rec, guard, scope=None) -> list:
        """The replies to one turn's tool calls, in order. Utilities run first; two or more
        delegations in one turn run in parallel (config.PARALLEL). A stop for the user's
        approval in any of them is raised once all have finished."""
        delegations = [tc for tc in tool_calls if tc["name"].startswith("delegate_to_")]
        out = {tc["id"]: self._act(tc, rec, guard, scope)
               for tc in tool_calls if tc not in delegations}
        if config.PARALLEL and len(delegations) > 1:
            self._log("  PARALLEL: " + ", ".join(tc["name"][len("delegate_to_"):]
                                                 for tc in delegations))
            with ThreadPoolExecutor(max_workers=min(config.MAX_PARALLEL, len(delegations))) as ex:
                results = list(ex.map(lambda tc: _capture(self._act, tc, rec, guard, scope),
                                      delegations))
        else:
            results = [_capture(self._act, tc, rec, guard, scope) for tc in delegations]
        waiting = None
        for tc, (reply, exc) in zip(delegations, results):
            if isinstance(exc, approval.AwaitApproval):
                waiting = waiting or exc
            elif exc is not None:
                raise exc
            out[tc["id"]] = reply
        if waiting:
            raise waiting
        return [out[tc["id"]] for tc in tool_calls]

    def _act(self, tc, rec, guard, scope=None) -> str:
        """One tool call of a supervisor. scope=None: the orchestrator (its teams and
        utilities); a set: a team supervisor and its members."""
        name, args = tc["name"], tc.get("args") or {}
        target = name[len("delegate_to_"):] if name.startswith("delegate_to_") else None
        task = str(args.get("task") or "")
        if scope is None and target in TEAMS:
            return self._team(target, task, rec, guard)
        if scope is not None and target in scope:
            with self._lock:
                n = self._tasks.get(target, 0)
                self._tasks[target] = n + 1
            if n >= config.MAX_TASKS_PER_SUBAGENT:
                self._ask_for_budget(f"tasks for {target}", n, task, rec)
                with self._lock:
                    rec.interventions.append({"kind": "task budget", "sub_agent": target})
                return (f"Not delegated, and it will not be in this request: {target} already had "
                        f"{n} tasks (this assistant's limit per request, not a limit of any "
                        f"analysis). Do not delegate to {target} again. Report what its results "
                        "show, and what is still missing.")
            return self._delegate(target, task, rec, guard, getattr(scope, "team_run", None))
        if scope is None and name in registry.UTILITY:
            self._log(f"  UTILITY {name}({json.dumps(args, default=str)[:200]})")
            output = registry.execute(name, args)
            error = _error_of(output)
            with self._lock:
                rec.steps.append({"kind": "utility", "tool": name, "arguments": args,
                                  "output": output, "status": "error" if error else "ok",
                                  "error": error})
            if not error:
                self._remember(name, args, output)
            return ("ERROR from the tool: " if error else "") + registry.for_model(output)
        return json.dumps({"error": f"'{name}' is not available here."})

    # -- a team supervisor -----------------------------------------------------------
    def _team(self, team, task, rec, guard) -> str:
        spec = TEAMS[team]
        self._log(f"  TEAM -> {team}: {task[:200]}")
        llm = self._llm("team").bind_tools(
            [_delegate_tool(m, SUBAGENTS[m].summary) for m in spec.members])
        messages = [SystemMessage(spec.prompt), HumanMessage(f"Task: {task}")]
        trace = {"team": team, "task": task, "delegations": [], "summary": None,
                 "model_calls": 0, "started": time.time()}
        with self._lock:
            rec.team_runs.append(trace)
            members = _team_scope(spec.members, len(rec.team_runs) - 1)
        for call in range(config.TEAM_MAX_CALLS):
            last = call == config.TEAM_MAX_CALLS - 1
            if last:
                messages.append(HumanMessage("Model-call budget reached: report the results now, "
                                             "without calling tools."))
            ai = _invoke(llm, messages)
            self._count(rec, "team")
            trace["model_calls"] += 1
            messages.append(ai)
            if not ai.tool_calls or last:
                trace["summary"] = _text(ai)
                break
            replies = self._act_all(ai.tool_calls, rec, guard, scope=members)
            for tc, reply in zip(ai.tool_calls, replies):
                trace["delegations"].append({"sub_agent": tc["name"][len("delegate_to_"):],
                                             "task": (tc.get("args") or {}).get("task"),
                                             "report": reply[:3000]})
                messages.append(ToolMessage(reply, tool_call_id=tc["id"]))
        trace["ended"] = time.time()
        return trace["summary"] or "The team supervisor gave no report."

    # -- a sub-agent -------------------------------------------------------------------
    def _delegate(self, sub, task, rec, guard, team_run=None) -> str:
        """A sub-agent's loop for one task: it calls context tools and its analysis tools
        (each through the guard), reads their results and errors, and ends with a report."""
        spec = SUBAGENTS[sub]
        with self._lock:
            delegation = next(self._delegation_ids)
        self._log(f"  DELEGATE -> {sub}: {task[:200]}")
        llm = self._llm("subagent").bind_tools(registry.get(registry.CONTEXT)
                                               + registry.get(spec.objective_tools))
        messages = [SystemMessage(spec.prompt),
                    HumanMessage(f"Task: {task}" + lessons.for_tools(spec.objective_tools))]
        mine, analyses = [], 0       # this delegation's tool calls; analyses that ran
        review = {"reviewer": Reviewer(), "sent_back": 0}
        report = None
        max_analyses, max_calls, call = config.SUBAGENT_MAX_ANALYSES, config.SUBAGENT_MAX_CALLS, -1
        began = time.time()
        while call < max_calls - 1:
            call += 1
            last = call == max_calls - 1
            if last:
                messages.append(HumanMessage("Model-call budget reached: report now, without "
                                             "calling tools."))
            ai = _invoke(llm, messages)
            self._count(rec, "subagent")
            messages.append(ai)
            if not ai.tool_calls or last:
                report = _text(ai)
                break
            for tc in ai.tool_calls:
                if tc["name"] in registry.CONTEXT:
                    out = registry.execute(tc["name"], tc.get("args") or {})
                    self._log(f"    context {tc['name']}")
                    reply = registry.for_model(out)
                elif analyses >= max_analyses:
                    self._ask_for_budget(f"analyses for one {sub} task", max_analyses, task, rec)
                    reply = (f"Not run: {max_analyses} analyses is the limit for "
                             "one task. Report now.")
                else:
                    step = self._run_tool(sub, task, delegation, tc, rec, guard, mine, review,
                                          team_run)
                    analyses += step.get("status") in ("ok", "error", "sent_back", "unresolved")
                    reply = step["reply"]
                    if sub == "stochastic":     # long stochastic runs: a person chatting is asked
                        self._ask_to_go_on(task, mine, began, rec)
                messages.append(ToolMessage(reply, tool_call_id=tc["id"]))
        if not mine:
            return f"{sub} ran no analysis. Its report: {report or '(none)'}"
        return report or "The sub-agent gave no report; its last tool result: " + (
            registry.for_model(mine[-1].get("output") or mine[-1].get("error") or ""))

    def _ask_for_budget(self, what, limit, task, rec):
        """Chat only: the first budget this request reaches stops it and asks. A yes runs the
        request on from what it kept, with its budgets counted anew; a limit reached after that
        is final. Scripts and evaluations are not asked."""
        if self.approval != "ask" or self._budget_asked:
            return
        raise approval.AwaitApproval(approval.budget_message(what, limit), {
            "kind": "budget_wait", "query": rec.query, "created": time.time(),
            "steps": self._kept_steps(rec), "call": {"task": task}})

    def _ask_to_go_on(self, task, steps, began, rec):
        """Chat only: a series of stochastic runs past config.STOCHASTIC_ASK_SECONDS stops and
        asks whether to continue, keeping what ran. Scripts and evaluations are not asked."""
        waited, progress = time.time() - began, _omega_progress(steps)
        if (self.approval != "ask" or self._stochastic_go
                or not (progress["stopped"] or progress["not_run"])
                or waited < config.STOCHASTIC_ASK_SECONDS):
            return
        raise approval.AwaitApproval(approval.stochastic_wait_message(waited / 60, progress), {
            "kind": "stochastic_wait", "query": rec.query, "created": time.time(),
            "steps": self._kept_steps(rec), "call": {"task": task}})

    def _diagram_range(self, args: dict) -> dict:
        """An LHS scan given no range for its continuation parameter runs over the range this
        session's bifurcation diagrams already swept for that model and parameter: its own
        default (the baseline value x0.1 to x10) can miss everything the diagram found."""
        if args.get("param_min") is not None or args.get("param_max") is not None:
            return args
        if not args.get("model") or not args.get("bif_param"):
            return args
        seen = (self.state["diagram_ranges"] or {}).get(
            diagram_range_key(args["model"], args["bif_param"]))
        if not seen:
            return args
        self._log(f"  (LHS range from this session's diagrams: {args['bif_param']} {seen})")
        return {**args, "param_min": seen[0], "param_max": seen[1]}

    def _run_tool(self, sub, task, delegation, tc, rec, guard, mine, review,
                  team_run=None) -> dict:
        """One analysis call: guard -> (approval) -> tool -> reviewer. Returns its step, with
        the reply for the sub-agent in step['reply'] (not saved in the record)."""
        tool = tc["name"]
        args = registry.coerce_args(tool, tc.get("args") or {})
        if tool in USER_SIZED_TOOLS:     # before the guard and the record: they see what runs
            args = _exact_states(_user_values_only(args, self._user_text), rec.steps)
        if tool in LHS_TOOLS:
            args = self._diagram_range(args)
        if self._approved and tool in approval.GATED_TOOLS and tool in registry.all_tools():
            args = approval.inject(tool, args, self._approved, set(registry.schema_of(
                registry.all_tools()[tool]).get("properties", {})))
        if (tool == "run_bifurcation_analysis" and args.get("plot_state")
                and not re.search(rf"(?<![\w.]){re.escape(str(args['plot_state']))}(?![\w])",
                                  self._user_text, re.I)):
            # a y-axis species the user did not name: the tool picks the one that shows the
            # branches apart, after the run
            args = {k: v for k, v in args.items() if k != "plot_state"}
        with self._lock:
            decision = guard.check(sub, tool, args)
        step = {"kind": "delegation", "sub_agent": sub, "task": task, "delegation_id": delegation,
                "team_run": team_run,
                "tool": tool, "arguments": args, "guard": decision.as_dict(), "status": None,
                "output": None, "error": None, "started": time.time()}
        with self._lock:
            rec.steps.append(step)
        mine.append(step)
        self._log(f"    CALL {tool}({json.dumps(args, default=str)[:200]})  guard: "
                  + ("duplicate: result reused" if decision.duplicate_of is not None else
                     "ok" if decision.allowed else f"REJECTED ({decision.reason})"))
        if not decision.allowed:
            step.update(status="rejected", error=decision.reason)
            with self._lock:
                rec.interventions.append({"kind": "guard", "sub_agent": sub, "tool": tool,
                                          "reason": decision.reason})
            return {**step, "reply": f"NOT RUN (guard): {decision.reason}"}
        if decision.duplicate_of is not None:
            earlier = decision.duplicate_of
            step.update(status=earlier.get("status"), output=earlier.get("output"),
                        duplicate=True)
            return {**step, "reply": "Already run earlier in this request; its result: "
                                     + registry.for_model(earlier.get("output") or "")}
        if tool == "run_bifurcation_analysis" and ASKS_CYCLE_FOLDS.search(rec.query or ""):
            # read from the user's own words, not left to the model to pass on
            args = {**args, "cycle_folds": True}
        if not self._approved and (approval.needed(tool, args, self.approval) or (
                self._resume is None and self._cycles_choice is None
                and approval.needed_for_user(tool, args, self._user_text, self.approval))):
            self._ask_approval(tool, args, step, rec)      # raises when there are candidates
        if tool == "custom_plot" and args.get("series"):
            # plotted numbers must come from earlier results of this request, earlier turns or
            # the user, never from this call's own arguments
            evidence = [self._user_text, *getattr(self, "_history_texts", [])]
            for s in rec.steps:
                if s is not step and s.get("status") == "ok":
                    evidence += [s.get("output") or "", json.dumps(s.get("arguments"), default=str)]
            made_up = unsupported_points(args["series"], evidence)
            if made_up:
                reason = ("these plotted numbers are in no earlier tool result: "
                          + ", ".join(f"{v:g}" for v in made_up[:10])
                          + ". Plot only values the results gave")
                step.update(status="rejected", error=reason)
                with self._lock:
                    rec.interventions.append({"kind": "unsupported_points", "sub_agent": sub,
                                              "tool": tool, "reason": reason})
                return {**step, "reply": f"NOT RUN: {reason}."}
        kept = (None if self._widen_ok else
                approval.widens_user_range(tool, args, self._user_text, self.approval))
        if kept is not None:
            # the user's range is theirs: a wider one is offered after the answer, not run
            reason = (f"the user asked for [{kept[0]:g}, {kept[1]:g}]: run that range as given. "
                      "If it holds no bifurcation point, say so; the user is then asked about a "
                      "wider range")
            step.update(status="rejected", error=reason)
            with self._lock:
                rec.interventions.append({"kind": "user_range", "sub_agent": sub, "tool": tool,
                                          "reason": reason})
            return {**step, "reply": f"NOT RUN: {reason}."}
        run_args = args
        if (tool == "crnt_bistability" and self.approval == "ask" and config.CRNT_ASK_SECONDS > 0
                and not args.get("job_id") and not args.get("structure_only")):
            # a person is chatting: a long search is not stopped, they are asked
            run_args = {**args, "_pause_after": config.CRNT_ASK_SECONDS}
        output = registry.execute(tool, run_args)
        error = _error_of(output)
        # the reviewer re-checks the result independently (exception 8); "replan" sends it
        # back with a correction, at most config.REVIEWER_MAX_REPLANS times per task
        verdict = review["reviewer"].review(tool, args, output, attempt=review["sent_back"])
        status = "error" if error else {"replan": "sent_back",
                                         "unresolved": "unresolved"}.get(verdict.verdict, "ok")
        step.update(output=output, status=status, error=error, reviewer=verdict.as_dict(),
                    seconds=round(time.time() - step["started"], 1))
        problem = error or ("; ".join(verdict.issues) if status != "ok" else None)
        with self._lock:
            guard.record(tool, args, step, error=problem)
        self._log(f"    -> {'ERROR: ' + error[:200] if error else status}  review: "
                  f"{verdict.verdict}" + (f" ({'; '.join(verdict.issues)[:200]})" if verdict.issues else ""))
        if status == "ok":
            self._remember(tool, args, output)
            earlier = [s for s in mine[:-1] if s["tool"] == tool]
            try:     # a call that failed, then succeeded with other arguments: a lesson
                for lesson in lessons.record(earlier + [step]):
                    self._log(f"    LESSON saved: {lesson['tool']}: {lesson['problem'][:90]}")
            except Exception:       # learning must never cost a run
                pass
        self._maybe_ask(sub, task, tool, args, output, step, rec)
        head = ("ERROR from the tool: " if error else "RESULT: ") + registry.for_model(output)
        if verdict.verdict == "replan":
            review["sent_back"] += 1
            fix = (f" Suggested correction: call {tool} again with these argument changes: "
                   f"{json.dumps(verdict.suggested_args)}." if verdict.suggested_args else "")
            return {**step, "reply": f"{head}\n\nREVIEWER: sent back ({review['sent_back']}/"
                                     f"{config.REVIEWER_MAX_REPLANS}). Issues: "
                                     f"{'; '.join(verdict.issues)}.{fix} Make ONE corrected call, "
                                     "or report the problem."}
        if verdict.verdict == "unresolved":
            return {**step, "reply": f"{head}\n\nREVIEWER: still failing after "
                                     f"{config.REVIEWER_MAX_REPLANS} corrections: "
                                     f"{'; '.join(verdict.issues)}. Report this; do not retry."}
        if verdict.issues:
            return {**step, "reply": f"{head}\n\nREVIEWER WARNINGS (report them): "
                                     f"{'; '.join(verdict.issues)}"}
        return {**step, "reply": head}

    def _maybe_ask(self, sub, task, tool, args, output, step, rec):
        """Ask the user (chat only): whether to wait for a CRNT search that is still running
        (the run stops, keeping what ran), or whether to look over a wider range when the
        user's own range holds no special point."""
        if self.approval != "ask":
            return
        try:
            res = json.loads(output)
        except (TypeError, ValueError):
            return
        if not isinstance(res, dict):
            return
        if tool == "crnt_bistability" and res.get("still_running"):
            step.update(status="awaiting_user")
            model = args.get("model_name_or_path") or res.get("model") or "the model"
            raise approval.AwaitApproval(approval.crnt_wait_message(model, res), {
                "kind": "crnt_wait", "query": rec.query, "created": time.time(),
                "job_id": res["job_id"], "model": model, "steps": self._kept_steps(rec, step),
                "call": {"task": task}})
        if (tool == "run_bifurcation_analysis" and step.get("status") == "ok"
                and self._cycle_offer is None and not self._widen_ok
                and not res.get("bifurcation_points") and not res.get("range_widened")):
            # the user's own range holds no special point: a wider one is proposed, not run
            param = res.get("swept_parameter") or args.get("param_name")
            rng = res.get("swept_range") or []
            if not param or len(rng) != 2 or not any(
                    abs(rng[0] - a) <= 1e-9 * max(1, abs(a)) and abs(rng[1] - b) <= 1e-9 * max(1, abs(b))
                    for a, b in approval.user_ranges(self._user_text, param)):
                return
            from dynalyst.orchestration.inputs import _model_for
            own = (args.get("fixed_params") or {}).get(param)
            if own is None:
                own = ((_model_for(str(args.get("model_name_or_path") or "")) or {})
                       .get("param_defaults") or {}).get(param)
            lo, hi = approval.wider_range(float(rng[0]), float(rng[1]),
                                          None if own is None else float(own))
            rerun = {**args, "param_name": param, "param_min": lo, "param_max": hi}
            self._cycle_offer = (
                f"AUTO-07p found no bifurcation point in your range {param} ∈ "
                f"[{rng[0]:g}, {rng[1]:g}]. Look over a wider range, {param} ∈ [{lo:g}, {hi:g}]? "
                "This runs AUTO again. Reply **yes** or **no**.",
                {"kind": "widen", "after_answer": True, "query": rec.query, "created": time.time(),
                 "calls": {param: [f"{param} in [{lo:g}, {hi:g}]",
                                   {"sub_agent": sub, "task": task, "tool": tool, "arguments": rerun}]}})

    @staticmethod
    def _kept_steps(rec, leave_out=None) -> list:
        """The run's tool calls so far, for continuing it after the user's reply."""
        return [{k: v for k, v in s.items() if k != "reply"} for s in rec.steps
                if s is not leave_out]

    def _remember(self, tool, args, output):
        try:
            with self._lock:     # the session's facts: one writer at a time
                self.state.update_from_tool(tool, args, output)
        except Exception as e:      # the facts are an aid; they must never break a run
            self._log(f"(state update failed: {e})")

    def _ask_approval(self, tool, args, step, rec):
        """Stop before AUTO with the ranked candidates (no parameter in the call). A model
        the screen cannot read, or with no candidate, runs without asking."""
        from dynalyst.analysis.candidates import propose
        model = args.get("model") or args.get("model_name_or_path")
        try:
            res = propose(model)
        except Exception as e:
            self._log(f"    (no candidates for approval: {e}; running without asking)")
            return
        if not res.get("candidates"):
            return
        res = self._phrase_candidates(model, res)
        with self._lock:
            self.state.set(candidates={"model": model, "list": res["candidates"][:6]})
        step.update(status="awaiting_approval")
        picked = args.get("param_name") or args.get("bif_param")
        raise approval.AwaitApproval(approval.message(model, tool, res, picked), {
            "query": rec.query, "tool": tool, "model": model, "arguments": args,
            "created": time.time(), "candidates": res["candidates"][:6],
            "method": res.get("method"), "other_candidates": res.get("other_candidates", [])})

    def _phrase_candidates(self, model, res) -> dict:
        """A word of the request that stands for several parameters ("kcat": k1cat ... k4cat):
        those parameters are the candidates (screened ones in the screen's order, then the
        rest over a decade either side of their value), and the question names the word."""
        from dynalyst.orchestration.inputs import _model_for
        m = _model_for(str(model)) or {}
        found = approval.ambiguous_phrase(self._user_text, m.get("param_names"))
        if not found:
            return res
        word, hits = found
        known = {c["parameter"]: c for c in (res.get("candidates") or [])
                 + (res.get("other_candidates") or [])}
        cands = []
        for p in sorted(hits, key=lambda p: (p not in known, hits.index(p))):
            v = float((m.get("param_defaults") or {}).get(p) or 0.0)
            cands.append(known.get(p) or {
                "parameter": p, "range": ([v / 10, v * 10] if v > 0 else [v * 10, v / 10] if v < 0
                                          else [0.0, 1.0]),
                "y_species": None, "expected": "unknown (not screened)",
                "reason": f"the screen found no crossing within a decade of {p} = {v:g}"})
        note = (f"Your request's '{word}' stands for several parameters ({', '.join(hits)}): "
                "which one should be continued?")
        return {**res, "candidates": cands, "other_candidates": [],
                "note": f"{note} {res['note']}" if res.get("note") else note}

    # -- the final-answer check (exception 7) ------------------------------------------
    def _final_check(self, query, history, rec):
        """What the answer claims must come from this request's results. Numbers, model IDs
        and paper titles must appear in a tool result (or the user's words); conclusions
        must be ones an analysis reached: no "absent" behaviour from a method that failed,
        no bistability from CRNT candidates alone, no bifurcation point no analysis found,
        and no fold the diagram found left out. Paragraphs that break this are rewritten
        once from the results; what is still wrong is removed, or noted."""
        rec.final_answer = qualify_empty_continuation(rec.final_answer, rec.steps)
        run_evidence = [query]
        for s in rec.steps:
            run_evidence += [s.get("output") or "", json.dumps(s.get("arguments"), default=str)]
        history_texts = [m.get("content") or "" for m in (history or [])]
        evidence = run_evidence + history_texts
        # model identities come from THIS request (history counts only when no tool ran)
        id_evidence = run_evidence if rec.steps else evidence
        identities = model_identities(run_evidence)
        answer = rec.final_answer
        bad = unsupported_values(answer, evidence)
        bad_ids = (unsupported_model_ids(answer, id_evidence, identities)
                   + unsupported_reference_ids(answer, evidence))
        made_up = unsupported_titles(answer, run_evidence + history_texts)
        blanks = placeholders(answer)
        undecided = undecided_absence_claims(answer, rec.steps)
        unchecked = stability_claim_without_check(answer, rec.steps)
        no_point = (unfound_hopf_claim(answer, rec.steps)
                    or unfound_bifurcation_claim(answer, rec.steps))
        folds = unreported_folds(answer, rec.steps)
        rec.final_check = {"unsupported_values": bad, "unsupported_model_ids": bad_ids,
                           "titles_not_found": made_up, "placeholders": blanks,
                           "undecided_claims": undecided, "stability_unchecked": unchecked,
                           "hopf_not_found": no_point, "folds_not_reported": folds,
                           "rewritten": False}
        wrong = bool(bad or bad_ids or made_up or blanks or undecided or unchecked or no_point)
        if not (wrong or folds):
            return
        self._log(f"FINAL CHECK: values {bad}, IDs {[b['id'] for b in bad_ids]}, titles "
                  f"{made_up}, placeholders {blanks}, undecided {undecided}, CRNT-only "
                  f"bistability {unchecked}, unfound point {no_point}, folds left out {folds}")
        if wrong:
            results = "\n\n".join(f"[{s.get('tool')}] {registry.for_model(s.get('output') or '')}"
                                  for s in rec.steps if s.get("output"))
            problems = []
            if bad:
                problems.append(f"These values appear in no tool result: {bad}. Use the values "
                                "the results give, or remove them.")
            if bad_ids:
                problems.append("These identifiers are not supported by this request's results: "
                                + "; ".join(f"{b['id']} ({b['reason']})" for b in bad_ids)
                                + ". Give a model only the ID its own result states.")
            if made_up or blanks:
                problems.append(f"These titles or template texts appear in no tool result: "
                                f"{made_up + blanks}. List only what the tools returned.")
            if undecided:
                problems.append(
                    f"The answer concludes that the model shows no {' / '.join(undecided)}, but no "
                    "analysis in this run decided that: the methods that ran did not apply, failed, "
                    "or left it open. Say that it was not decided and why; never conclude that the "
                    "behaviour is absent.")
            if no_point:
                problems.append(
                    "The answer says a Hopf, fold or other bifurcation point was found, but no "
                    "bifurcation analysis in this run found one (a screen's 'expected' type is a "
                    "prediction, not a result). Say that the analysis found no such point.")
            if unchecked:
                problems.append(
                    "The answer calls the model bistable from CRNT alone. Candidate sets do not "
                    "establish two stable states: report the CRNT outcome as it is.")
            literals = list(bad) + [b["id"] for b in bad_ids] + list(made_up) + list(blanks)
            patterns = claim_patterns(undecided, bool(no_point), bool(unchecked))
            answer = self._fix_paragraphs(query, rec, answer, results, problems, literals, bad,
                                          evidence_numbers(evidence), patterns)
        still_ids = (unsupported_model_ids(answer, id_evidence, identities)
                     + unsupported_reference_ids(answer, evidence))
        if still_ids:       # never leave a model mislabelled: drop the IDs outright
            ids = sorted({b["id"] for b in still_ids})
            answer = strip_model_ids(answer, ids) + (
                f"\n\n_Note: removed identifier(s) {', '.join(ids)}, which no result of this "
                "request supports._")
        still_made_up = unsupported_titles(answer, run_evidence + history_texts) + placeholders(answer)
        if still_made_up:   # never pass on a paper no tool returned: drop those lines only
            kept = [ln for ln in answer.splitlines() if not any(x in ln for x in still_made_up)]
            answer = "\n".join(kept).strip() + (
                f"\n\n_Note: removed {len(answer.splitlines()) - len(kept)} line(s) quoting a "
                "title or text that no tool result of this request contains._")
        if stability_claim_without_check(answer, rec.steps):
            answer += ("\n\n_Note: the CRNT result does not establish two coexisting stable "
                       "steady states; candidate sets alone prove neither presence nor absence._")
        still_folds = unreported_folds(answer, rec.steps)
        if still_folds:
            answer += "\n\n_Note: the bifurcation diagram also found " + "; ".join(still_folds) + "._"
        still_undecided = undecided_absence_claims(answer, rec.steps)
        if still_undecided:
            answer += ("\n\n_Note: no analysis in this run decided whether the model shows "
                       f"{' / '.join(still_undecided)}; the methods that ran did not apply or "
                       "failed, so its absence is not shown._")
        still = unsupported_values(answer, evidence)
        if still:
            answer += ("\n\n_Note: these values could not be traced to a tool result: "
                       + ", ".join(still) + "._")
        rec.final_answer = answer
        rec.final_check.update(unsupported_after_rewrite=still,
                               model_ids_removed=[b["id"] for b in still_ids],
                               undecided_after_rewrite=still_undecided)

    def _review_answer(self, query, history, rec):
        """The reviewer reads the answer against this request's results (one model call) and
        returns the claims none supports (empty when all are supported)."""
        steps = [s for s in rec.steps if s.get("output")]
        earlier = [m.get("content") or "" for m in (history or []) if m.get("role") == "assistant"]
        llm = self._llms.get("reviewer") if self._llms_given else self._llm("orchestrator")
        if llm is None:   # no reviewer is configured (for example, a scripted offline run)
            return []
        results = "\n\n".join(f"[step {i}: {s.get('tool')}] {registry.for_model(s['output'])}"
                              for i, s in enumerate(steps, 1))
        if earlier:         # what earlier turns established ("as before", "the other parameter")
            results += "\n\nEarlier answers in this chat:\n\n" + "\n\n".join(earlier[-5:])
        issues = None
        for attempt in (1, 2):     # a reply that is not JSON gets one more try
            try:
                text = _text(llm.invoke([SystemMessage(ANSWER_REVIEW_PROMPT), HumanMessage(
                    f"Question: {query}\n\nTool results:\n{results}\n\nAnswer:\n{rec.final_answer}")]))
                self._count(rec, "reviewer")
                issues = json.loads(text.strip().removeprefix("```json").removeprefix("```")
                                    .removesuffix("```"))["issues"]
                break
            except Exception as e:     # a failed review must never cost the answer
                rec.answer_review = {"error": f"{type(e).__name__}: {e}"[:300], "attempts": attempt}
        if issues is None:
            return []
        issues = [i for i in issues if isinstance(i, dict) and i.get("claim")]
        rec.answer_review = {"issues": issues, "revised": False}
        if not issues:
            return []
        self._log(f"ANSWER REVIEW: {issues}")
        return issues

    def _fix_paragraphs(self, query, rec, answer, results, problems, literals, bad,
                        evidence_nums, patterns=()) -> str:
        """The answer with only the paragraphs that hold a problem rewritten (one model
        call). A rewrite that lost a plot or a supported number is not taken."""
        blocks = split_blocks(answer)
        idx = blocks_with(blocks, literals, patterns)
        if not idx:
            return answer
        messages = [
            SystemMessage(
                "You correct parts of an answer. Each numbered paragraph below contains a problem "
                "listed after them. Return only a JSON object mapping each paragraph number to its "
                "corrected text. Change only what the problems require: keep every other number, "
                "table row, plot link and sentence exactly as written. A paragraph that is nothing "
                "but the wrong content may become an empty string."),
            HumanMessage(f"User request: {query}\n\nTool results:\n{results}\n\n"
                         f"The whole answer, for context:\n{answer}\n\nParagraphs to correct:\n"
                         + "\n\n".join(f"[{i}]\n{blocks[i]}" for i in idx)
                         + "\n\nProblems:\n" + "\n".join(problems))]
        try:
            text = _text(_invoke(self._llm("orchestrator"), messages))
            self._count(rec, "orchestrator")
        except Exception as e:
            rec.final_check["rewrite_error"] = f"{type(e).__name__}: {e}"[:300]
            return answer
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            fixed = json.loads(text)
        except ValueError:
            if len(idx) != 1:
                rec.final_check["rewrite_error"] = "the correction was not the JSON asked for"
                return answer
            fixed = {str(idx[0]): text}
        for i in idx:
            new = fixed.get(str(i), fixed.get(i)) if isinstance(fixed, dict) else None
            if isinstance(new, str) and keeps_content(blocks[i], new, set(bad), evidence_nums):
                blocks[i] = new.strip()
                rec.final_check["rewritten"] = True
        return "\n\n".join(b for b in blocks if b.strip())

    def _fallback_answer(self, rec) -> str:
        lines = ["No final answer was written. Results so far:"]
        for s in rec.steps:
            if s.get("tool"):
                lines.append(f"- {s.get('sub_agent') or 'orchestrator'}: {s['tool']} -> "
                             f"{s.get('status')}" + (f" ({s['error'][:200]})" if s.get("error") else ""))
        lines += [f"![plot]({p})" for p in rec.plots()]
        return "\n".join(lines)

    def _save(self, rec):
        runs = config.OUTPUT_DIR / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        path = runs / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        rec.record_path = str(path)
        path.write_text(json.dumps(asdict(rec), indent=2, default=str))


class _TeamScope(frozenset):
    """A team supervisor's members, with the index of its task in rec.team_runs, so each
    step records the team task it belongs to."""
    team_run = None


def _team_scope(members, index):
    scope = _TeamScope(members)
    scope.team_run = index
    return scope


def _capture(fn, *args):
    """(result, None) or (None, the exception): a worker thread's outcome."""
    try:
        return fn(*args), None
    except Exception as e:      # re-raised by the caller in the supervisor's thread
        return None, e


def _superseded(step, later) -> bool:
    """A failure that a later success in the same team task replaced (another sub-agent or
    tool took over), or a warning on a result the same tool later redid successfully."""
    same_task = [s for s in later if s.get("team_run") == step.get("team_run")
                 and s.get("status") == "ok"]
    if step.get("status") != "ok":
        return bool(same_task)
    return any(s.get("tool") == step.get("tool") for s in same_task)


def _review_note(issues) -> str:
    return ("ANSWER REVIEW: the following evidence, completeness, or reasoning issues were "
            "identified:\n"
            + "\n".join(f"- \"{i['claim']}\": {i.get('why', '')}" for i in issues)
            + "\nRun any missing analysis that an objective of the user's question needs and that "
              "is available; an analysis beyond the question is proposed, not run. Otherwise "
              "correct, qualify, or remove the affected conclusion. Then write the whole answer "
              "again. Do not substitute assumptions for unavailable evidence. Write it for the "
              "user: do not mention this review.")


def _problem_notes(steps, answer) -> str:
    """Tool errors and reviewer warnings of the results the answer rests on, stated by code so
    none is lost in a summary; failures or warnings that a later success superseded are left
    out, as are warnings the answer already states."""
    calls = [s for s in steps if s.get("tool") and s.get("status") != "rejected"]
    failed, warned = [], []
    for i, s in enumerate(calls):
        if _superseded(s, calls[i + 1:]):
            continue
        if s.get("status") in ("error", "unresolved", "sent_back"):
            why = s.get("error") or "; ".join((s.get("reviewer") or {}).get("issues") or [])
            failed.append(f"{s['tool']} ({why[:160]})")
        else:
            for issue in (s.get("reviewer") or {}).get("issues") or []:
                if issue[:40] not in (answer or "") and issue not in warned:
                    warned.append(issue)
    out = ""
    if failed:
        out += ("\n\n_Did not succeed: " + "; ".join(dict.fromkeys(failed))
                + ". Nothing in this answer comes from these._")
    if warned:
        out += "\n\n_Reviewer warnings: " + "; ".join(warned) + "._"
    return out


def _text(ai) -> str:
    c = ai.content
    if isinstance(c, list):  # e.g. Anthropic content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return c or ""
