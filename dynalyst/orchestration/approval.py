"""Asking the user mid-run (one of the agent's fixed exceptions: a long, expensive run is
the user's choice): before AUTO-07p when no parameter is named, before continuing the
limit cycles from a Hopf point, and when a CRNT search runs long.

When a bifurcation sub-agent is about to run AUTO on a model and its call names no
continuation parameter, the run pauses: the ranked candidates
(analysis/candidates.py) are shown and the user picks one -- "yes" runs the
first, a number or a parameter name picks another, "no" keeps the report only.
The choice is kept in the shared state, so the NEXT message answers it.
Only an agent a person is chatting with (Agent(approval="ask"), as the chat entry
points create it) pauses; called from code (scripts, evaluations, tests) it does not.
"""
import re

# tools that run AUTO and choose a continuation parameter themselves when none is given
GATED_TOOLS = {"run_bifurcation_analysis", "lhs_bifurcation_scan", "categorized_lhs_scan"}
# Match the WHOLE reply. A word such as "run", a candidate name, or a number
# inside a new request must never approve the previous model's pending task.
_YES = r"(?:yes|y|ok|okay|sure|fine|yep|yeah|approved)"
_ACTION = (r"(?:go(?: ahead| on)?|run(?: it)?|do(?: it)?|approve|proceed|continue"
           r"(?: (?:with )?(?:the )?limit cycles)?|carry on)")
_POLITE_END = r"(?:[, ]+(?:please|thanks|thank you))?"
_AFFIRM = re.compile(rf"(?:please )?(?:{_YES}(?:[, ]+{_ACTION})?|{_ACTION}){_POLITE_END}", re.I)
_REJECT = re.compile(
    rf"(?:please )?(?:no|n|nope|cancel|stop|don't|dont|skip|"
    rf"(?:just |only )?(?:the )?report(?: only)?){_POLITE_END}", re.I)
_CHOICE_PREFIX = (rf"(?:{_YES}[, ]+)?(?:please )?"
                  r"(?:(?:use|choose|select|pick) |run(?: it in| in| with)? |continue (?:in|with) )?"
                  r"(?:(?:candidate|option|parameter)\s*#?\s*)?")


def _reply_text(reply: str) -> str:
    return re.sub(r"\s+", " ", (reply or "").replace("\u2019", "'").strip()).strip(".!?,; ")


def _is_choice(text: str, name: str, *, ignore_case: bool = False) -> bool:
    # English choice words are case-insensitive; parse_name keeps model names exact.
    key = re.escape(name)
    if ignore_case:
        key = rf"(?i:{key})"
    return bool(re.fullmatch(rf"(?i:{_CHOICE_PREFIX})`?{key}`?(?i:{_POLITE_END})", text))


class AwaitApproval(Exception):
    """Raised inside a run to stop it and ask the user."""

    def __init__(self, message, pending):
        super().__init__(message)
        self.message, self.pending = message, pending


def needed(tool, args, mode: str = "auto") -> bool:
    """Pause before this call? mode: the agent's approval mode ("ask" / "auto")."""
    if mode != "ask" or tool not in GATED_TOOLS:
        return False
    model = args.get("model") or args.get("model_name_or_path")
    if not model:
        return False
    return not (args.get("bif_param") or args.get("param_name"))


def user_named_parameter(text: str, model_arg, param: str | None = None) -> bool:
    """Did the user's own words name this parameter (param), or with param None any of the
    model's parameters? The model may fill one in itself (after a parameter screen, or a
    guess): that is not the user's choice. A model that cannot be read here counts as named
    (the tool then reports the problem)."""
    from dynalyst.orchestration.inputs import _model_for
    model = _model_for(str(model_arg)) if model_arg else None
    if not model or not model.get("param_names"):
        return True
    return any(name_ends(text, p) for p in ([param] if param else model["param_names"]))


# names that are also ordinary words ("a", "I", "k", "time"): they count as named only where
# they are plainly a parameter, or "Draw a bifurcation diagram" would name the parameter a
_WORDLIKE = {"and", "the", "for", "all", "any", "not", "can", "has", "per", "off", "out", "one",
             "max", "min", "time", "rate", "total"}


def name_ends(text: str, p: str) -> list:
    """Where the user's words name parameter p (the end of each mention). A name of three or
    more letters that is not an ordinary word: anywhere, any case. A short or word-like one
    (a, I, k, time): written as given, and quoted, or followed by =, a colon, 'from', 'over',
    'between' or a range, or after 'along', 'vary', 'versus', 'sweep', 'parameter'."""
    t, e = text or "", re.escape(p)
    if len(p) > 2 and p.lower() not in _WORDLIKE:
        return [m.end() for m in re.finditer(rf"(?<![\w.]){e}(?![\w])", t, re.I)]
    cue = "|".join(f"(?<={w} )" for w in ("along", "versus", "vs", "vs.", "vary", "varying",
                                           "sweep", "sweeping", "parameter", "param"))
    return [m.end() for m in re.finditer(
        rf"(?<=[`'\"]){e}(?=[`'\"])"
        rf"|(?<![\w.]){e}(?=\s*(?:=|∈|:|from\b|over\b|between\b|in\s*[\[(]|[\[(]\s*-?\d))"
        rf"|(?:{cue}){e}(?![\w])", t)]


def needed_for_user(tool, args, user_text: str, mode: str = "auto") -> bool:
    """Pause before an AUTO-07P run (a diagram or an LHS scan) whose continuation parameter the
    call names but the USER's own words do not (the model picked it, even with another of the
    model's parameters in the request): the user picks from the ranked candidates. A call that
    names none is paused by needed()."""
    model = args.get("model") or args.get("model_name_or_path")
    param = args.get("param_name") or args.get("bif_param")
    return (mode == "ask" and tool in GATED_TOOLS and bool(model) and bool(param)
            and not user_named_parameter(user_text, model, str(param)))


_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_RANGES = re.compile(
    rf"[\[(]\s*({_NUM})\s*[,;]\s*({_NUM})\s*[\])]"
    rf"|\bfrom\s+({_NUM})\s+(?:to|until|up to)\s+({_NUM})"
    rf"|\bbetween\s+({_NUM})\s+and\s+({_NUM})"
    rf"|(?<![\w.\-])({_NUM})\s*(?:to|-|–|\.\.)\s*({_NUM})(?![\w.])", re.I)


def user_ranges(text: str, param: str | None = None) -> list:
    """The parameter ranges the user's own words give: [a, b], from a to b, between a and b,
    a to b, a-b; with param, only a range written within a few words after that parameter's
    name ("MAPKK over [1, 10]", "along stress from 0 to 6")."""
    near = name_ends(text, param) if param else None
    out = []
    for m in _RANGES.finditer(text or ""):
        if near is not None and not any(0 <= m.start() - e <= 40 for e in near):
            continue
        a, b = (float(g) for g in m.groups() if g is not None)
        if a != b:
            out.append((min(a, b), max(a, b)))
    return out


def widens_user_range(tool, args, user_text: str, mode: str = "auto") -> tuple | None:
    """In a chat: the user's range for the swept parameter that a diagram or LHS call does not
    keep (the model widened, moved or left it out), or None. The user is asked about a wider
    range after the answer, not overruled."""
    param = args.get("param_name") or args.get("bif_param")
    if mode != "ask" or tool not in GATED_TOOLS or not param:
        return None
    lo, hi = args.get("param_min"), args.get("param_max")
    ranges = user_ranges(user_text, str(param))
    if not ranges:
        return None
    if lo is None or hi is None:        # left out: the tool's own window is not the user's
        return ranges[0]
    if any(abs(float(lo) - a) <= 1e-9 * max(1, abs(a)) and abs(float(hi) - b) <= 1e-9 * max(1, abs(b))
           for a, b in ranges):
        return None
    return ranges[0]


def wider_range(lo: float, hi: float, own: float | None) -> tuple:
    """The range proposed when the user's holds no special point: it reaches a decade past
    the model's own value of the parameter (and past the user's range), keeping its sign."""
    width = hi - lo
    if own is None or own == 0:
        return (lo - width if lo < 0 else max(0.0, lo - width), hi + width)
    a = min(lo, own / 10) if own > 0 else min(lo, own * 10)
    b = max(hi, own * 10) if own > 0 else max(hi, own / 10)
    if (a, b) == (lo, hi):
        a, b = (max(0.0, lo - width) if lo >= 0 else lo - width), hi + width
    return float(a), float(b)


def ambiguous_phrase(text: str, param_names) -> tuple | None:
    """A word of the request that stands for several parameters (kcat for k1cat ... k4cat,
    Km for Km1 ... Km5) with none of them named exactly: (word, [those parameters])."""
    names = list(param_names or [])
    if any(name_ends(text, p) for p in names):
        return None
    for w in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z_]+", text or "")):
        hits = [p for p in names if re.sub(r"[\d_]+", "", p).lower() == w.replace("_", "").lower()]
        if len(hits) > 1:
            return w, hits
    return None


def message(model, tool, res, picked: str | None = None) -> str:
    from dynalyst.analysis.candidates import format_candidates
    listing = format_candidates(res, species=tool != "run_bifurcation_analysis")
    why = (f"the agent picked **{picked}** as the continuation parameter, but your request does "
           f"not name it" if picked else "no continuation parameter was named")
    return (f"Before running AUTO-07p on **{model}**: {why}, so "
            f"here are the ranked candidates ({res.get('method', 'eigenvalue screen')}):\n\n"
            f"{listing}\n\n" + (f"{res['note']}\n\n" if res.get("note") else "")
            + f"Reply **yes** to run candidate 1, a **number** or a **parameter "
            f"name** to run another, or **no** to keep this report only (no AUTO run).")


def parse_name(reply: str, candidates: list) -> tuple[str, int | None]:
    """Approve an entire candidate-selection reply, never a name inside a new task."""
    text = _reply_text(reply)
    named = [i for i, c in enumerate(candidates) if c.get("parameter")
             and _is_choice(text, c["parameter"])]
    return ("approve", named[0]) if len(named) == 1 else ("other", None)


def parse(reply: str, candidates: list) -> tuple[str, int | None]:
    """('approve', index) | ('reject', None) | ('other', None) for the user's reply."""
    text = _reply_text(reply)
    if _REJECT.fullmatch(text):
        return "reject", None
    named = [i for i, c in enumerate(candidates) if c.get("parameter")
             and _is_choice(text, c["parameter"], ignore_case=True)]
    if len(named) == 1:
        return "approve", named[0]
    if named:  # ambiguous duplicate names must not fall through to another choice
        return "other", None
    for i in range(len(candidates)):
        if _is_choice(text, str(i + 1)):
            return "approve", i
    if _AFFIRM.fullmatch(text):
        return "approve", 0
    return "other", None


_CONTINUE = re.compile(
    rf"(?:please )?(?:continue(?: (?:the )?(?:search|CRNT search))?|wait|finish|"
    rf"keep (?:searching|waiting)|let it (?:run|finish)|go on|carry on|{_YES}|search)"
    rf"(?:,? then plot)?{_POLITE_END}", re.I)
_PLOT = re.compile(
    rf"(?:please )?(?:(?:stop|cancel)(?: (?:the )?(?:search|CRNT search))?(?: and plot)?|"
    rf"plot(?: (?:the )?(?:diagram|bifurcation diagram))?(?: now)?|diagram|bifurcation|"
    rf"now|no|n|skip|don'?t wait){_POLITE_END}", re.I)


def parse_crnt_wait(reply: str) -> str:
    """'continue' (wait for the CRNT search) | 'plot' (stop it, draw the diagram now) |
    'other' (a new request) for the user's reply to the CRNT question."""
    text = _reply_text(reply)
    if _PLOT.fullmatch(text):
        return "plot"
    if _CONTINUE.fullmatch(text):
        return "continue"
    return "other"


def budget_message(what, limit) -> str:
    return (f"This request has reached its limit of **{limit} {what}**.\n\n"
            "Reply **yes** to allow that many more, or **no** to get the answer from what ran.")


def stochastic_wait_message(minutes, progress) -> str:
    sizes = lambda key: ", ".join(f"{x:g}" for x in progress[key]) or "none"
    return (f"The stochastic runs have taken **{minutes:.0f} min** so far. System sizes (omega): "
            f"finished {sizes('finished')}; stopped before t_end {sizes('stopped')}; not run "
            f"{sizes('not_run')}. The rest will take more time.\n\n"
            "Reply **yes** to continue them, or **no** to stop and get the answer from what ran.")


def crnt_wait_message(model, res) -> str:
    net = res.get("network") or {}
    size = (f" ({net.get('n_species')} species, {net.get('n_reactions')} reactions)"
            if net.get("n_species") else "")
    seconds = res.get("seconds") or 0
    waited = f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"
    return (f"CRNT has been searching **{model}**{size} for **{waited}** for rate "
            f"constants with several steady states and has not finished.\n\n"
            "Reply **continue** to let the search finish; its first candidate set is then "
            "checked for bistability with AUTO-07p and the bifurcation diagram plotted.\n"
            "Reply **plot** to stop the search and draw the bifurcation diagram of the model "
            "itself now.")


def directive(cand, tool=None) -> str:
    """What the approved run must use, appended to the original request. A one-diagram AUTO
    run names no species: the tool picks the y-axis after the run (the one that shows the
    branches apart), and a species named here would read as the user's request."""
    lo, hi = cand["range"]
    if tool == "run_bifurcation_analysis":
        return f"(Approved by the user: continue in {cand['parameter']} over [{lo:.6g}, {hi:.6g}].)"
    return (f"(Approved by the user: continue in {cand['parameter']} over [{lo:.6g}, {hi:.6g}], "
            + (f"plotting {cand['y_species']}.)" if cand.get("y_species")
               else "plotting the species that changes most.)"))


def inject(tool, args, cand, allowed) -> dict:
    """The approved candidate's parameter / range / species on a call that names none, or
    that names another parameter (the model's own pick, which the user was asked about): the
    user's choice holds, with its range."""
    named = args.get("bif_param") or args.get("param_name")
    if not cand or named == cand.get("parameter"):
        return args
    out = {k: v for k, v in args.items()
           if not named or k not in ("param_name", "bif_param", "param_min", "param_max")}
    lo, hi = cand["range"]
    for key, val in (("param_name", cand["parameter"]), ("bif_param", cand["parameter"]),
                     ("param_min", lo), ("param_max", hi)):
        if key in allowed and out.get(key) is None:
            out[key] = val
    for key in ("plot_state", "y_species"):
        if key in allowed and not out.get(key) and cand.get("y_species"):
            out[key] = cand["y_species"]
    return out
