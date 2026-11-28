"""Final-answer check (deterministic): every number the answer quotes must
appear in a tool output or the user's own messages -- allowing for
rounding. Small integers (<= 12: counts, list numbering) and digits inside
identifiers (k4, Author2004, file names) are not checked. Numbers written in
scientific notation (3.2 x 10^-5, 3.2 \\times 10^{-5}, 3.2·10⁻⁵, 10^{-5}) are read
as the value they denote, the same as 3.2e-5."""
import re

# dotted codes (EC 2.7.1.1, version 1.2.3) are identifiers, not values
_NUMBER = re.compile(r"(?<![\w.\-/])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w/]|\.\d)")
# an identifier inside a link or path in a result (pubmed/11744196, taxonomy/4932) supports
# the same identifier written on its own in the answer
_DIGITS = re.compile(r"(?<![\d.])\d+(?![\d.])")

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_EXPONENT = (r"(?:\^\s*\{?\s*\(?\s*(?P<e1>[-+]?\d+)\s*\)?\s*\}?"
             r"|\*\*\s*\(?\s*(?P<e2>[-+]?\d+)\s*\)?"
             r"|(?P<e3>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))")
_SCIENTIFIC = re.compile(
    r"(?<![\w.])(?P<m>[-+]?\d+(?:\.\d+)?)\s*(?:×|✕|\\times|\\cdot|·|⋅|\*|x|X)\s*10"
    # "4.5×10-4": a superscript exponent flattened onto the line
    r"(?:(?P<e4>-\d+)(?![\d.])|\s*" + _EXPONENT + ")")
_POWER_OF_TEN = re.compile(r"(?<![\w.×·⋅*])10\s*" + _EXPONENT)


def _exponent(m) -> str:
    groups = m.groupdict()
    return next((groups[g] for g in ("e1", "e2", "e3", "e4") if groups.get(g)), "").translate(
        _SUPERSCRIPT)


def normalise_numbers(text: str) -> str:
    """Scientific notation in any common written form turned into 3.2e-5 form;
    the Unicode minus sign into '-'."""
    text = (text or "").replace("−", "-").replace("$", " ")
    text = _SCIENTIFIC.sub(lambda m: f"{m.group('m')}e{_exponent(m)}", text)
    return _POWER_OF_TEN.sub(lambda m: f"1e{_exponent(m)}", text)


def numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(normalise_numbers(text))]


def _sig_digits(token: str) -> int:
    mantissa = re.split(r"[eE]", token.lstrip("+-"))[0].replace(".", "").lstrip("0")
    return max(len(mantissa), 1)


def _supported(token: str, evidence: list[float]) -> bool:
    v = float(token)
    digits = _sig_digits(token)
    for e in evidence:
        if e == v:
            return True
        if e != 0 and float(f"{e:.{digits}g}") == v:
            return True
        if abs(v - e) <= 0.01 * max(abs(e), 1e-300):
            return True
    return False


# ---------------------------------------------------------------------------
# model identifiers
# ---------------------------------------------------------------------------
# A BioModels ID next to a model name says which model a result is about, so a
# wrong one mislabels the whole answer -- e.g. a model named with the ID of a
# different model fetched earlier in the chat. The number check above cannot
# see these (digits inside identifiers are skipped).

MODEL_ID = re.compile(r"\b(BIOMD\d{10}|MODEL\d{8,12})\b")
_JSON_IDENTITY = re.compile(r'"file":\s*"([^"]+)",\s*"model_name":\s*(?:"(?:[^"\\]|\\.)*"|null),'
                            r'\s*"biomodels_id":\s*(null|"([^"]+)")')
_LINE_IDENTITY = re.compile(r"Model: (\S+?)(?: \(\"[^\"]*\"\))?(?:, BioModels ID (\S+) \(declared"
                            r"|;[^\n]*?(?:declares no|has no) BioModels ID)")


def _stem(name: str) -> str:
    name = name.lower().strip()
    for suffix in (".xml", ".txt", ".ode"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def model_identities(evidence_texts: list[str]) -> dict:
    """{model file stem (lower case): its BioModels ID or None} from the
    'model_identity' objects and 'Model: ...' lines the tools put in their results."""
    out = {}
    for t in evidence_texts:
        for m in _JSON_IDENTITY.finditer(t or ""):
            out[_stem(m.group(1))] = m.group(3)
        for m in _LINE_IDENTITY.finditer(t or ""):
            out[_stem(m.group(1))] = m.group(2)
    return out


def unsupported_model_ids(answer: str, evidence_texts: list[str], identities: dict) -> list[dict]:
    """BioModels IDs in the answer that this query's evidence does not support:
      * the ID appears in no tool result of this query, nor in the user's request;
      * or it is written next to a model whose result gives it a different ID (or none).
    Each is {'id', 'reason'}."""
    seen = set()
    for t in evidence_texts:
        seen.update(MODEL_ID.findall(t or ""))
    bad, flagged = [], set()
    low = answer.lower()
    for m in MODEL_ID.finditer(answer):
        mid = m.group(1)
        if mid in flagged:
            continue
        if mid not in seen:
            bad.append({"id": mid, "reason": "no result of this query mentions it"})
            flagged.add(mid)
            continue
        # the model named closest before the ID (same sentence, <= 80 characters)
        window_start = max(0, m.start() - 80)
        cut = max(low.rfind(". ", window_start, m.start()), low.rfind("\n", window_start, m.start()))
        window_start = max(window_start, cut + 1)
        best, best_pos = None, -1
        for stem in identities:
            pos = low.rfind(stem, window_start, m.start())
            if pos > best_pos:
                best, best_pos = stem, pos
        if best is not None and identities[best] != mid:
            known = identities[best]
            bad.append({"id": mid, "model": best,
                        "reason": (f"it is written next to {best}, whose file declares "
                                   + (f"{known}" if known else "no BioModels ID"))})
            flagged.add(mid)
    return bad


# PubMed IDs and UniProt accessions in an answer must come from this query's results too
_PMID_REF = re.compile(r"\bPMID[:\s]*(\d{5,9})\b|pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})", re.I)
_UNIPROT_REF = re.compile(r"\b(?:(?i:UniProt(?:KB)?(?:\s+(?:accession|ID|entry))?)[:\s]*|"
                          r"(?i:uniprot\.org/uniprotkb/))([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}"
                          r"[0-9]){1,2})\b")


def unsupported_reference_ids(answer: str, evidence_texts: list[str]) -> list[dict]:
    """PMIDs / UniProt accessions in the answer that no tool result (or the
    request) of this query contains: invented ones. Each is {'id', 'reason'};
    'id' is the whole reference as written, so it can be stripped."""
    seen = " ".join(t or "" for t in evidence_texts)
    bad = []
    for kind, pat in (("PubMed ID", _PMID_REF), ("UniProt accession", _UNIPROT_REF)):
        for m in pat.finditer(answer):
            value = next(g for g in m.groups() if g)
            ref = m.group(0).strip()
            if (not re.search(rf"(?<!\w){re.escape(value)}(?!\w)", seen)
                    and ref not in [b["id"] for b in bad]):
                bad.append({"id": ref, "reason": f"{kind} {value} is in no result of this query"})
    return bad


def strip_model_ids(answer: str, ids) -> str:
    """Remove the given IDs from an answer, with a parenthesis that held only them."""
    for mid in ids:
        answer = re.sub(r"\s*\(\s*" + re.escape(mid) + r"\s*\)", "", answer)
        answer = re.sub(r"\s*\b" + re.escape(mid) + r"\b", "", answer)
    return answer


def evidence_numbers(evidence_texts: list[str]) -> list[float]:
    """Every number the tool results, the request and the plan contain."""
    evidence = []
    for t in evidence_texts:
        for tok in numbers_in(t) + _DIGITS.findall(t or ""):
            try:
                evidence.append(float(tok))
            except ValueError:
                pass
    return evidence


def unsupported_values(answer: str, evidence_texts: list[str]) -> list[str]:
    evidence = evidence_numbers(evidence_texts)
    out = []
    # thousands separators in prose ("2,986.47") are one number, not 2 and 986.47
    tokens = numbers_in(re.sub(r"(?<=\d),(?=\d{3}(?![\d,]))", "", answer or ""))
    # a percentage written for a fraction a result gives ("7.51%" for 0.0751) is supported
    percents = set(re.findall(r"(\d+(?:\.\d+)?)\s*%", answer or ""))
    for tok in tokens:
        v = float(tok)
        if v.is_integer() and abs(v) <= 12:
            continue
        if tok in percents and _supported(f"{v / 100:.{_sig_digits(tok)}g}", evidence):
            continue
        if not _supported(tok, evidence) and tok not in out:
            out.append(tok)
    if out:
        # simple arithmetic on two values the answer quotes from the results (a window's
        # width hi - lo, a fold change hi / lo, a share 19 of 20 = 95 %, a change of 20 %)
        # is supported by them
        # every number the answer writes that the results support, ends of "39.2-57.4" included
        quoted = [float(x) for x in re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                                               normalise_numbers(answer or ""))
                  if _supported(x, evidence)][:80]
        derived = [f(a, b) for a in quoted for b in quoted if a != b
                   for f in (lambda x, y: x - y, lambda x, y: x + y,
                             lambda x, y: x / y if y else x, lambda x, y: x * y,
                             lambda x, y: 100 * x / y if y else x,
                             lambda x, y: 100 * (x - y) / y if y else x)]
        out = [t for t in out if not _supported(t, derived)]
    return out


def unsupported_points(series, evidence_texts: list[str]) -> list[float]:
    """Numbers in custom_plot's series that no earlier tool result (or the user) gave: a plot
    must not show values nobody computed. A malformed series is left to the tool to refuse."""
    evidence = evidence_numbers(evidence_texts)
    out = []
    for s in series if isinstance(series, list) else []:
        for key in ("x", "y"):
            for v in (s.get(key) if isinstance(s, dict) and isinstance(s.get(key), list) else []):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if not _supported(f"{f:g}", evidence) and f not in out:
                    out.append(f)
    return out


# A conclusion that a behaviour is absent ("not bistable", "does not oscillate") needs an
# analysis that could have shown it. A method that did not apply or failed shows nothing.
_ABSENT = {
    "bistability": re.compile(
        r"\b(?:is|was|are|were|be)\s+not\s+(?:bi|multi)stable\b|\bnot\s+(?:bi|multi)stable\b"
        r"|\b(?:shows?|has|have|exhibits?)\s+no\s+(?:bi|multi)stab\w*|\bmonostable\b"
        r"|\bno\s+(?:bi|multi)stab\w*\s+(?:is|was)\s+(?:found|present|possible)", re.I),
    "oscillation": re.compile(
        r"\b(?:does|do|did)\s+not\s+oscillate\b|\bis\s+not\s+oscillat\w*"
        r"|\b(?:shows?|has|have|exhibits?)\s+no\s+oscillation\w*|\bnon-?oscillat\w*", re.I),
}


_ACCEPTED_STATUS = ("ok",)    # the tool succeeded and the reviewer accepted it


def _decided(step: dict) -> bool:
    """Did this analysis step run to a result that can decide a behaviour? Accepted by
    the reviewer, not an error, and not a verdict left open (bistable: null)."""
    import json
    if step.get("kind") != "delegation" or step.get("status") not in _ACCEPTED_STATUS:
        return False
    out = step.get("output") or ""
    try:
        data = json.loads(out)
    except (TypeError, ValueError):
        return bool(out.strip()) and not out.lstrip().lower().startswith("error")
    if not isinstance(data, dict):
        return True
    if data.get("error"):
        return False
    if step.get("tool") == "crnt_bistability":
        return bool(data.get("proof") and data.get("bistable") is False)
    return not ("bistable" in data and data["bistable"] is None)


def undecided_absence_claims(answer: str, steps: list) -> list[str]:
    """Behaviours the answer says are absent although no analysis in this run decided
    anything (every step failed, did not apply, or left the question open)."""
    if any(_decided(s) for s in steps or []):
        return []
    return [kind for kind, rx in _ABSENT.items() if rx.search(answer or "")]


# CRNT finds rate constants with several steady states; it does not check their stability.
# An answer whose only analyses are network-structure ones may not call the model bistable.
_STRUCTURE_ONLY_TOOLS = {"crnt_bistability", "draw_reaction_network"}
_BISTABLE_CLAIM = re.compile(
    r"\b(?:is|are|was|be)\s+(?:confirmed\s+(?:to\s+be\s+)?)?bistable\b"
    r"|\bbistability\s+(?:is|was)\s+(?:confirmed|shown|present)"
    r"|\b(?:shows?|exhibits?|has|have)\s+bistability\b", re.I)


def stability_claim_without_check(answer: str, steps: list) -> bool:
    """The answer calls the model bistable, but every analysis that ran was a network-
    structure one (CRNT), which shows several steady states, not that two are stable."""
    import json
    ran = [s for s in steps or [] if s.get("kind") == "delegation"
           and s.get("status") in _ACCEPTED_STATUS]
    if not ran or any(s.get("tool") not in _STRUCTURE_ONLY_TOOLS for s in ran):
        return False
    text = answer or ""
    for m in _BISTABLE_CLAIM.finditer(text):
        before = text[max(0, m.start() - 30):m.start()].lower()
        if re.search(r"\b(?:not|cannot|can't|never|no|non|neither|nor|whether|if|may|might|could"
                     r"|can|would|potential\w*|candidate|possibl\w*)\b(?:\W+\w+){0,3}\W*$",
                     before) or re.match(r"\W*non-", text[m.start():].lower()):
            continue          # "cannot exhibit bistability", "whether it is bistable"
        return True
    return False


# A Hopf point exists when a bifurcation analysis found one, not when a parameter screen
# expected one.
_HOPF_CLAIM = re.compile(
    r"\bHopf\s+(?:bifurcation\s+)?points?\s+(?:is|are|was|were)\s+(?:present|found|detected|identified)"
    r"|\b(?:has|have|shows?|exhibits?|found|detected|identified)\s+(?:an?\s+|two\s+|\d+\s+)?Hopf"
    r"|\bHopf\s+(?:bifurcation\s+)?points?\s+at\b", re.I)


# a claim inside a clause that negates or hedges it ("No branch points or Hopf points were
# found", "whether a fold exists") is not a claim; the clause ends at a sentence boundary,
# ';', or a contrast ("..., but a fold was found" is a claim)
_CLAUSE_START = re.compile(r"[.!?;\n]|\b(?:but|while|whereas|however)\b", re.I)
_NEGATOR = re.compile(r"\b(?:not|no|none|neither|nor|never|without|whether|if|cannot|"
                      r"expected|candidate|predict\w*)\b", re.I)


def _negated(text: str, start: int) -> bool:
    before = text[:start]
    cut = max((m.end() for m in _CLAUSE_START.finditer(before)), default=0)
    return bool(_NEGATOR.search(before[cut:]))


def unfound_hopf_claim(answer: str, steps: list) -> bool:
    """The answer says a Hopf point exists, but no accepted analysis reported an HB point."""
    text = answer or ""
    if any('HB' in reported_point_types(s) for s in steps or []):
        return False
    for m in _HOPF_CLAIM.finditer(text):
        if _negated(text, m.start()):
            continue
        return True
    return False


_BIF_CLAIM = re.compile(
    r"\b(?:identified|found|detected|located|revealed|shows?|has|have|exhibits?)\s+"
    r"(?:an?\s+|two\s+|\d+\s+|a\s+significant\s+|significant\s+)?(?:bifurcation|fold|saddle-node|"
    r"limit|branch|transcritical|pitchfork)(?:\s+(?:bifurcation\s+)?points?)?\b"
    r"|\b(?:bifurcation|fold|saddle-node|limit|branch)\s+points?\s+(?:is|are|was|were)\s+"
    r"(?:present|found|detected|identified|located)", re.I)
# a point AUTO-07p reported: 'LP' in a result, or an LHS scan's signature ('2 LP (bistable)',
# '1 BP + 1 HB')
_AUTO_POINT = re.compile(r"""["'](?:LP|BP|HB)["']|["']\d+ (?:LP|BP|HB)\b""")


def unfound_bifurcation_claim(answer: str, steps: list) -> bool:
    """The answer says a fold / branch point / bifurcation point was found, but no
    analysis reported one (a parameter screen's crossing is a prediction, not a result)."""
    if any(reported_point_types(s) for s in steps or []):
        return False
    text = answer or ""
    for m in _BIF_CLAIM.finditer(text):
        if _negated(text, m.start()):
            continue
        return True
    return False


def reported_point_types(step):
    """Point records count as evidence; zero-count headings and failed calls do not."""
    import json
    if (step.get('kind') != 'delegation' or step.get('tool') == 'propose_bifurcation_candidates'
            or step.get('status') not in (None, 'ok')):
        return set()
    text = step.get('output') or ''
    try:
        result = json.loads(text)
    except (TypeError,ValueError):
        details = re.search(r'^Special point details:\s*(\[.*\])\s*$', text, re.M)
        if details:
            try:
                points = json.loads(details.group(1))
                return {p.get('type') for p in points
                        if isinstance(p,dict) and p.get('type') in ('LP','BP','HB')}
            except ValueError:
                pass
        return set(re.findall(r'\b[1-9]\d* (LP|BP|HB)\b',text))
    if not isinstance(result,dict) or result.get('error'):
        return set()
    points = result.get('bifurcation_points',result.get('special_points',[]))
    return {p.get('type') for p in points if isinstance(p,dict) and p.get('type') in ('LP','BP','HB')}


def qualify_empty_continuation(answer, steps):
    """Keep a no-point result tied to its actual parameter and computed window."""
    import json
    if not re.search(r'\bno\s+(?:detected\s+)?bifurcation|\b(?:never|cannot)\s+bifurcat',answer,re.I):
        return answer
    notes, results = [], []
    for step in steps or []:
        if step.get('tool') != 'run_bifurcation_analysis' or step.get('status') not in _ACCEPTED_STATUS:
            continue
        try:
            results.append(json.loads(step.get('output') or '{}'))
        except (TypeError,ValueError):
            continue
    # a run that found points for the same parameter (e.g. on another branch) makes the note false
    found = {r.get('swept_parameter') for r in results if r.get('bifurcation_points')}
    for result in results:
        if result.get('error') or result.get('bifurcation_points') != []:
            continue
        parameter, window = result.get('swept_parameter'), result.get('swept_range')
        if parameter and window and parameter not in found:
            notes.append(f"No special point was detected on the computed branches for {parameter} "
                         f"in {window}. This is not a global absence result; any reported incomplete "
                         "coverage also limits that statement.")
    return answer + ('\n\n'+'\n'.join(dict.fromkeys(notes)) if notes else '')


# --- fixing only what is wrong ------------------------------------------------------------
# A whole-answer rewrite can lose what was right (tables, plots, numbers): only the
# paragraphs that hold a problem are rewritten, and a rewritten paragraph must keep what it had.

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)\s]+\)")
_BISTABILITY_MENTION = re.compile(r"\b(?:bi|multi)stab\w*", re.I)


def split_blocks(text: str) -> list[str]:
    """An answer's paragraphs (tables, lists and plots stay whole), joined back with '\\n\\n'."""
    return re.split(r"\n\s*\n", (text or "").strip())


def blocks_with(blocks: list[str], literals=(), patterns=()) -> list[int]:
    """Indices of the paragraphs that contain one of the literal strings or match a pattern."""
    out = []
    for i, b in enumerate(blocks):
        norm = normalise_numbers(b)
        if (any(lit and (lit in b or lit in norm) for lit in literals)
                or any(p.search(b) for p in patterns)):
            out.append(i)
    return out


def claim_patterns(undecided=(), no_hopf=False, crnt_only=False) -> list:
    """The patterns that find the sentences making an unsupported claim."""
    pats = [_ABSENT[b] for b in undecided if b in _ABSENT]
    if no_hopf:
        pats += [_HOPF_CLAIM, _BIF_CLAIM]
    if crnt_only:
        pats.append(_BISTABILITY_MENTION)
    return pats


def keeps_content(old: str, new: str, wrong_numbers, evidence: list[float]) -> bool:
    """A rewritten paragraph keeps every plot and every number of the old one that the
    results support (only the wrong ones may go). evidence: evidence_numbers(...)."""
    if any(img not in new for img in _IMAGE.findall(old or "")):
        return False
    have = [float(t) for t in re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", normalise_numbers(new))]
    for tok in re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", normalise_numbers(old or "")):
        if tok in wrong_numbers or (float(tok).is_integer() and abs(float(tok)) <= 12):
            continue
        if _supported(tok, evidence) and not _supported(tok, have):
            return False
    return True


_MENTIONS_FOLD = re.compile(r"\b(?:fold|saddle[- ]node|limit points?|LP|bistab\w*|hysteresis|"
                            r"two stable|coexist\w*)\b", re.I)


def unreported_folds(answer: str, steps: list) -> list[str]:
    """What a bifurcation diagram of this run found about folds and bistability that the
    answer does not mention at all ("2 folds (LP) in k1AA; 2 stable steady states coexist
    at k1AA = 0.2988"): a question about one bifurcation type still gets the others the
    diagram shows."""
    import json as _json
    if _MENTIONS_FOLD.search(answer or ""):
        return []
    out = []
    for s in steps or []:
        if s.get("kind") != "delegation" or s.get("tool") != "run_bifurcation_analysis":
            continue
        try:
            data = _json.loads(s.get("output") or "")
        except (TypeError, ValueError):
            continue
        cls = (data.get("classification") or {}) if isinstance(data, dict) else {}
        ev = cls.get("evidence") or {}
        n_lp = (ev.get("counts") or {}).get("LP", 0)
        param = data.get("swept_parameter", "the parameter")
        if cls.get("category") == "bistable" and ev.get("at") is not None:
            item = (f"{n_lp} fold(s) (LP) in {param}; {ev.get('max_coexisting_stable', 2)} stable "
                    f"steady states coexist at {param} = {float(ev['at']):.6g} "
                    f"({cls.get('multistability') or 'bistable'})")
        elif n_lp:
            item = f"{n_lp} fold(s) (LP) in {param}"
        else:
            continue
        if item not in out:
            out.append(item)
    return out


# What an answer leaves out (the orchestrator's OUTPUT rules, applied in code as well): that
# the reviewer accepted a result, matrices the user did not ask for, and closing offers.
_ACCEPTED = re.compile(
    r"[^.!?\n]*\b(?:review(?:er|ed)?|verdict|checks?)\b[^.!?\n]*\baccept\w*[^.!?\n]*[.!?]?"
    r"|[^.!?\n]*\b(?:was|were|has been|have been)\s+(?:reviewed\s+and\s+)?accepted\b[^.!?\n]*[.!?]?",
    re.I)
_MATRIX = re.compile(r"(?:^[^\n]*:\s*\n)?\s*\\\[\s*\\begin\{[bpv]?matrix\}.*?\\end\{[bpv]?matrix\}"
                     r"\s*\\\]", re.S | re.M)
_OFFER = re.compile(r"\s*(?:If you (?:have|need|would like|want)[^.!?\n]*[.!?]|(?:Please )?let me know"
                    r"[^.!?\n]*[.!?]|Feel free[^.!?\n]*[.!?])", re.I)


# A revision's reply to the answer review, not part of the answer: "The review is right — I
# presented ...", "Let me write the corrected answer."
_REVIEW_REPLY = re.compile(
    r"^\W*(?:(?:the |your )?review(?:er)?(?:'s point)? (?:is|was) (?:right|correct|valid)"
    r"|(?:you are|you're) (?:right|correct)|(?:good|valid|fair) (?:point|catch)"
    r"|let me (?:now )?(?:write|rewrite|correct|revise|fix|qualify|redo|restate|update)"
    r"|(?:here is|below is|this is) (?:the|a|my) (?:corrected|revised|updated|rewritten|qualified) "
    r"(?:answer|version|response)"
    r"|I (?:need|have|should|must) to (?:correct|qualify|revise|fix|rewrite|restate)"
    r"|I (?:previously |incorrectly |wrongly |earlier )?(?:presented|stated|claimed|reported|wrote) "
    r"[^.!?\n]*\b(?:but|however|without|incorrectly|wrongly)\b"
    r"|(?:in response to|addressing|following|per) the (?:answer )?review)", re.I)
# a dot between digits (2.26) is part of a number, not the end of a sentence
_SENTENCE = re.compile(r"(?:[^.!?\n]|(?<=\d)\.(?=\d))+(?:[.!?]+|$)", re.M)


def _without_review_replies(text: str) -> str:
    """Sentences that answer the reviewer removed, unless one holds a number the rest of the
    answer does not: a result is never dropped with the reply around it."""
    for s in [m.group(0) for m in _SENTENCE.finditer(text) if _REVIEW_REPLY.search(m.group(0))]:
        rest = text.replace(s, "", 1)
        kept = {float(n) for n in _NUMBER.findall(rest)}
        if all(float(n) in kept for n in _NUMBER.findall(s)):
            text = rest
    return text


def tidy_answer(answer: str, query: str = "") -> str:
    """The answer without reviewer narration ("... was accepted without issues", replies to
    the answer review), matrices the request does not ask for, and closing offers. Warnings
    stay."""
    text = _without_review_replies(answer or "")
    text = _ACCEPTED.sub(lambda m: m.group(0) if re.search(r"warning|unresolved|fail",
                                                            m.group(0), re.I) else "", text)
    if not re.search(r"\b(?:jacobian|matrix|matrices)\b", query or "", re.I):
        text = _MATRIX.sub("", text)
    text = _OFFER.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# A paper's title in the answer must come from a tool result: a language model easily writes
# plausible titles and authors that do not exist.
_QUOTED_TITLE = re.compile(r"[\"\u201c]([^\"\u201d\n]{20,250})[\"\u201d]")
_TITLE_FIELD = re.compile(r"\btitle\**\s*:\**\s*[\"\u201c*]*([^\n\"\u201d*]{12,250})", re.I)


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


# a quoted phrase is a paper's title only where the line cites a paper (a year, authors,
# a DOI or PubMed ID); elsewhere it quotes a result ("subcritical: the cycles born ...")
_CITATION = re.compile(r"\b(?:19|20)\d{2}\b|\bet al\b|\bdoi\b|doi\.org|\bPMID\b|pubmed", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def unsupported_titles(answer: str, evidence_texts: list[str]) -> list[str]:
    """Paper titles quoted in the answer (in quotes on a line citing a paper, or after
    'Title:') that no tool result contains. A title written as a link is compared by its
    text, not its address."""
    evidence = _plain(" ".join(evidence_texts))
    text = answer or ""
    out = []
    quoted = [m for m in _QUOTED_TITLE.finditer(text)
              if _CITATION.search(text[text.rfind("\n", 0, m.start()) + 1:
                                       (text.find("\n", m.end()) + 1 or len(text) + 1) - 1])]
    for m in quoted + list(_TITLE_FIELD.finditer(text)):
        title = m.group(1).strip()
        link = re.match(r"\[([^\]]+)\]\(", title)       # the address may hold brackets: (01)
        words = _plain(link.group(1) if link else _LINK.sub(r"\1", title)).split()
        if len(words) < 4:
            continue
        if " ".join(words[:8]) not in evidence and title not in out:
            out.append(title)
    return out


def failed_tools(steps: list) -> list[str]:
    """Analysis tools that returned an error in this run and never succeeded afterwards."""
    import json as _json
    failed, ok = {}, set()
    for s in steps or []:
        if s.get("kind") != "delegation":
            continue
        for a in (s.get("attempts") or []) + [{"tool": s.get("tool"), "output": s.get("output")}]:
            tool, out = a.get("tool"), a.get("output") or ""
            if not tool or not out:
                continue
            try:
                err = (_json.loads(out) or {}).get("error") if out.lstrip().startswith("{") else None
            except (TypeError, ValueError, AttributeError):
                err = None
            if err:
                failed.setdefault(tool, str(err)[:160])
            else:
                ok.add(tool)
    return [f"{t} ({e})" for t, e in failed.items() if t not in ok]


# Template text standing in for a result ("[Paper 1 Title]", "Author A", "Journal Name"): the
# answer was written without the tool result in view.
_PLACEHOLDER = re.compile(
    # a bracket that is the text of a real link ("[Journal of Cell Biology](https://...)") is not one
    r"\[[^\]\n]{0,40}\b(?:title|authors?|doi|link|journal|name|value|year|url)\b[^\]\n]{0,20}\]"
    r"(?!\(\s*(?:https?://|www\.|doi\.org/))"
    r"|\bAuthor [A-Z]\b|\bJournal Name\b|\bTitle of (?:the )?paper\b"
    # results said to exist but left out
    r"|\b(?:were|was|been|are)\s+omitted\b|\bomitted for brevity\b"
    r"|\bnot (?:directly )?(?:included|summari[sz]ed|provided|shown) in the (?:response|results?|summary)\b"
    r"|\bdetails (?:were|have been) (?:successfully )?retrieved\b", re.I)


def placeholders(answer: str) -> list[str]:
    return sorted({m.group(0) for m in _PLACEHOLDER.finditer(answer or "")})
