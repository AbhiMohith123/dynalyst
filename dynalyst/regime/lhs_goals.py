"""Goal-driven LHS: "find bistability", "find a branch point", ...

A goal does three things, and nothing else:

  * picks the DEFAULT sampling window when the user gave none (a fold around
    the baseline: tight for structural features, wider for robust ones).
    Anything the user specified -- percentages, bounds, baselines, fixed or
    vary-only parameters -- is honoured as given;
  * adds a CONTROL run of the unperturbed baseline, so "not found" can be told
    apart: the baseline has it but the samples lost it (window too wide), or
    the baseline never had it (window too narrow, or the model lacks it);
  * writes the GOAL REPORT: found / not found per goal, what the samples did
    show instead, the evidence for any failure, reasons clearly marked as
    unverified, and concrete ways to widen or narrow the search.

It never re-runs anything by itself: the user decides whether to widen or narrow.
"""
import re
from collections import Counter
from dataclasses import dataclass, replace

from dynalyst.regime.range_spec import RangeExtras
from dynalyst.regime.taxonomy import CATEGORY_ORDER, FAILED, ULTRASENSITIVE_THRESHOLD, analyse


@dataclass(frozen=True)
class Goal:
    name: str
    feature: str            # taxonomy feature that counts as success
    label: str
    aliases: tuple
    default_fold: float     # default window: [baseline/f, baseline*f], when the user gave none
    caveat: str


GOALS = {g.name: g for g in [
    Goal('bistability', 'bistable', 'bistability (coexisting stable equilibria)',
         ('bistability', 'bistable', 'multistability', 'multistable', 'hysteresis', 'two folds'),
         1.5,
         "Bistability sits in a narrow part of parameter space (Michaelis constants small next "
         "to the totals): a wide window mostly leaves it."),
    Goal('branch_point', 'branch_point', 'a branch point (BP)',
         ('branch point', 'bp', 'pitchfork', 'pf', 'transcritical', 'tc', 'branching'),
         1.25,
         "A branch point needs exact relations between parameters (a symmetry or a structure), "
         "so random sampling usually unfolds it into folds or removes it. It is expected only "
         "where the baseline has one. AUTO reports transcritical and pitchfork alike as BP and "
         "they are not told apart here."),
    Goal('ultrasensitivity', 'ultrasensitive', 'ultrasensitivity (steep monotonic response)',
         ('ultrasensitivity', 'ultrasensitive', 'ultra sensitivity', 'ultra sensitive', 'switch like',
          'switch'),
         2.0,
         "The ultrasensitive / graded split uses a response ratio computed over a finite window, "
         "which reads low on unsaturated branches: treat it as a ranking."),
    Goal('hopf', 'hopf', 'a Hopf point (HB, onset of oscillations)',
         ('hopf', 'hopf bifurcation', 'hopf point', 'hb', 'oscillation', 'oscillations',
          'oscillatory', 'limit cycle', 'limit cycles', 'sustained oscillations'),
         1.5,
         "A Hopf point needs at least two independent state variables, and continuing a steady "
         "state finds only where oscillations start (the HB point), not the oscillations "
         "themselves. It is robust to small parameter changes, so samples near a baseline that "
         "has one usually keep it."),
]}

MIN_STATES = {'hopf': 2}    # below this, the goal is impossible in the model


def _norm(text):
    return re.sub(r'[\s_\-]+', ' ', text.strip().lower())


def parse_goals(text):
    """Goal objects for 'bistability', 'branch point', 'bistability, ultrasensitive', ...
    (None / '' gives []). Unknown words raise, listing the goals."""
    if not text:
        return []
    goals = []
    for part in re.split(r'[,;/&]|\band\b', text):
        word = _norm(part)
        if not word:
            continue
        match = next((g for g in GOALS.values()
                      if word == _norm(g.name) or word in g.aliases or word == _norm(g.feature)), None)
        if match is None:
            raise ValueError(f"unknown goal {part.strip()!r}; goals are {sorted(GOALS)} "
                             "(aliases: " + "; ".join(f"{g.name}: {', '.join(g.aliases)}"
                                                      for g in GOALS.values()) + ")")
        if match not in goals:
            goals.append(match)
    return goals


def prepare(goal_text, extras=None):
    """(goals, extras with the goal's default fold filled in, control?). The
    default fold is the smallest of the goals': start near the baseline and
    let the user widen."""
    goals = parse_goals(goal_text)
    extras = extras or RangeExtras()
    if goals and extras.default_fold is None:
        extras = replace(extras, default_fold=min(g.default_fold for g in goals))
    return goals, extras, bool(goals)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

_ARG = {'fold': 'spread_factor', 'rel': 'rel_window', 'offset': 'additive_offset'}


def _step(strategy, direction):
    """(argument name, new value) that widens or narrows the current window, or None."""
    kind, amount = strategy
    if kind not in _ARG or not amount:
        return None
    if kind == 'fold':
        new = amount * 1.5 if direction == 'widen' else 1 + (amount - 1) / 2
    else:
        new = amount * 2 if direction == 'widen' else amount / 2
    return _ARG[kind], float(f"{new:.3g}")


def _first_error(results):
    errors = Counter((r['error'] or '').splitlines()[0][:140] for r in results if r.get('error'))
    return errors.most_common(1)[0] if errors else None


def _hit_ranges(hits):
    varied = [p for p in hits[0]['params'] if len({r['params'][p] for r in hits}) > 1]
    return {p: (min(r['params'][p] for r in hits), max(r['params'][p] for r in hits)) for p in varied}


def report(goals, results, bif_param, y_species, window_text=None,
           ultrasensitive_threshold=ULTRASENSITIVE_THRESHOLD):
    """The goal report (a text block), or '' when there are no goals. `results`
    is a batch_lhs_runner.BatchResults (it carries .control and .resolution)."""
    if not goals:
        return ""
    for r in results:
        analyse(r, bif_param, y_species, ultrasensitive_threshold)
    control = getattr(results, 'control', None)
    if control is not None:
        analyse(control, bif_param, y_species, ultrasensitive_threshold)
    resolution = getattr(results, 'resolution', None)
    strategy = getattr(resolution, 'strategy', None) or ('explicit', None)
    n = len(results)
    failed = [r for r in results if r['category'] == FAILED]
    shown = Counter(r['category'] for r in results)
    lines = ["Goal check:"]

    # baseline control
    if control is None:
        lines.append("- Baseline control: not run (this scan has no single baseline).")
    elif control['category'] == FAILED:
        lines.append(f"- Baseline control: the unperturbed model itself failed ({control['error']}).")
    else:
        lines.append(f"- Baseline control: the unperturbed model shows "
                     f"{', '.join(control['features'])} (signature {control['signature']}).")

    n_states = getattr(results, 'n_states', None)
    missing, found_any, impossible = [], [], []
    for g in goals:
        hits = [r for r in results if g.feature in (r.get('features') or [])]
        need = MIN_STATES.get(g.name, 0)
        if n_states is not None and n_states < need:
            impossible.append(g)
            lines.append(f"- IMPOSSIBLE {g.label}: the model is continued in {n_states} independent "
                         f"state variable(s) and this needs at least {need}; no sampling can find it.")
        elif hits:
            found_any.append((g, hits))
            ids = ", ".join(str(r['sample_id']) for r in hits[:12]) + (" ..." if len(hits) > 12 else "")
            lines.append(f"- FOUND {g.label}: {len(hits)}/{n} samples (ids {ids}).")
            ranges = _hit_ranges(hits)
            if ranges and len(hits) > 1:
                lines.append("  Where: " + "; ".join(f"{p} in [{lo:.4g}, {hi:.4g}]"
                                                      for p, (lo, hi) in ranges.items()))
        else:
            missing.append(g)
            lines.append(f"- NOT FOUND {g.label}: none of the {n} samples showed it.")

    if missing or impossible:
        got = ", ".join(f"{c}: {shown[c]}" for c in CATEGORY_ORDER if shown.get(c))
        lines.append(f"What the samples did show instead: {got or 'nothing (every sample failed)'}.")
    if failed:
        err = _first_error(results)
        lines.append(f"Evidence: {len(failed)}/{n} samples failed to produce a diagram"
                     + (f"; most common error: {err[0]} (x{err[1]})." if err else "."))
    if window_text:
        lines.append(f"Evidence: continuation window {window_text}.")

    # reasons (clearly unverified) and what to try
    ctrl_has = {g.name: control is not None and control['category'] != FAILED
                and g.feature in control['features'] for g in goals}
    if missing:
        lines.append("Possible reasons (not verified):")
        for g in missing:
            if control is not None and ctrl_has[g.name]:
                lines.append(f"  - {g.name}: the baseline has it but no sample kept it, so the sampled "
                             "range is probably too wide.")
            elif control is not None:
                lines.append(f"  - {g.name}: the baseline does not show it, so sampling close to the "
                             "baseline is unlikely to; the window may miss it or the model may lack it.")
            lines.append(f"  - {g.name}: {g.caveat}")
        if len(failed) > n / 2:
            lines.append("  - most samples failed, which points at the window, the continuation "
                         "parameter or samples too far from a working baseline.")
    if impossible and not missing and not found_any:   # widening or narrowing cannot help
        lines.append("Suggestions:")
        lines.append(f"  1. Use a model with at least {max(MIN_STATES[g.name] for g in impossible)} "
                     "independent state variables for this.")
        return "\n".join(lines)
    suggestions = []
    lead_narrow = bool(failed and len(failed) > n / 2) or any(ctrl_has.get(g.name) for g in missing)
    order = ('narrow', 'widen') if (lead_narrow or not missing) else ('widen', 'narrow')
    for direction in order:
        step = _step(strategy, direction)
        if step:
            what = "closer to the baseline" if direction == 'narrow' else "further from the baseline"
            suggestions.append(f"{direction.capitalize()} the search ({what}): re-run with "
                               f"{step[0]}={step[1]:g}.")
    if strategy[0] == 'explicit' or not suggestions:
        suggestions.append("The ranges were given explicitly and were honoured as given; to widen or "
                           "narrow the search, change those bounds.")
    if missing:
        suggestions.append("Or change the continuation parameter or its window (param_min / param_max): "
                           "a window that misses the feature looks the same as no feature.")
    elif found_any and any(len(h) < n / 2 for _, h in found_any):
        suggestions.append("To concentrate on what was found, restrict the varied parameters to the "
                           "ranges listed under 'Where' (param_windows / manual_bounds).")
    lines.append("Suggestions:")
    lines += [f"  {i}. {s}" for i, s in enumerate(suggestions, 1)]
    return "\n".join(lines)
