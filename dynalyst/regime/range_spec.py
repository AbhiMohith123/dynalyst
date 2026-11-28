"""One range-resolution layer for every LHS entry point.

Whatever the request says -- a percentage, a fold, an absolute offset, exact
bounds, a changed baseline, fixed / vary-only parameters -- it is turned here
into the final per-parameter bounds {param: (lo, hi, is_log)}. The single LHS
engine (lhs_sampling.lhs_sample_params) then samples those bounds and never
needs to know how the request was phrased.

Per parameter, first match wins:

  1. excluded            left out (the continuation parameter)
  2. fixed               held at its fixed value, else at its (changed) baseline
  3. not in vary_only    held at its (changed) baseline -- unless it was given
                         explicit bounds or a window, which means "vary this"
  4. exact bounds        manual / {"abs": [lo, hi]}, clipped to `allowed`
  5. per-parameter window  {"rel": x} | {"fold": f} | {"offset": d}
  6. global window       rel > offset > fold
  7. default             a fold of `default_fold` around the baseline

Windows are centred on the baseline (`baseline` overrides the model's own
values), clipped to `allowed` where given, and never change the sign of a
parameter. `rel` is a fraction of the allowed range's width when one is known,
else of the baseline value; `fold` f gives [v/f, v*f] sampled log-uniformly;
`offset` d gives [v, v+d]. Contradictory requests are reported in
Resolution.conflicts, never resolved silently; unknown parameter names raise.
"""
import json
import re
from dataclasses import dataclass, field

LOG_SPAN_THRESHOLD = 10.0     # hi/lo above which exact / offset bounds are sampled log-uniformly
DEFAULT_FOLD = 2.0


@dataclass
class RangeExtras:
    """What the tools add to the classic range arguments."""
    baseline: dict = field(default_factory=dict)   # {param: changed baseline value}
    fixed: dict = field(default_factory=dict)      # {param: value | None (= its baseline)}
    windows: dict = field(default_factory=dict)    # {param: {"rel"|"fold"|"offset"|"abs": ...}}
    default_fold: float | None = None              # fold to use when nothing else is given


@dataclass
class RangeSpec:
    rel: float | None = None
    offset: float | None = None
    fold: float | None = None
    default_fold: float = DEFAULT_FOLD
    vary_only: list | None = None
    fixed: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    allowed: dict = field(default_factory=dict)
    rel_reference: dict = field(default_factory=dict)   # allowed range to use for `rel` only
    windows: dict = field(default_factory=dict)
    manual: dict = field(default_factory=dict)          # {param: (lo, hi) or (lo, hi, is_log)}
    exclude: set = field(default_factory=set)


@dataclass
class Resolution:
    bounds: dict                 # {param: (lo, hi, is_log)}
    baseline: dict               # value each parameter is centred on / held at
    notes: dict                  # {param: how its bounds were decided}
    conflicts: list
    strategy: tuple              # ('rel', x) | ('offset', d) | ('fold', f) | ('explicit', None)

    def varied(self):
        return [p for p, (lo, hi, _) in self.bounds.items() if lo != hi]

    def describe(self):
        lines = [f"Bounds (lo, hi) per parameter; {len(self.varied())} varied, "
                 f"{len(self.bounds) - len(self.varied())} held:"]
        for p, (lo, hi, is_log) in self.bounds.items():
            if lo == hi:
                lines.append(f"  {p}: held at {lo:.6g} ({self.notes.get(p, '')})")
            else:
                lines.append(f"  {p}: [{lo:.6g}, {hi:.6g}]{' log' if is_log else ''}"
                             f" ({self.notes.get(p, '')})")
        if self.conflicts:
            lines.append("Conflicts in the request (resolved as stated):")
            lines += [f"  - {c}" for c in self.conflicts]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# window arithmetic
# ---------------------------------------------------------------------------

def _clip(lo, hi, allowed):
    if allowed is None:
        return lo, hi
    a_lo, a_hi = float(allowed[0]), float(allowed[1])
    lo, hi = max(lo, a_lo), min(hi, a_hi)
    if lo > hi:                       # window entirely outside the allowed range
        lo = hi = min(max(lo, a_lo), a_hi)
    return lo, hi


def rel_window_bounds(base, rel, allowed=None):
    """(lo, hi, False): +-rel around base; rel is a fraction of the allowed
    range's width when `allowed` is given, else of |base|. Clipped to allowed."""
    half = rel * ((float(allowed[1]) - float(allowed[0])) if allowed is not None else abs(base))
    lo, hi = _clip(base - half, base + half, allowed)
    return float(lo), float(hi), False


def _is_log(lo, hi):
    return lo > 0 and (hi / lo) >= LOG_SPAN_THRESHOLD


def _window(v, kind, amount, allowed):
    """(lo, hi, is_log, text) for one windowed parameter with baseline v."""
    if kind == 'rel':
        lo, hi, log = rel_window_bounds(v, amount, allowed)
        of = "of its allowed range" if allowed is not None else "of the baseline"
        return lo, hi, log, f"+-{amount * 100:g}% {of}"
    if kind == 'fold':
        lo, hi = _clip(v / amount, v * amount, allowed)
        return lo, hi, True, f"fold {amount:g} around {v:.6g}"
    if kind == 'offset':
        lo, hi = _clip(v, v + amount, allowed)
        return lo, hi, _is_log(lo, hi), f"[baseline, baseline +{amount:g}]"
    raise ValueError(f"unknown window kind {kind!r}; use rel, fold, offset or abs")


def _normalise_manual(spec):
    if len(spec) == 3:
        lo, hi, is_log = spec
    else:
        lo, hi = spec
        is_log = _is_log(float(lo), float(hi))
    lo, hi = float(lo), float(hi)
    if lo > hi:
        raise ValueError(f"bounds [{lo}, {hi}] have lo > hi")
    return lo, hi, bool(is_log)


# ---------------------------------------------------------------------------
# the resolver
# ---------------------------------------------------------------------------

def _check_names(names, known, what):
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"{what} names {unknown}, which are not parameters here; "
                         f"the parameters are {sorted(known)}")


def resolve_bounds(base_values: dict, spec: RangeSpec) -> Resolution:
    """Turn a RangeSpec into final bounds for the parameters in base_values."""
    known = set(base_values)
    for names, what in ((spec.fixed, "fixed"), (spec.baseline, "baseline_values"),
                        (spec.windows, "param_windows"), (spec.manual, "manual_bounds"),
                        (spec.allowed, "allowed_bounds"), (spec.vary_only or [], "vary_only")):
        _check_names(names, known, what)

    vary = set(spec.vary_only) if spec.vary_only is not None else None
    manual = dict(spec.manual)      # the spec itself is left untouched
    if spec.rel is not None:
        strategy = ('rel', spec.rel)
    elif spec.offset is not None:
        strategy = ('offset', spec.offset)
    elif spec.fold is not None:
        strategy = ('fold', spec.fold)
    else:
        strategy = ('fold', spec.default_fold)
    if manual and not (spec.rel or spec.offset or spec.fold) and \
            set(manual) >= (vary or set(base_values) - spec.exclude):
        strategy = ('explicit', None)

    bounds, baseline, notes, conflicts = {}, {}, {}, []
    for p, v0 in base_values.items():
        if p in spec.exclude:
            if p in spec.windows or p in manual or p in spec.fixed:
                conflicts.append(f"{p} is the continuation parameter and is not sampled; "
                                 "its range request was ignored")
            continue
        v = float(spec.baseline.get(p, v0))
        changed = p in spec.baseline
        tag = " (changed baseline)" if changed else ""

        def hold(value, why):
            bounds[p], baseline[p], notes[p] = (value, value, False), value, why

        if p in spec.fixed:
            value = v if spec.fixed[p] is None else float(spec.fixed[p])
            if p in spec.windows or p in manual or (vary is not None and p in vary):
                conflicts.append(f"{p} is both fixed and given a range; fixed wins")
            hold(value, "fixed")
            continue
        explicit = p in manual or p in spec.windows
        if vary is not None and p not in vary and not explicit:
            hold(v, "held at baseline" + tag)
            continue
        if vary is not None and p not in vary and explicit:
            conflicts.append(f"{p} has an explicit range but is not in vary_only; "
                             "the explicit range makes it vary")
        allowed = spec.allowed.get(p)
        baseline[p] = v

        window = dict(spec.windows.get(p, {}))
        if 'abs' in window:
            spec_abs = tuple(window.pop('abs'))
            if p in manual:
                conflicts.append(f"{p} has both manual_bounds and an abs window; manual_bounds wins")
            else:
                manual[p] = spec_abs
        if p in manual:
            lo, hi, log = _normalise_manual(manual[p])
            if allowed is not None:
                c_lo, c_hi = _clip(lo, hi, allowed)
                if (c_lo, c_hi) != (lo, hi):
                    conflicts.append(f"{p}: requested [{lo:g}, {hi:g}] lies outside its allowed range "
                                     f"[{allowed[0]:g}, {allowed[1]:g}]; clipped to [{c_lo:g}, {c_hi:g}]")
                lo, hi = c_lo, c_hi
            bounds[p], notes[p] = (lo, hi, log), "exact bounds"
            continue
        if v <= 0 and not (window and 'rel' in window):
            hold(v, "non-positive baseline, held")
            continue
        if window:
            kind, amount = next(iter(window.items()))
            if len(window) > 1:
                conflicts.append(f"{p} has several windows {sorted(window)}; using {kind}")
        elif spec.rel is not None:
            kind, amount = 'rel', spec.rel
        elif spec.offset is not None:
            kind, amount = 'offset', spec.offset
        elif spec.fold is not None:
            kind, amount = 'fold', spec.fold
        else:
            kind, amount = 'fold', spec.default_fold
        use_allowed = allowed if allowed is not None else (
            spec.rel_reference.get(p) if kind == 'rel' else None)
        zero_span = (allowed is None and use_allowed is not None
                     and float(use_allowed[1]) == float(use_allowed[0]))
        if zero_span:
            # a reference range of zero width: "+-p% of its allowed range" would pin
            # the parameter, although the user asked for it to vary
            use_allowed = None
        lo, hi, log, text = _window(v, kind, float(amount), use_allowed)
        if zero_span:
            text += " (its reference range is a single value, so taken of the baseline)"
        if lo == hi:
            text = "no room inside its allowed range"
        bounds[p], notes[p] = (float(lo), float(hi), bool(log)), text + tag
    if vary is not None and not [p for p in bounds if bounds[p][0] != bounds[p][1]]:
        conflicts.append("no parameter is left to vary: every one is fixed, held or has no room")
    return Resolution(bounds, baseline, notes, conflicts, strategy)


# ---------------------------------------------------------------------------
# building a RangeSpec from the classic arguments + extras, and from tool JSON
# ---------------------------------------------------------------------------

def spec_from_args(*, spread_factor=None, additive_offset=None, rel_window=None,
                   manual_bounds=None, allowed_bounds=None, vary_only=None,
                   exclude_params=None, extras: RangeExtras | None = None,
                   rel_reference=None) -> RangeSpec:
    """The classic LHS arguments (plus RangeExtras) as one RangeSpec. A
    spread_factor / additive_offset of None means "not specified"."""
    extras = extras or RangeExtras()
    if isinstance(vary_only, str):
        vary_only = [vary_only]
    windows = {p: dict(w) for p, w in extras.windows.items()}
    return RangeSpec(
        rel=rel_window, offset=additive_offset, fold=spread_factor,
        default_fold=extras.default_fold or DEFAULT_FOLD,
        vary_only=list(vary_only) if vary_only is not None else None,
        fixed=dict(extras.fixed), baseline=dict(extras.baseline),
        allowed={p: tuple(v[:2]) for p, v in (allowed_bounds or {}).items()},
        rel_reference=dict(rel_reference or {}),
        windows=windows, manual=dict(manual_bounds or {}), exclude=set(exclude_params or ()))


def _json(text, what):
    if text is None or text == "":
        return None
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{what} must be valid JSON, got {text!r} ({e})")


def _lenient(text, what, names_ok):
    """JSON, or the plain forms a person (or a model) writes: "km1, km3" (names,
    if names_ok) and "p2=12, p1: 0.2" (name = value)."""
    if text is None or text == "" or not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    parts = [x.strip() for x in re.split(r"[,;]", text.strip().strip("[]{}")) if x.strip()]
    pairs = [re.split(r"\s*[=:]\s*", x, maxsplit=1) for x in parts]
    try:
        if parts and all(len(pr) == 2 for pr in pairs):
            return {k.strip("'\" "): float(v) for k, v in pairs}
        if names_ok and parts and all(len(pr) == 1 for pr in pairs):
            return [x.strip("'\" ") for x in parts]
    except ValueError:
        pass
    raise ValueError(f"{what} must be valid JSON (or like 'p2=12, p1=0.2'"
                     + (" or 'km1, km3'" if names_ok else "") + f"), got {text!r}")


_WINDOW_KEYS = {'rel', 'fold', 'offset', 'abs'}
_KEY_ALIASES = {'relative': 'rel', 'percent': 'rel', 'pct': 'rel', 'fraction': 'rel',
                'factor': 'fold', 'fold_change': 'fold', 'times': 'fold', 'spread': 'fold',
                'add': 'offset', 'plus': 'offset', 'range': 'abs', 'bounds': 'abs',
                'absolute': 'abs', 'between': 'abs'}
_NUM = r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
_TEXT_FOLD = re.compile(rf"^(?:x|×|\*|fold|factor(?: of)?)\s*{_NUM}$|^{_NUM}\s*(?:x|×|-?fold|times)$")
_TEXT_REL = re.compile(rf"^(?:\+-|±|\+/-)?\s*{_NUM}\s*(%|percent|pct)$")
_TEXT_ABS = re.compile(rf"^\[?\s*{_NUM}\s*(?:,|to|\.\.|-)\s*{_NUM}\s*\]?$")


def window_spec(name, w) -> dict:
    """One parameter's window in whatever form it was written, as
    {"rel"|"fold"|"offset"|"abs": value}. Accepts the dict form, aliases of its
    keys ({"percent": 10} is +-10%), [lo, hi], and text: "x3", "3-fold",
    "factor of 3", "+-10%", "10 percent", "10 to 30"."""
    def bad():
        return ValueError(f"param_windows[{name!r}] = {w!r} is not a window; use e.g. "
                          '{"fold": 3}, {"rel": 0.1}, {"offset": 5}, {"abs": [10, 30]}, '
                          '"x3", "+-10%" or [10, 30]')
    if isinstance(w, (list, tuple)) and len(w) == 2:
        return {"abs": [float(w[0]), float(w[1])]}
    if isinstance(w, str):
        t = w.strip().lower()
        m = _TEXT_FOLD.match(t)
        if m:
            return {"fold": float(m.group(1) or m.group(2))}
        m = _TEXT_REL.match(t)
        if m:
            return {"rel": float(m.group(1)) / 100}
        m = _TEXT_ABS.match(t)
        if m:
            return {"abs": [float(m.group(1)), float(m.group(2))]}
        raise bad()
    if not isinstance(w, dict) or not w:
        raise bad()
    out = {}
    for k, v in w.items():
        key = _KEY_ALIASES.get(str(k).lower(), str(k).lower())
        if key not in _WINDOW_KEYS:
            raise ValueError(f"param_windows[{name!r}] has unknown keys [{k!r}]; use rel, fold, "
                             "offset or abs")
        if str(k).lower() in ('percent', 'pct'):
            v = float(v) / 100
        out[key] = v
    return out


def extras_from_tool_args(baseline_values=None, fixed_params=None, param_windows=None,
                          default_fold=None) -> RangeExtras:
    """RangeExtras from the tools' JSON-string arguments:
      baseline_values  '{"p2": 12.5}'                      changed baselines
      fixed_params     '{"p1": 0.2}' or '["p1", "p2"]'   held (value, else baseline)
      param_windows    '{"p2": {"rel": 0.1}, "p1": {"fold": 1.5},
                         "p3": {"offset": 5}, "p4": {"abs": [10, 30]}}'"""
    baseline = _lenient(baseline_values, "baseline_values", names_ok=False) or {}
    fixed = _lenient(fixed_params, "fixed_params", names_ok=True) or {}
    if isinstance(fixed, list):
        fixed = {p: None for p in fixed}
    windows = {p: window_spec(p, w) for p, w in (_json(param_windows, "param_windows") or {}).items()}
    return RangeExtras(baseline={p: float(v) for p, v in baseline.items()},
                       fixed={p: (None if v is None else float(v)) for p, v in fixed.items()},
                       windows=windows, default_fold=default_fold)
