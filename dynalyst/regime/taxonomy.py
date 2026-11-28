"""What an LHS sample turned out to show.

Each sample gets its FEATURES (everything it shows) and one PRIMARY category
(the highest in PRIORITY), so the single overlay plot can colour by one label
while the per-category view and the CSV keep every feature.

  bistable         two stable equilibria demonstrated at the same parameter value
  branch_point     at least one branch point (BP). AUTO reports transcritical and
                   pitchfork alike as BP and this scheme does not tell them apart
  hopf             at least one Hopf point (HB)
  single_fold      exactly one fold (LP)
  multiple_folds   multiple folds without demonstrated stable coexistence
  ultrasensitive   none of the above, and a steep monotonic response
  graded           none of the above, and a shallow monotonic response
  incomplete       none of the above, but the continuation did not cover the range: no verdict
  no_bifurcation   none of the above and the response could not be rated
  failed           AUTO did not produce a diagram for this sample

ultrasensitive / graded use the Goldbeter-Koshland response ratio (x at 90% of
the branch's y-range over x at 10%; < 81 is steeper than a plain
Michaelis-Menten response). Only meaningful for a monotonic branch. The
finite continuation window makes it read
low (see compute_response_ratio), so treat that split as a ranking.
"""
import numpy as np

ULTRASENSITIVE_THRESHOLD = 81.0  # Goldbeter-Koshland: ratio=81 for a plain Michaelis-Menten response

PRIORITY = ['bistable', 'branch_point', 'hopf', 'single_fold', 'multiple_folds',
            'ultrasensitive', 'graded', 'incomplete', 'no_bifurcation']
FAILED = 'failed'
CATEGORY_ORDER = PRIORITY + [FAILED]


def compute_response_ratio(branch0, bif_param, y_species):
    """Goldbeter-Koshland-style response ratio for a monotonic branch:
    the x-value where y crosses 90% of its (ymin, ymax) range, divided by
    the x-value where it crosses 10%. Returns None (rather than raising)
    if the branch is too short, flat, or otherwise degenerate -- a failed
    ratio just means "can't classify this one", not a pipeline bug.

    CAVEAT: ymin/ymax here are the values AUTO's continuation actually
    reaches within param_range, not the curve's true (infinite-domain)
    asymptote. If a branch hasn't fully saturated by the end of the
    scanned range, this systematically UNDER-estimates the ratio -- a
    genuinely graded (Michaelis-Menten-like) response can come out looking
    steeper than it really is. Trust the ratio as a relative ranking within
    one batch (lower = steeper) more than as an absolute reproduction of the
    textbook value 81; widening param_range is the fix for unsaturated
    branches.
    """
    x = np.asarray(branch0[bif_param], dtype=float)
    y = np.asarray(branch0[y_species], dtype=float)
    if len(x) < 3:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]
    ymin, ymax = float(y.min()), float(y.max())
    if ymax - ymin < 1e-12:
        return None  # flat branch, no response to speak of

    y10 = ymin + 0.1 * (ymax - ymin)
    y90 = ymin + 0.9 * (ymax - ymin)

    # np.interp needs its "xp" (here: y) array increasing; the branch may be
    # monotonically increasing OR decreasing in y as x increases, so sort on y
    y_order = np.argsort(y)
    y_sorted, x_at_y = y[y_order], x[y_order]
    try:
        x10 = float(np.interp(y10, y_sorted, x_at_y))
        x90 = float(np.interp(y90, y_sorted, x_at_y))
    except Exception:
        return None

    if x10 == 0:
        return None
    return abs(x90 / x10)


def point_counts(special):
    """{'LP': n, 'BP': n, 'HB': n} from a sample's (index, type, branch) special points."""
    counts = {'LP': 0, 'BP': 0, 'HB': 0}
    for _, ty, _ in special or []:
        if ty in counts:
            counts[ty] += 1
    return counts


def _stable_counts(curves, n_grid=200, within=None, state_tol=None):
    """(grid, number of distinct stable steady states at each grid value). curves are
    (xs, ys, signs) per branch, signs > 0 = stable (AUTO's own stability). States closer than
    1e-6 relative are one state (a branch continued forward and backward meets itself at its
    start). within: only this parameter range counts. state_tol: ys is a (points x states)
    array per branch, and two states are one when every variable differs by at most state_tol
    of its scale (the same branch followed twice is interpolated at different points, so it
    never matches to 1e-6; comparing one plotted variable alone also merges distinct states
    that share its value)."""
    xs_all = np.concatenate([np.asarray(c[0], float) for c in curves if len(c[0])])         if curves else np.array([])
    if xs_all.size < 2:
        return np.array([]), []
    low, high = xs_all.min(), xs_all.max()
    if within is not None:
        low, high = max(low, within[0]), min(high, within[1])
        if low >= high:
            return np.array([]), []
    grid = np.linspace(low, high, n_grid)[1:-1]
    found = [[] for _ in grid]       # stable states at each grid value, from every branch
    for cx, cy, sg in curves:
        cx, cy, sg = np.asarray(cx, float), np.asarray(cy, float), np.asarray(sg)
        if cx.size < 2:
            continue
        seg = (sg[:-1] > 0) & (sg[1:] > 0) & (cx[:-1] != cx[1:])      # stable segments
        x0, x1, y0, y1 = cx[:-1][seg], cx[1:][seg], cy[:-1][seg], cy[1:][seg]
        lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
        for g, x in enumerate(grid):         # vectorised over the segments
            inside = (lo <= x) & (x <= hi)
            if inside.any():
                w = ((x - x0[inside]) / (x1[inside] - x0[inside]))
                if state_tol is not None:
                    w = w[:, None]
                found[g].extend(list(y0[inside] + w * (y1[inside] - y0[inside])))
    if state_tol is not None:
        allv = [np.atleast_1d(v) for ys in found for v in ys]
        scale = np.max(np.abs(np.vstack(allv)), axis=0) + 1e-300 if allv else 1.0
    counts = []
    for ys in found:
        distinct = []
        for y in ys:
            if state_tol is None:
                same = any(abs(y - d) <= 1e-6 * max(1.0, abs(y)) for d in distinct)
            else:
                same = any(np.all(np.abs(y - d) <= state_tol * scale) for d in distinct)
            if not same:
                distinct.append(y)
        counts.append(len(distinct))
    return grid, counts


def max_coexisting_stable(curves, n_grid=200, within=None, state_tol=None):
    """The most stable steady states that coexist at one parameter value, and where (see
    _stable_counts). within: a retry can continue well past the range that was asked for."""
    grid, counts = _stable_counts(curves, n_grid, within, state_tol)
    if not counts:
        return 0, None
    g = int(np.argmax(counts))
    return counts[g], float(grid[g])


def coexistence_windows(curves, n_grid=400, within=None, state_tol=None):
    """The parameter windows where two or more stable steady states coexist: a list of
    {"from", "to", "stable_states"} (the most coexisting in the window), to within one grid
    step of (within or the branches' span) / n_grid."""
    grid, counts = _stable_counts(curves, n_grid, within, state_tol)
    windows, current = [], None
    for x, k in zip(grid, counts):
        if k >= 2:
            if current is None:
                current = {"from": float(x), "to": float(x), "stable_states": k}
            current["to"] = float(x)
            current["stable_states"] = max(current["stable_states"], k)
        elif current is not None:
            windows.append(current)
            current = None
    if current is not None:
        windows.append(current)
    # a single grid value is noise (two interpolated copies of one state), not a window
    return [w for w in windows if w["to"] > w["from"]]


def classify_diagram(types, curves=None, bif_param=None, y_species=None,
                     ultrasensitive_threshold=ULTRASENSITIVE_THRESHOLD, covered=True, within=None,
                     state_curves=None, state_tol=1e-3):
    """One labeller for any AUTO result (a single diagram or an LHS sample).
    types: the special-point types found (LP / BP / HB ...); curves: optional
    (xs, ys, signs) per branch. With curves, 'bistable' means two stable steady
    states really coexist at one parameter value (so two folds that do not
    enclose a bistable window are not called bistable, and a pitchfork's two
    stable arms are). Without stability evidence, folds alone do not establish bistability.
    covered=False: the continuation stopped short of the range, so finding nothing proves nothing.
    Returns {'category', 'features', 'evidence'}."""
    n = {'LP': 0, 'BP': 0, 'HB': 0}
    for t in types:
        if t in n:
            n[t] += 1
    # state_curves (all state variables per point): coexistence judged on whole states
    coexist, at = (max_coexisting_stable(state_curves, within=within, state_tol=state_tol)
                   if state_curves else
                   max_coexisting_stable(curves, within=within) if curves else (None, None))
    evidence = {'counts': n}
    feats = []
    if coexist is not None:
        evidence.update(max_coexisting_stable=coexist, at=at)
        if coexist >= 2:
            feats.append('bistable')
        elif n['LP'] >= 2:
            evidence['note'] = ('two folds were detected; stable coexistence was not '
                                'demonstrated on the sampled branches')
    else:
        evidence['note'] = 'stability/coexistence evidence unavailable'
    if n['BP'] >= 1:
        feats.append('branch_point')
    if n['HB'] >= 1:
        feats.append('hopf')
    if n['LP'] == 1:
        feats.append('single_fold')
    elif n['LP'] >= 2 and 'bistable' not in feats:
        feats.append('multiple_folds')
    if not feats and not covered:
        evidence['note'] = 'the continuation did not reach both ends of the range: no verdict on it'
        feats.append('incomplete')
    if not feats:
        ratio = None
        if curves and bif_param and y_species:
            try:
                ratio = compute_response_ratio({bif_param: curves[0][0], y_species: curves[0][1]},
                                               bif_param, y_species)
            except Exception:
                ratio = None
        evidence['response_ratio'] = ratio
        feats.append('no_bifurcation' if ratio is None else
                     'ultrasensitive' if ratio < ultrasensitive_threshold else 'graded')
    feats.sort(key=PRIORITY.index)
    return {'category': feats[0], 'features': feats, 'evidence': evidence}


def analyse(entry, bif_param, y_species, ultrasensitive_threshold=ULTRASENSITIVE_THRESHOLD):
    """Classify one batch_lhs_runner result entry. Stores the outcome on the
    entry ('category', 'features', and '_response_ratio' for monotonic
    samples) and returns {'category', 'features', 'evidence'} for successful samples."""
    if entry.get('error') is not None or entry.get('special') is None:
        entry['category'], entry['features'] = FAILED, [FAILED]
        return {'category': FAILED, 'features': [FAILED]}

    curves = []
    for b in entry.get('all_branches') or []:
        try:
            x, y = np.asarray(b[bif_param]), np.asarray(b[y_species])
            signs = np.zeros(len(x), dtype=int)
            start = 0
            for end in b.stability() if hasattr(b, 'stability') else []:
                stop = min(abs(int(end)), len(x))
                signs[start:stop] = 1 if end < 0 else -1
                start = stop
            curves.append((x, y, signs))
        except (KeyError, TypeError, AttributeError):
            continue
    branches = entry.get('all_branches') or []
    covered = all(getattr(b, 'range_covered', True) for b in branches)
    within = next((b.requested_range for b in branches if getattr(b, 'requested_range', None)), None)
    result = classify_diagram([p[1] for p in entry['special']], curves or None,
                              bif_param, y_species, ultrasensitive_threshold, covered=covered, within=within)
    entry.update(category=result['category'], features=result['features'],
                 classification=result, _response_ratio=result['evidence'].get('response_ratio'))
    return result
