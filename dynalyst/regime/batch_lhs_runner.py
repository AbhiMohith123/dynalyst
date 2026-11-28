"""
Runs the AUTO continuation of analysis/bifurcation_auto.py once per
Latin-Hypercube-sampled parameter set drawn from lhs_sampling.py, then
overlays every resulting bifurcation diagram on one shared axis so the
range of qualitatively different bifurcation structures produced across the
sampled parameter space is visible at a glance.

Unlike run_bifurcation_analysis, which runs ONE model and saves ONE plot, this module runs MANY parameter sets
and produces:
  1. one combined figure (all diagrams overlaid, colored by bifurcation
     "signature" -- i.e. how many LP/BP/HB points AUTO found)
  2. a CSV summary (one row per sample: id, parameter values, signature,
     any error)

bif_param and y_species are held FIXED across all samples so every diagram shares the
same axes and is actually
comparable -- letting each sample auto-select its own bifurcation
parameter, as the single-model tools do, would make overlay meaningless.
"""
import os
import csv
import uuid
import traceback
import json
import hashlib
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynalyst import config
from dynalyst.regime.lhs_sampling import resolve_model_bounds, lhs_sample_params
from dynalyst.regime import lhs_goals, taxonomy
from dynalyst.regime.bifurcation_markers import mark_special_points, special_point_legend_handles
from dynalyst.regime import model_source
from dynalyst.auto_env import AutoUnavailable



class BatchResults(list):
    """The per-sample result dicts of one batch (a plain list to every
    consumer) plus how the batch was set up:
      resolution  range_spec.Resolution: final bounds, baseline, how each was decided
      control     result entry of the unperturbed baseline run, or None
    """
    resolution = None
    control = None
    n_states = None     # independent state variables the model is continued in
    time_note = None    # time-dependent inputs frozen into constants, if any
    identity = None     # which model (and BioModels ID) this is
    window = None       # (continuation range, where it came from)


def summarize_windows(results):
    """One line saying which continuation window a batch used."""
    if not results.window:
        return "n/a (no sample completed)"
    (lo, hi), source = results.window
    return f"[{lo:g}, {hi:g}] ({source})"


class _Branch(dict):
    """One continued branch: arrays by name, and stability() as AUTO gives it: the end of each
    run of one stability, counted from 1, negative for a stable run."""

    def __init__(self, columns, stable):
        super().__init__(columns)
        self._ends = [-i if stable[i - 1] else i for i in range(1, len(stable) + 1)
                      if i == len(stable) or stable[i] != stable[i - 1]]

    def stability(self):
        return self._ends


def compute_bifurcation(param_values, bif_param, param_range, y_species, parsed_base,
                        log_probes=False):
    """One sample through the main AUTO runner, plot-free and without cycles. Returns
    (branches, special): special holds (point index, type, branch index), each point once
    (forward and backward runs share their start) and inside the range. parsed_base: the
    model's parsed form (model_source.prepare); param_values override its values."""
    from dynalyst.analysis.bifurcation_auto import _run_single_bifurcation_analysis
    out = json.loads(_run_single_bifurcation_analysis(
        model_name_or_path=parsed_base['source_path'], param_name=bif_param,
        param_min=param_range[0], param_max=param_range[1], plot_state=y_species,
        fixed_params={**param_values, bif_param: parsed_base['param_values'][bif_param]},
        # a sampled parameter set can split the branch point that joins a fold to the start's
        # branch: the states off that branch are continued from as well
        probe_only=True, show_auto_log=False, find_other_branches=True, log_probes=log_probes))
    if out.get('auto_unavailable'):
        raise AutoUnavailable(out['error'])
    if out.get('error'):
        raise RuntimeError(out['error'])
    if not out['branches']:
        raise RuntimeError(f"AUTO-07p produced no branch with {y_species}")
    branches, special, seen = [], [], []
    for number, b in enumerate(out['branches']):
        branches.append(_Branch({bif_param: np.array(b['parameter']), y_species: np.array(b['state'])},
                                b['stable']))
        branches[-1].range_covered = out.get('range_covered', True)
        branches[-1].requested_range = tuple(param_range)
        for pt in b['points']:
            at = b['parameter'][pt['index']]
            if (param_range[0] <= at <= param_range[1] and not any(
                    ty == pt['type'] and abs(at - x) <= 1e-6 * max(abs(x), 1e-12) for ty, x in seen)):
                seen.append((pt['type'], at))
                special.append((pt['index'], pt['type'], number))
    # the turn of a pitchfork's arms is labelled a fold at the branch point itself: one point
    at_point = lambda sp: (branches[sp[2]][bif_param][sp[0]], branches[sp[2]][y_species][sp[0]])
    span = max(abs(param_range[1] - param_range[0]), 1e-12)
    points = [at_point(sp) for sp in special if sp[1] == 'BP']
    special = [sp for sp in special if not (sp[1] == 'LP' and any(
        abs(at_point(sp)[0] - x) <= 1e-4 * span and abs(at_point(sp)[1] - y) <= 1e-3 * (1 + abs(y))
        for x, y in points))]
    return branches, special


def classify_signature(special):
    """Turns the (idx, type, branch) special-point list of a sample
    into a short human-readable label describing the bifurcation structure
    found, e.g. '2 LP', '1 BP', '1 BP + 1 HB', 'none'."""
    counts = {'LP': 0, 'BP': 0, 'HB': 0}
    for _, ty, _ in special:
        if ty in counts:
            counts[ty] += 1

    parts = []
    if counts['LP']:
        parts.append(f"{counts['LP']} LP")
    if counts['BP']:
        parts.append(f"{counts['BP']} BP")
    if counts['HB']:
        parts.append(f"{counts['HB']} HB")
    return ' + '.join(parts) if parts else 'none'


def _format_samples_preview(samples, bounds):
    """Human-readable preview of the drawn LHS samples and the bounds used
    to draw them, printed BEFORE the (slow) AUTO loop starts so you can see
    exactly what's about to run rather than waiting for the whole batch."""
    lines = ["Bounds used for this scan (lo, hi):"]
    for p, (lo, hi, is_log) in sorted(bounds.items()):
        tag = " [log-uniform]" if is_log else ""
        lines.append(f"  {p}: [{lo:.4g}, {hi:.4g}]{tag}")

    param_names = sorted(samples[0].keys()) if samples else []
    lines.append("")
    lines.append(f"Drew {len(samples)} LHS samples:")
    header = "  id  " + "  ".join(f"{p:>12}" for p in param_names)
    lines.append(header)
    for i, s in enumerate(samples):
        row = f"  {i:>2}  " + "  ".join(f"{s[p]:>12.4g}" for p in param_names)
        lines.append(row)
    lines.append("")
    lines.append(f"Now running the full AUTO pipeline on each of the {len(samples)} "
                 f"samples above -- this is the slow part, please wait...")
    return "\n".join(lines)


def _run_samples(samples, bif_param, param_range, out_dir, y_species, parsed_base, unavailable=None,
                 log_probes=False):
    """Shared loop: runs compute_bifurcation for each sampled parameter
    dict, catching per-sample failures (AUTO not converging etc.) rather
    than letting one bad sample abort the whole batch."""
    results = BatchResults()
    for i, params in enumerate(samples):
        workdir = os.path.join(out_dir, f'sample_{i:03d}')
        os.makedirs(workdir, exist_ok=True)
        entry = {'sample_id': i, 'params': params, 'all_branches': None,
                 'special': None, 'signature': None, 'error': None}
        try:
            if unavailable:
                raise AutoUnavailable(unavailable)
            all_branches, special = compute_bifurcation(params, bif_param, param_range, y_species,
                                                        parsed_base, log_probes)
            entry['all_branches'] = all_branches
            entry['special'] = special
            entry['signature'] = classify_signature(special)
        except AutoUnavailable as e:
            unavailable = str(e)
            entry['error'] = 'AUTO unavailable: ' + unavailable
            entry['auto_unavailable'] = True
        except Exception as e:
            entry['error'] = f"{type(e).__name__}: {e}"
            entry['_traceback'] = traceback.format_exc()
        results.append(entry)
        Path(workdir, 'result.json').write_text(json.dumps(
            {k:v for k,v in entry.items() if k != 'all_branches'}, indent=2, default=str), encoding='utf-8')
    return results


def _run_control(resolution, bif_param, param_range, out_dir, y_species, parsed_base, log_probes=False):
    """The unperturbed baseline through the same pipeline, as a result entry
    (sample_id 'control'). It says whether the model itself shows a feature,
    which is what tells "the sampling lost it" from "it was never there"."""
    entry = _run_samples([dict(resolution.baseline)], bif_param, param_range,
                         os.path.join(out_dir, 'control'), y_species, parsed_base,
                         log_probes=log_probes)[0]
    entry['sample_id'] = 'control'
    return entry


def _finish(results, resolution, control_entry, n_states):
    results.resolution, results.control, results.n_states = resolution, control_entry, n_states
    return results


def run_lhs_batch(n_samples, bif_param=None, y_species=None, param_range=None,
                   additive_offset=None, manual_bounds=None, seed=None, out_dir=None,
                   model=None, spread_factor=None, rel_window=None, allowed_bounds=None,
                   vary_only=None, extras=None, control=False, log_probes=False):
    """LHS batch over a model file (run_lhs_batch_for_model). Returns (results, out_dir);
    results is a BatchResults of dicts {sample_id, params, all_branches, special,
    signature, error} (all_branches/special/signature are None and error is set for a
    sample whose AUTO run failed -- expected for some parameter sets, not a bug)."""
    results, _, out_dir = run_lhs_batch_for_model(
        model_source.require_model(model), n_samples, bif_param=bif_param,
        y_species=y_species, param_range=param_range, spread_factor=spread_factor,
        additive_offset=additive_offset, rel_window=rel_window, allowed_bounds=allowed_bounds,
        manual_bounds=manual_bounds, vary_only=vary_only, seed=seed, out_dir=out_dir,
        extras=extras, control=control, log_probes=log_probes)
    return results, out_dir


def run_lhs_batch_for_model(model, n_samples, bif_param=None, y_species=None,
                            param_range=None, spread_factor=None, additive_offset=None,
                            rel_window=None, allowed_bounds=None, manual_bounds=None,
                            vary_only=None, seed=None, out_dir=None, extras=None,
                            control=False, log_probes=False):
    """LHS batch for ANY SBML / XPP model (the generic path of the tools).

    Every parameter the model's equations use, except the continuation
    parameter bif_param, is sampled around its own baseline (or the changed
    baseline in `extras`); range_spec.resolve_bounds decides each range from
    whatever was requested -- by default a fold of 2 (log-uniform baseline/2
    .. baseline*2). bif_param / y_species default to the model's conserved total
    (else the screened parameter) / last state variable; param_range defaults
    to a decade either side of bif_param's baseline.

    control=True also runs the unperturbed baseline (results.control).

    Returns (results, model label, out_dir)."""
    target = model_source.prepare(model, bif_param, y_species)
    bif_param = target['bif_param']
    parsed = target['parsed']
    changed = (extras.baseline if extras else {}).get(bif_param)
    if changed is not None:   # a changed baseline of the continuation parameter re-centres its window
        parsed = {**parsed, 'param_values': {**parsed['param_values'], bif_param: changed}}
        target = {**target, 'parsed': parsed}
    param_range, source = model_source.window_for(target, *(param_range or (None, None)))
    resolution = resolve_model_bounds(
        parsed, spread_factor=spread_factor, additive_offset=additive_offset,
        manual_bounds=manual_bounds, vary_only=vary_only, exclude_params=[bif_param],
        rel_window=rel_window, allowed_bounds=allowed_bounds, extras=extras)
    model_source.require_variation(resolution, bif_param, target['label'])
    bounds = resolution.bounds
    out_dir = out_dir or os.path.join(str(config.OUTPUT_DIR / 'lhs'),
                                      f"lhs_{target['label']}_{uuid.uuid4().hex[:8]}")
    os.makedirs(out_dir, exist_ok=True)

    samples = lhs_sample_params(bounds, n_samples, seed=seed)
    source_path = Path(target['path'])
    Path(out_dir, 'design.json').write_text(json.dumps({
        'model': str(source_path.resolve()), 'model_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'seed': seed, 'log_probes': log_probes, 'bif_param': bif_param, 'y_species': target['y_species'],
        'requested_range': list(param_range), 'bounds': bounds, 'samples': samples,
        'conserved_totals': parsed.get('conserved_totals', {}),
        'reconstruction': {n:str(e) for n,e in parsed.get('reconstruction', {}).items()},
    }, indent=2), encoding='utf-8')
    print(f"Model {target['label']}: continuation range for {bif_param}: "
          f"{tuple(param_range)} ({source})", flush=True)
    print(_format_samples_preview(samples, bounds), flush=True)
    y_species = target['y_species']
    control_entry = (_run_control(resolution, bif_param, param_range, out_dir, y_species, parsed,
                                  log_probes) if control else None)
    failed_control = ({"unavailable": control_entry["error"]}
                      if control_entry and control_entry.get("auto_unavailable") else {})
    results = _run_samples(samples, bif_param, param_range, out_dir, y_species, parsed,
                           log_probes=log_probes, **failed_control)
    results = _finish(results, resolution, control_entry, len(parsed['species']))
    results.time_note = target.get('time_note')
    results.identity = model_source.identity_line(model)
    results.window = (tuple(param_range), source)
    results.bif_baseline = parsed['param_values'].get(bif_param)
    return results, target['label'], out_dir


def _plot_branch_with_stability(branch, color, label, bif_param, y_species):
    """One branch on the current axes, solid where stable, dashed where unstable. A None
    label stays unlabeled: samples of an already labeled category add no legend entry."""
    x = branch[bif_param]
    y = branch[y_species]
    stab = branch.stability()
    boundaries = [0] + [abs(s) - 1 for s in stab]

    first = True
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] + 1
        is_stable = stab[i] < 0
        seg_label = None
        if first and label is not None:
            seg_label = f'{label} (solid=stable, dashed=unstable)'
        first = False
        plt.plot(x[start:end], y[start:end],
                  linestyle='-' if is_stable else '--',
                  color=color, label=seg_label)


CATEGORY_COLORS = {'bistable': 'tab:red', 'branch_point': 'tab:green', 'hopf': 'tab:purple',
                   'single_fold': 'tab:orange', 'multiple_folds': 'tab:brown', 'ultrasensitive': 'tab:blue',
                   'graded': 'tab:cyan', 'incomplete': 'tab:brown',
                   'no_bifurcation': 'tab:gray'}


def plot_combined(results, bif_param, y_species, out_path, title=None):
    """Overlays every successful sample's bifurcation diagram on one axis,
    coloured by its PRIMARY category (taxonomy.py: bistable, branch_point, hopf,
    single_fold, ultrasensitive, graded, incomplete, no_bifurcation), with the number of
    samples in each in the legend. The unperturbed baseline, when it was run, is
    drawn in black. Less interesting categories are drawn first so the
    interesting ones stay on top."""
    successes = [r for r in results if r['error'] is None]
    for r in successes:
        taxonomy.analyse(r, bif_param, y_species)
    counts = {c: sum(1 for r in successes if r['category'] == c) for c in taxonomy.PRIORITY}
    present = [c for c in taxonomy.PRIORITY if counts[c]]

    plt.figure(figsize=(10, 8))
    found_types = set()
    for cat in reversed(present):
        label = f"{cat} (n={counts[cat]})"
        for r in (r for r in successes if r['category'] == cat):
            for branch in r['all_branches']:
                _plot_branch_with_stability(branch, CATEGORY_COLORS[cat], label, bif_param, y_species)
                label = None  # only label the first branch of each category
            # Small/light markers so hundreds of overlaid samples don't turn into a wall
            found_types |= mark_special_points(r['all_branches'], r['special'],
                                                bif_param, y_species, alpha=0.5, annotate=False)
    control = getattr(results, 'control', None)
    if control is not None and control['error'] is None:
        label = 'baseline (unperturbed)'
        for branch in control['all_branches']:
            _plot_branch_with_stability(branch, 'black', label, bif_param, y_species)
            label = None

    n_fail = len(results) - len(successes)
    subtitle = f" ({len(successes)}/{len(results)} samples converged"
    subtitle += f", {n_fail} failed)" if n_fail else ")"
    plt.xlabel(bif_param)
    plt.ylabel(f'{y_species} (steady state)')
    plt.title((title or f'LHS scan: {y_species} vs {bif_param}') + subtitle)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = {c: i for i, c in enumerate(taxonomy.CATEGORY_ORDER)}
    pairs = sorted(zip(handles, labels), key=lambda hl: order.get(hl[1].split(' (n=')[0], -1))
    handles = [h for h, _ in pairs] + special_point_legend_handles(found_types)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8,
               title='Category', borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def write_summary_csv(results, out_path):
    """One row per sample: id, primary category, every feature, signature,
    error, then the parameter values. The baseline control, when run, is the
    first row (id 'control')."""
    rows = list(results)
    control = getattr(results, 'control', None)
    if control is not None:
        rows = [control] + rows
    param_names = sorted(rows[0]['params'].keys()) if rows else []
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'category', 'features', 'signature', 'error',
                         'LP_count', 'BP_count', 'HB_count', 'max_coexisting_stable'] + param_names)
        for r in rows:
            counts = taxonomy.point_counts(r.get('special'))
            evidence = r.get('classification', {}).get('evidence', {})
            writer.writerow([
                r['sample_id'], r.get('category') or '', ", ".join(r.get('features') or []),
                r['signature'] or '', r['error'] or '',
                counts['LP'], counts['BP'], counts['HB'], evidence.get('max_coexisting_stable', '')
            ] + [r['params'][p] for p in param_names])
    return out_path


MAX_TABLE_ROWS = 50


def format_sampled_sets(results):
    """Every sample's parameter values and outcome as a text table, so the
    sampled parameter sets come back with the report (the same values are in
    the summary CSV)."""
    if not results:
        return "Sampled parameter sets: none"
    names = sorted(results[0]['params'])
    varied = [p for p in names if len({r['params'][p] for r in results}) > 1]
    fixed = {p: results[0]['params'][p] for p in names if p not in varied}
    width = max([len(r.get('category') or 'failed') for r in results] + [7])
    lines = [f"Sampled parameter sets ({len(results)}; varied parameters only):",
             f"  id  {'outcome':<{width}}  " + "  ".join(f"{p:>12}" for p in varied)]
    for r in results[:MAX_TABLE_ROWS]:
        outcome = 'failed' if r['error'] else (r.get('category') or r['signature'] or '')
        lines.append(f"  {r['sample_id']:>2}  {outcome:<{width}}  "
                     + "  ".join(f"{r['params'][p]:.17g}" for p in varied))
    if len(results) > MAX_TABLE_ROWS:
        lines.append(f"  ... {len(results) - MAX_TABLE_ROWS} more rows in the summary CSV")
    if fixed:
        lines.append("Held at baseline in every sample: "
                     + ", ".join(f"{p}={v:.17g}" for p, v in fixed.items()))
    return "\n".join(lines)


def _signature_counts(results):
    counts = {}
    for r in results:
        k = r['signature'] if r['error'] is None else 'FAILED'
        counts[k] = counts.get(k, 0) + 1
    return counts


def _category_counts(results, bif_param, y_species):
    counts = {}
    for r in results:
        c = taxonomy.analyse(r, bif_param, y_species)['category']
        counts[c] = counts.get(c, 0) + 1
    return {c: counts[c] for c in taxonomy.CATEGORY_ORDER if c in counts}


def _states_at(entry, bif_param, y_species, value):
    """(stable, unstable) equilibria of one sample at bif_param = value: where its computed
    branches cross that value, each with AUTO's own stability there; the same equilibrium
    met on two branches (they share their start point) counts once."""
    found = {}
    for branch in entry.get('all_branches') or []:
        try:
            x = np.asarray(branch[bif_param], dtype=float)
            y = np.asarray(branch[y_species], dtype=float)
            stab = branch.stability()
        except Exception:
            continue
        ends = [0] + [abs(s) - 1 for s in stab]
        for k in range(len(ends) - 1):
            lo, hi = ends[k], min(ends[k + 1], len(x) - 1)
            for i in range(lo, hi):
                a, b = x[i] - value, x[i + 1] - value
                if a == 0 or a * b < 0:
                    yv = y[i] if a == 0 else y[i] + (y[i + 1] - y[i]) * a / (a - b)
                    key = round(float(yv), 6 - int(np.floor(np.log10(abs(yv) + 1e-300))) if yv else 6)
                    found.setdefault(key, stab[k] < 0)
            if hi == len(x) - 1 and x[hi] == value:
                found.setdefault(round(float(y[hi]), 9), stab[k] < 0)
    stable = sum(found.values())
    return stable, len(found) - stable


def _at_own_value(results, bif_param, y_species):
    """Per sample, whether it has a stable steady state at the model's own value of the
    continuation parameter: the measured answer to 'how many samples settle / oscillate'
    (a bifurcation signature alone says what happens somewhere in the window)."""
    value = getattr(results, 'bif_baseline', None)
    if value is None:
        return ""
    settle, none_hopf, none_other, missed = [], [], [], []
    for r in results:
        if r['error']:
            continue
        stable, unstable = _states_at(r, bif_param, y_species, value)
        hopf = any(ty == 'HB' for _, ty, _ in (r.get('special') or []))
        if stable:
            settle.append(r['sample_id'])
        elif unstable:
            (none_hopf if hopf else none_other).append(r['sample_id'])
        else:
            missed.append(r['sample_id'])
    n = len(settle) + len(none_hopf) + len(none_other) + len(missed)
    lines = [f"At the model's own {bif_param} = {value:g} (each sample's other parameters as sampled), "
             f"from the continued branches and their stability:",
             f"  a stable steady state there: {len(settle)}/{n}",
             f"  no stable steady state there, a Hopf point on the branch (oscillation): "
             f"{len(none_hopf)}/{n}" + (f" -- samples {none_hopf}" if none_hopf else "")]
    if none_other:
        lines.append(f"  no stable steady state there, no Hopf point (another attractor, or runaway): "
                     f"{len(none_other)}/{n} -- samples {none_other}")
    if missed:
        lines.append(f"  {bif_param} = {value:g} not on the computed branches: {len(missed)}/{n} "
                     f"-- samples {missed}")
    return "\n".join(lines)


def _range_section(results):
    res = getattr(results, 'resolution', None)
    return res.describe() if res is not None else ""


def _report(header, results, bif_param, y_species, plot_path, csv_path, goals, separate_paths):
    lines = [header,
             getattr(results, 'identity', None) or "",
             getattr(results, 'time_note', None) or "",
             f"Continuation window: {summarize_windows(results)}",
             _range_section(results),
             f"Combined plot: {_shown(plot_path)}",
             f"Summary CSV: {_shown(csv_path)}"]
    job_dir = Path(csv_path).parent
    lines.extend([f"Full-precision sampling design: {_shown(job_dir / 'design.json')}",
                  f"Structured results: {_shown(job_dir / 'results.json')}"])
    if separate_paths:
        lines.append(f"Per-category subplots: {separate_paths[0]}")
        lines.append(f"Category-counts bar chart: {separate_paths[1]}")
        lines.append("Per-category Excel files:\n"
                     + "\n".join(f"  {c}: {p}" for c, p in separate_paths[2].items()))
    lines.append(f"Categories (primary, per sample): {_category_counts(results, bif_param, y_species)}")
    lines.append(f"Bifurcation signatures found: {_signature_counts(results)}")
    lines.append(_at_own_value(results, bif_param, y_species))
    lines.append(format_sampled_sets(results))
    goal_text = lhs_goals.report(goals, results, bif_param, y_species,
                                 window_text=summarize_windows(results))
    if goal_text:
        lines.append(goal_text)
    return "\n".join(x for x in lines if x)


def _separate(results, bif_param, y_species, out_dir, title):
    from dynalyst.regime.categorize_lhs import run_categorized_report
    plot_path, counts_path, excels, _ = run_categorized_report(
        results, bif_param, y_species, out_dir, title=title)
    return plot_path, counts_path, excels


def _sampling_desc(rel_window, allowed_bounds, additive_offset, spread_factor, base, resolution):
    kind, amount = getattr(resolution, 'strategy', (None, None)) or (None, None)
    if kind == 'rel':
        of = "each parameter's allowed range" if allowed_bounds else f"{base}"
        return f"within +-{amount * 100:g}% of {of}, around the baseline"
    if kind == 'offset':
        return f"[baseline, baseline +{amount:g}]"
    if kind == 'fold':
        return f"a fold of {amount:g} around {base} (log-uniform {1 / amount:.3g}x .. {amount:g}x)"
    return "the explicit bounds given"


def save_batch_results(results, out_dir, bif_param, y_species, artifacts):
    """Save complete outcomes before propagating an AUTO installation failure."""
    successful = sum(r.get('error') is None for r in results)
    path = Path(out_dir, 'results.json')
    path.write_text(json.dumps({
        'status': 'completed' if successful == len(results) else 'partial' if successful else 'failed',
        'n_requested': len(results), 'n_completed': successful, 'n_failed': len(results)-successful,
        'bif_param': bif_param, 'y_species': y_species,
        'samples': [{k:v for k,v in r.items() if k not in ('all_branches', '_traceback')} for r in results],
        'artifacts': {**artifacts, 'design': str(Path(out_dir,'design.json'))},
    }, indent=2, default=str), encoding='utf-8')
    unavailable = next((r['error'] for r in results if r.get('auto_unavailable')), None)
    if unavailable:
        raise AutoUnavailable(f"{unavailable}. The design and per-sample failures were saved in {path}.")
    return path


def run_and_report(n_samples, bif_param=None, y_species=None, param_range=None,
                    additive_offset=None, manual_bounds=None, seed=None, model=None,
                    spread_factor=None, rel_window=None, allowed_bounds=None, vary_only=None,
                    extras=None, goal=None, separate=False, log_probes=False):
    """Runs the batch and writes ONE overlay plot of everything found (coloured
    by category, with counts), a summary CSV, and a text report. separate=True
    additionally splits the samples into per-category subplots and Excel
    files; `goal` ('bistability', 'branch point', ...) adds a control run of
    the baseline and the found / not-found goal report (lhs_goals.py). model: any
    SBML/XPP model file, sampled around its own baseline."""
    bif_param, y_species = model_source.resolve_target(model, bif_param, y_species)
    goals, extras, control = lhs_goals.prepare(goal, extras)
    results, out_dir = run_lhs_batch(
        n_samples, bif_param, y_species, param_range, additive_offset, manual_bounds, seed,
        model=model, spread_factor=spread_factor, rel_window=rel_window,
        allowed_bounds=allowed_bounds, vary_only=vary_only, extras=extras, control=control,
        log_probes=log_probes)
    plot_path = os.path.join(out_dir, 'combined_bifurcation_diagram.png')
    csv_path = os.path.join(out_dir, 'summary.csv')
    plot_combined(results, bif_param, y_species, plot_path)
    write_summary_csv(results, csv_path)
    sep = _separate(results, bif_param, y_species, out_dir, "LHS scan by category") if separate else None
    save_batch_results(results, out_dir, bif_param, y_species,
                       {'overlay': plot_path, 'summary_csv': csv_path})

    what = (f"{n_samples} parameter sets sampled around the baseline of "
            f"{model_source.display(model)}, "
            f"{_sampling_desc(rel_window, allowed_bounds, additive_offset, spread_factor, 'its baseline', results.resolution)}"
            + (f"; only varying {vary_only}" if vary_only else "")
            + (f"; {bif_param} is the continuation parameter, so it is followed along its range "
               "and not sampled" if vary_only and bif_param in vary_only else ""))
    header = (f"LHS batch scan complete: {what} "
              f"(bif_param={bif_param}, y_species={y_species}).")
    return _report(header, results, bif_param, y_species, plot_path, csv_path, goals, sep), \
        results, out_dir


def _shown(path):
    """A file path as shown to the user: relative to the project when inside it."""
    from dynalyst import config
    return config.display_path(path) if path else path
