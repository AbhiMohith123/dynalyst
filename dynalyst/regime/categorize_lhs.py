"""
Splits LHS batch results (from batch_lhs_runner.run_lhs_batch) by what each
sample showed, so every category gets
its own subplot panel and its own Excel sheet instead of one crowded combined
plot where you have to cross-reference a colour back to a sample.

The categories and their definitions live in taxonomy.py: bistable (stable coexistence),
branch_point (BP), hopf, single_fold, ultrasensitive / graded (monotonic
branches, split by the Goldbeter-Koshland response ratio), no_bifurcation,
failed. A sample can show several features; each has ONE primary category
(used for counts and the single overlay plot) and appears in every panel of
the features it shows in the separated view.

AUTO reports transcritical and pitchfork branch points alike as 'BP': the
category is branch_point, without guessing which.
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

from dynalyst.regime.batch_lhs_runner import (
    _plot_branch_with_stability, run_lhs_batch, write_summary_csv, save_batch_results)
from dynalyst.regime import model_source
from dynalyst.regime.bifurcation_markers import mark_special_points, special_point_legend_handles
from dynalyst.regime.taxonomy import ULTRASENSITIVE_THRESHOLD, FAILED, analyse


def categorize_results(results, bif_param, y_species,
                        ultrasensitive_threshold=ULTRASENSITIVE_THRESHOLD, multi=False):
    """{category_label: [entries]}. Failed samples are left out (the counts
    chart shows them as 'unclassified').

    multi=False: each sample goes under its PRIMARY category, so the buckets
    partition the classified samples. multi=True: a sample goes under EVERY
    feature it shows (the separated per-category view)."""
    buckets = {}
    for r in results:
        a = analyse(r, bif_param, y_species, ultrasensitive_threshold)
        if a['category'] == FAILED:
            continue
        for cat in (a['features'] if multi else [a['category']]):
            buckets.setdefault(cat, []).append(r)
    return buckets


def plot_categorized(buckets, bif_param, y_species, out_path, title=None):
    """One subplot panel per category (up to however many were found),
    each showing only that category's samples with stability + LP/BP/HB
    markers -- same visual language as the single combined plot, just
    split so you're not untangling 150 overlapping lines by eye."""
    cats = list(buckets.keys())
    n = len(cats)
    if n == 0:
        # every sample failed: say so in the figure rather than failing the whole
        # report -- the counts chart, CSV and report still show why they failed
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis('off')
        ax.text(0.5, 0.5, "No sample produced a bifurcation diagram, so there is nothing to\n"
                          "split by category. The category counts and the report show the failures.",
                ha='center', va='center', fontsize=11)
        fig.suptitle(title or f'LHS samples by category: {y_species} vs {bif_param}')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_path
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

    for i, cat in enumerate(cats):
        ax = axes[i // ncols][i % ncols]
        plt.sca(ax)
        entries = buckets[cat]
        found_types = set()
        for r in entries:
            for branch in r['all_branches']:
                _plot_branch_with_stability(branch, 'tab:blue', None, bif_param, y_species)
            found_types |= mark_special_points(r['all_branches'], r['special'],
                                                bif_param, y_species, alpha=0.6, annotate=False)
        ax.set_title(f"{cat} (n={len(entries)})")
        ax.set_xlabel(bif_param)
        ax.set_ylabel(y_species)
        ax.grid(alpha=0.3)
        handles = special_point_legend_handles(found_types)
        if handles:
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                      fontsize=7, borderaxespad=0)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(title or f'LHS samples by category: {y_species} vs {bif_param}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_category_counts(buckets, n_total, out_path, title=None):
    """A single bar chart: one bar per category showing how many samples
    landed in it (by primary category, so the bars add up to n_total). Any
    sample that could not be classified (a failed AUTO run) is shown as its
    own 'unclassified' bar rather than silently missing from the total."""
    cats = list(buckets.keys())
    counts = [len(buckets[c]) for c in cats]
    n_classified = sum(counts)
    if n_total > n_classified:
        cats = cats + ['unclassified']
        counts = counts + [n_total - n_classified]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cats)), 5))
    bars = ax.bar(cats, counts, color='tab:blue')
    for bar, count in zip(bars, counts):
        ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom')
    ax.set_ylabel('number of samples')
    ax.set_title(title or f'Category breakdown ({n_total} samples total)')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def write_category_excels(buckets, out_dir):
    """One .xlsx per category. Columns: sample_id, signature, the sample's
    primary category and all its features, response_ratio (monotonic samples
    only), every sampled model parameter, and an xppaut_par_lines column with
    a ready-to-paste 'par NAME=VALUE' block for that sample -- so checking a
    specific point in XPPAUT is copy-paste, not cross-referencing a plot
    colour back to a sample_id first."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for cat, entries in buckets.items():
        rows = []
        for r in entries:
            row = {'sample_id': r['sample_id'], 'signature': r['signature'],
                   'category': r.get('category'), 'features': ", ".join(r.get('features') or []),
                   'response_ratio': r.get('_response_ratio')}
            row.update(r['params'])
            par_lines = "\n".join(f"par {k}={v:.6g}" for k, v in sorted(r['params'].items()))
            row['xppaut_par_lines'] = par_lines
            rows.append(row)
        df = pd.DataFrame(rows)
        safe_name = cat.replace(' ', '_').replace('(', '').replace(')', '')
        path = os.path.join(out_dir, f'{safe_name}.xlsx')
        df.to_excel(path, index=False)
        paths[cat] = path
    return paths


def _render(results, buckets, feature_buckets, bif_param, y_species, out_dir, title, interactive):
    plot_path = os.path.join(out_dir, 'categorized_subplots.png')
    plot_categorized(feature_buckets, bif_param, y_species, plot_path, title=title)
    counts_plot_path = os.path.join(out_dir, 'category_counts.png')
    plot_category_counts(buckets, len(results), counts_plot_path,
                          title=(f"{title} - category breakdown" if title else None))
    excel_paths = write_category_excels(feature_buckets, out_dir)
    if interactive:
        from dynalyst.regime.interactive_plotting import (
            plot_categorized_interactive,
            plot_category_counts_interactive,
        )
        plot_categorized_interactive(feature_buckets, bif_param, y_species, title=title)
        plot_category_counts_interactive(buckets, len(results),
                                          title=(f"{title} - category breakdown" if title else None))
    return plot_path, counts_plot_path, excel_paths


def run_categorized_report(results, bif_param, y_species, out_dir, title=None,
                            ultrasensitive_threshold=ULTRASENSITIVE_THRESHOLD, interactive=True):
    """categorize -> per-category plot (matplotlib PNG) -> counts bar chart ->
    Excels -> optionally ALSO push interactive Plotly versions to
    plot_registry. The panels and Excels use the multi-label buckets (a sample
    appears under every feature it shows); the counts chart and the returned
    buckets use the primary category, so they add up. All plots draw from the
    same already-computed results: no extra AUTO runs.
    Returns (plot_path, counts_plot_path, excel_paths, buckets)."""
    os.makedirs(out_dir, exist_ok=True)
    buckets = categorize_results(results, bif_param, y_species,
                                  ultrasensitive_threshold=ultrasensitive_threshold)
    feature_buckets = categorize_results(results, bif_param, y_species,
                                          ultrasensitive_threshold=ultrasensitive_threshold,
                                          multi=True)
    plot_path, counts_plot_path, excel_paths = _render(
        results, buckets, feature_buckets, bif_param, y_species, out_dir, title, interactive)
    return plot_path, counts_plot_path, excel_paths, buckets


# ---------------------------------------------------------------------------
# Convenience: run a fresh batch and categorize it in one call
# ---------------------------------------------------------------------------

def run_and_categorize(n_samples, bif_param=None, y_species=None, param_range=None,
                        additive_offset=None, manual_bounds=None, seed=None,
                        ultrasensitive_threshold=ULTRASENSITIVE_THRESHOLD, model=None,
                        spread_factor=2.0, rel_window=None, allowed_bounds=None,
                        vary_only=None, extras=None, control=False, log_probes=False):
    """model: any SBML/XPP model file, sampled around its own baseline.
    extras: range_spec.RangeExtras; control: also run the unperturbed baseline."""
    bif_param, y_species = model_source.resolve_target(model, bif_param, y_species)
    results, out_dir = run_lhs_batch(
        n_samples=n_samples, bif_param=bif_param, y_species=y_species,
        param_range=param_range, additive_offset=additive_offset, manual_bounds=manual_bounds,
        seed=seed, model=model, spread_factor=spread_factor, rel_window=rel_window,
        allowed_bounds=allowed_bounds, vary_only=vary_only, extras=extras, control=control,
        log_probes=log_probes)
    title = f"LHS scan of {model_source.display(model)} by category"
    plot_path, counts_plot_path, excel_paths, buckets = run_categorized_report(
        results, bif_param, y_species, out_dir, title=title,
        ultrasensitive_threshold=ultrasensitive_threshold)
    csv_path = write_summary_csv(results, os.path.join(out_dir, 'summary.csv'))
    save_batch_results(results, out_dir, bif_param, y_species,
                       {'category_plot': plot_path, 'counts_plot': counts_plot_path,
                        'summary_csv': csv_path, **{f'excel_{k}':v for k,v in excel_paths.items()}})
    return plot_path, counts_plot_path, excel_paths, buckets, results, out_dir
