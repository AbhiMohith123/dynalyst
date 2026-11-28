"""
Plotly equivalents of the matplotlib plots of categorize_lhs.py.

These exist ONLY to give an interactive (zoomable, hoverable) version of the
same diagrams for display inside a Jupyter output cell -- the matplotlib
PNGs are still generated too, since those are what's referenced from
reports/theses. Written to standalone HTML files (include_plotlyjs=True, so
they render with no internet access) rather than returned as figure objects,
because these functions run inside a LangChain tool call, not inside the
notebook process itself.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dynalyst.regime import plot_registry
from dynalyst.regime.bifurcation_markers import MARKER_STYLE

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PLOTLY_MARKER_SYMBOL = {'LP': 'circle', 'BP': 'square', 'HB': 'star'}


def _stability_segments(branch, bif_param, y_species):
    """Same boundary logic as batch_lhs_runner._plot_branch_with_stability: splits a
    branch into contiguous stable/unstable runs using AUTO's own stability()
    sign convention (negative = stable, positive = unstable)."""
    x = branch[bif_param]
    y = branch[y_species]
    stab = branch.stability()
    boundaries = [0] + [abs(s) - 1 for s in stab]
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] + 1
        is_stable = stab[i] < 0
        segments.append((x[start:end], y[start:end], is_stable))
    return segments


def add_branch_traces(fig, branch, color, label, bif_param, y_species,
                       hovertext=None, legendgroup=None, row=None, col=None):
    """Adds one Plotly trace per stability segment of `branch`. Only the
    first (stable-or-not) segment carries a legend entry, matching the
    matplotlib version's one-label-per-branch behavior, so overlaying many
    samples doesn't flood the legend with duplicate entries. row/col are
    optional Plotly subplot coordinates (1-indexed) -- pass them when
    adding to a make_subplots() grid; omitted, the trace goes to whatever
    fig.add_trace's own default targets (fine for a single-panel figure)."""
    segments = _stability_segments(branch, bif_param, y_species)
    first = True
    for x, y, is_stable in segments:
        trace = go.Scatter(
            x=list(x), y=list(y), mode='lines',
            line=dict(color=color, dash='solid' if is_stable else 'dash', width=1.5),
            name=label if first else None,
            legendgroup=legendgroup if legendgroup is not None else label,
            showlegend=first and label is not None,
            hovertemplate=(hovertext + '<extra></extra>') if hovertext else
                          f'{bif_param}=%{{x:.4g}}<br>{y_species}=%{{y:.4g}}<extra></extra>',
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
        first = False


def _add_special_point_traces(fig, all_branches, special, bif_param, y_species,
                               row, col, showlegend_types):
    """Plotly equivalent of bifurcation_markers.mark_special_points: adds
    one marker trace per special point found, onto a specific subplot (row, col).
    showlegend_types
    is a set the caller passes in and this mutates in place, so each point
    TYPE (LP/BP/HB) gets exactly one legend entry across the WHOLE figure,
    not one per occurrence -- otherwise a category with 15 bistable samples
    would add 30 identical "LP (fold)" legend rows."""
    for idx, ty, number in special or []:
        style = MARKER_STYLE.get(ty)
        if style is None:
            continue
        x, y = all_branches[number][bif_param][idx], all_branches[number][y_species][idx]
        first_of_type = ty not in showlegend_types
        showlegend_types.add(ty)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(symbol=PLOTLY_MARKER_SYMBOL.get(ty, 'circle'),
                        color=style['color'], size=style['markersize'] + 2,
                        line=dict(width=1, color='white')),
            name=style['label'], legendgroup=f'marker_{ty}',
            showlegend=first_of_type,
            hovertemplate=f"{ty}<br>{bif_param}=%{{x:.4g}}<br>{y_species}=%{{y:.4g}}<extra></extra>",
        ), row=row, col=col)


def plot_categorized_interactive(buckets, bif_param, y_species, title=None):
    """Interactive (zoomable/hoverable) equivalent of
    categorize_lhs.plot_categorized: one subplot panel per category present
    in `buckets` (bistable / branch_point / ultrasensitive / graded, or
    however many actually got samples -- empty categories simply don't get
    a panel, same as the matplotlib version), each overlaying that
    category's samples with stability styling (solid=stable, dashed=
    unstable) plus LP/BP/HB markers, all inside ONE Plotly figure.

    Pushed to plot_registry -- NO HTML file is written. If you're calling
    this from inside ask()/agent.invoke(), the notebook's own ask() helper
    already pops plot_registry and calls fig.show() for you. If you're
    calling the underlying categorize_lhs functions directly (bypassing
    the agent), pop and show it yourself in the same cell:

        import plot_registry
        for fig in plot_registry.pop_all():
            fig.show()
    """
    cats = list(buckets.keys())
    n = len(cats)
    if n == 0:   # every sample failed: nothing to draw (the counts chart shows the failures)
        return None
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)  # ceil division

    fig = make_subplots(rows=nrows, cols=ncols,
                         subplot_titles=[f"{cat} (n={len(buckets[cat])})" for cat in cats])

    showlegend_types = set()
    for i, cat in enumerate(cats):
        row, col = i // ncols + 1, i % ncols + 1
        for j, r in enumerate(buckets[cat]):
            color = PALETTE[j % len(PALETTE)]
            param_str = '<br>'.join(f'{k}={v:.4g}' for k, v in sorted(r['params'].items()))
            hover = f"sample {r['sample_id']}<br>{param_str}"
            for branch in r['all_branches']:
                add_branch_traces(fig, branch, color, None, bif_param, y_species,
                                   hovertext=hover, row=row, col=col)
            _add_special_point_traces(fig, r['all_branches'], r['special'], bif_param,
                                       y_species, row, col, showlegend_types)
        fig.update_xaxes(title_text=bif_param, row=row, col=col)
        fig.update_yaxes(title_text=f'{y_species} (steady state)', row=row, col=col)

    fig.update_layout(
        title=title or f'LHS samples by category: {y_species} vs {bif_param}',
        template='plotly_white', height=650 * nrows, showlegend=True,
    )
    plot_registry.push(fig)
    return fig


def plot_category_counts_interactive(buckets, n_total, title=None):
    """Interactive (hoverable) equivalent of categorize_lhs.plot_category_counts
    -- one bar per category with the sample count. Same 'unclassified' bar
    handling as the static version (any samples that couldn't be assigned
    a category at all are shown separately, not dropped) so the bars
    always sum to n_total. Pushed to plot_registry -- no HTML file."""
    cats = list(buckets.keys())
    counts = [len(buckets[c]) for c in cats]
    n_classified = sum(counts)
    if n_total > n_classified:
        cats = cats + ['unclassified']
        counts = counts + [n_total - n_classified]

    fig = go.Figure(go.Bar(
        x=cats, y=counts, text=counts, textposition='outside',
        marker_color=PALETTE[0],
        hovertemplate='%{x}: %{y} samples<extra></extra>',
    ))
    fig.update_layout(
        title=title or f'Category breakdown ({n_total} samples total)',
        xaxis_title='category', yaxis_title='number of samples',
        template='plotly_white', height=450,
    )
    plot_registry.push(fig)
    return fig