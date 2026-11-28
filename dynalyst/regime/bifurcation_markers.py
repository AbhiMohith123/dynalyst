"""Special-point (LP/BP/HB) markers of the LHS plots. A special point is
(point index, type, branch index) into the sample's list of branches."""
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

MARKER_STYLE = {
    'LP': dict(marker='o', color='black', markersize=8, label='LP (fold)'),
    'BP': dict(marker='s', color='purple', markersize=8, label='BP (branch point)'),
    'HB': dict(marker='*', color='darkorange', markersize=12, label='HB (Hopf)'),
}


def mark_special_points(all_branches, special, bif_param, y_species, alpha=1.0, annotate=True):
    """Plots a marker (on the CURRENT matplotlib axes) at each special point. Returns the
    set of point types plotted, so the caller can build a legend for just those."""
    seen_types = set()
    for idx, ty, number in special or []:
        style = MARKER_STYLE.get(ty)
        if style is None:
            continue
        x, y = all_branches[number][bif_param][idx], all_branches[number][y_species][idx]
        seen_types.add(ty)
        plt.plot(x, y, marker=style['marker'], color=style['color'],
                  markersize=style['markersize'], linestyle='None', alpha=alpha,
                  zorder=5)
        if annotate:
            plt.annotate(ty, (x, y), textcoords="offset points",
                          xytext=(6, 6), fontsize=8, color=style['color'], zorder=6)
    return seen_types


def special_point_legend_handles(types):
    """Proxy Line2D handles for the special-point types found, so they can
    be appended to a plot's existing branch/signature legend entries."""
    handles = []
    for ty in sorted(types):
        style = MARKER_STYLE[ty]
        handles.append(Line2D([0], [0], marker=style['marker'], color=style['color'],
                               linestyle='None', markersize=style['markersize'],
                               label=style['label']))
    return handles
