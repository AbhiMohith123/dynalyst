"""Interactive twins of the plots: every PNG a tool saves also gets an interactive Plotly page next
to it (<name>.html: zoom, pan, hover values, legend toggles; works offline, plotly.js is written
once per folder) and the figure itself (<name>.plotly.json, which the notebook shows inline).

Two ways in:
  * install_savefig_hook(): any matplotlib figure saved as .png is converted (lines, markers,
    scatter, bars, filled areas, images, text labels, axis lines, log axes, subplots). A
    conversion that fails never costs the PNG.
  * bifurcation_figure(): AUTO diagrams, drawn from the branch data like XPPAUT's / MatCont's
    window: a menu picks the y variable among all states, buttons step to each labelled point
    (LP / BP / HB / ...), hover shows the parameter, every state and the stability.

Set DYNALYST_INTERACTIVE=0 to switch it off.
"""
import json
import os
from pathlib import Path

import numpy as np

_DASH = {"-": "solid", "solid": "solid", "--": "dash", "dashed": "dash", ":": "dot",
         "dotted": "dot", "-.": "dashdot", "dashdot": "dashdot"}
_SYMBOL = {"o": "circle", "s": "square", "^": "triangle-up", "v": "triangle-down", "D": "diamond",
           "d": "diamond", "x": "x", "+": "cross", "*": "star", ".": "circle", "p": "pentagon",
           "h": "hexagon", "<": "triangle-left", ">": "triangle-right"}


def enabled() -> bool:
    return os.environ.get("DYNALYST_INTERACTIVE", "1") not in ("0", "false", "no")


def _hex(c):
    from matplotlib.colors import to_hex
    try:
        return to_hex(c, keep_alpha=False)
    except (ValueError, TypeError):
        return None


def _grid(fig):
    """(rows, cols, {axes: (row, col)}) from the axes' subplot specs; extra axes (colorbars,
    insets) are left out."""
    place, rows, cols = {}, 1, 1
    for ax in fig.axes:
        spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
        if spec is None:
            continue
        g = spec.get_gridspec()
        rows, cols = max(rows, g.nrows), max(cols, g.ncols)
        r, c = divmod(spec.num1, g.ncols)
        place.setdefault(ax, (r + 1, c + 1))
    if not place and fig.axes:
        place[fig.axes[0]] = (1, 1)
    return rows, cols, place


def mpl_to_plotly(fig):
    """A Plotly figure drawn from a matplotlib figure: what each axes shows, with the same colours,
    line styles, labels, limits and log scales."""
    import plotly.graph_objects as go
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from plotly.subplots import make_subplots

    rows, cols, place = _grid(fig)
    titles = [""] * (rows * cols)
    for ax, (r, c) in place.items():
        titles[(r - 1) * cols + (c - 1)] = ax.get_title()
    out = make_subplots(rows=rows, cols=cols, subplot_titles=titles if any(titles) else None)
    seen = set()
    for ax, (r, c) in place.items():
        where = dict(row=r, col=c)
        for ln in ax.get_lines():
            if not isinstance(ln, Line2D) or not ln.get_visible():
                continue
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if x.size == 0:
                continue
            color = _hex(ln.get_color())
            if ln.get_transform() != ax.transData:          # axhline / axvline (axes coordinates)
                kind = dict(line_color=color, line_dash=_DASH.get(ln.get_linestyle(), "solid"),
                            line_width=1)
                if np.allclose(x, x[0]):
                    out.add_vline(x=float(x[0]), **kind, **where)
                elif np.allclose(y, y[0]):
                    out.add_hline(y=float(y[0]), **kind, **where)
                continue
            style = ln.get_linestyle()
            marker = ln.get_marker()
            has_line = style not in ("None", "", " ", None)
            has_marker = marker not in ("None", "", " ", None)
            label = ln.get_label()
            show = bool(label) and not label.startswith("_") and label not in seen
            seen.add(label)
            out.add_trace(go.Scatter(
                x=x, y=y, name=label if show else None, showlegend=show,
                mode="+".join([m for m, on in (("lines", has_line), ("markers", has_marker)) if on]) or "lines",
                line=dict(color=color, width=max(1.0, ln.get_linewidth()), dash=_DASH.get(style, "solid")),
                marker=dict(color=_hex(ln.get_markerfacecolor()) or color, size=max(4.0, ln.get_markersize()),
                            symbol=_SYMBOL.get(marker, "circle"),
                            line=dict(color=_hex(ln.get_markeredgecolor()) or color, width=1)),
                opacity=ln.get_alpha() if ln.get_alpha() is not None else 1.0), **where)
        for coll in ax.collections:
            if isinstance(coll, PathCollection) and len(coll.get_offsets()):
                xy = np.asarray(coll.get_offsets(), float)
                fc = coll.get_facecolors()
                colors = [_hex(f) for f in fc] if len(fc) > 1 else (_hex(fc[0]) if len(fc) else None)
                label = coll.get_label()
                show = bool(label) and not label.startswith("_")
                out.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", name=label if show else None,
                                         showlegend=show, marker=dict(color=colors, size=7)), **where)
            elif isinstance(coll, PolyCollection):        # fill_between and other filled areas
                for path in coll.get_paths():
                    v = np.asarray(path.vertices, float)
                    if len(v) < 3:
                        continue
                    fc = coll.get_facecolor()
                    out.add_trace(go.Scatter(x=v[:, 0], y=v[:, 1], fill="toself", mode="lines",
                                             line=dict(width=0), showlegend=False, hoverinfo="skip",
                                             fillcolor=_hex(fc[0]) if len(fc) else None, opacity=0.3), **where)
        bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_width() and p.get_height()]
        if bars:
            out.add_trace(go.Bar(x=[p.get_x() + p.get_width() / 2 for p in bars],
                                 y=[p.get_height() for p in bars], base=[p.get_y() for p in bars],
                                 width=[p.get_width() for p in bars],
                                 marker_color=[_hex(p.get_facecolor()) for p in bars],
                                 showlegend=False, opacity=0.8), **where)
        for im in ax.images:
            arr = np.asarray(im.get_array(), float)
            if arr.ndim == 2:
                x0, x1, y0, y1 = im.get_extent()
                out.add_trace(go.Heatmap(z=arr, x=np.linspace(x0, x1, arr.shape[1]),
                                         y=np.linspace(y0, y1, arr.shape[0]), colorscale="Viridis"), **where)
        for t in ax.texts:
            if t.get_text().strip() and t.get_transform() == ax.transData:
                x, y = t.get_position()
                out.add_annotation(x=x, y=y, text=t.get_text(), showarrow=False, yshift=12,
                                   font=dict(size=12), **where)
        out.update_xaxes(title_text=ax.get_xlabel(), range=list(ax.get_xlim()) if ax.get_xscale() != "log"
                         else list(np.log10(np.clip(ax.get_xlim(), 1e-300, None))),
                         type="log" if ax.get_xscale() == "log" else None, **where)
        out.update_yaxes(title_text=ax.get_ylabel(), range=list(ax.get_ylim()) if ax.get_yscale() != "log"
                         else list(np.log10(np.clip(ax.get_ylim(), 1e-300, None))),
                         type="log" if ax.get_yscale() == "log" else None, **where)
    title = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else ""
    out.update_layout(template="plotly_white", title=title or None, hovermode="closest",
                      height=max(450, 380 * rows), legend=dict(itemclick="toggle"))
    return out


def save(pfig, png_path) -> str:
    """Write <png stem>.html (plotly.js inside it) and <png stem>.plotly.json; the html path.
    Jupyter serves .html files sandboxed: a page there cannot load a plotly.min.js beside it."""
    png_path = Path(png_path)
    html = png_path.with_suffix(".html")
    pfig.write_html(html, include_plotlyjs=True, full_html=True,
                    config={"scrollZoom": True, "displaylogo": False})
    png_path.with_suffix(".plotly.json").write_text(pfig.to_json(), encoding="utf-8")
    return str(html)


def install_savefig_hook():
    """Every matplotlib figure saved as .png also gets its interactive twin (once per process)."""
    if not enabled():
        return
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, "_dynalyst_twin", False):
        return
    original = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        if (isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png")
                and not getattr(self, "_dynalyst_no_twin", False)):
            try:
                save(mpl_to_plotly(self), fname)
            except Exception:       # an interactive twin must never cost the PNG
                pass
        return result

    savefig._dynalyst_twin = True
    Figure.savefig = savefig


def bifurcation_figure(branch_rows, cycle_rows, points, param, plot_state, title="", x_range=None):
    """An AUTO diagram as XPPAUT / MatCont show it: equilibrium branches (solid stable, dashed red
    unstable), cycle max/min (green), labelled points; a menu for the y variable (all states),
    buttons that step to each labelled point, and hover with the parameter, all states and the
    stability. x_range: the view it opens on, with the y axis fitted to what lies inside it (the
    branch data can reach far past it; zoom out or 'whole diagram' shows all)."""
    import plotly.graph_objects as go

    states = [k for k in (branch_rows[0] if branch_rows else {})
              if k not in ("branch", param, "stability", "L2-NORM")]
    if plot_state in states:
        states = [plot_state] + [s for s in states if s != plot_state]
    fig = go.Figure()
    owners = []          # per trace: the y variable it follows ("*" = every variable)
    by_branch = {}
    for r in branch_rows:
        by_branch.setdefault(r["branch"], []).append(r)
    hover = (f"{param} = %{{x:.6g}}<br>" + "<br>".join(f"{s} = %{{customdata[{i}]:.6g}}"
                                                       for i, s in enumerate(states))
             + "<br>%{meta}<extra>branch %{text}</extra>")
    for b, rows in by_branch.items():
        start = 0
        for i in range(1, len(rows) + 1):
            if i < len(rows) and rows[i]["stability"] == rows[start]["stability"]:
                continue
            seg = rows[start:min(i + 1, len(rows))]
            stab = rows[start]["stability"]
            stable = stab == "stable"
            fig.add_trace(go.Scatter(
                x=[q[param] for q in seg], y=[q[plot_state] for q in seg], mode="lines",
                line=dict(color="black" if stable else "red", width=2.5, dash="solid" if stable else "dash"),
                name="stable" if stable else "unstable", legendgroup=stab, showlegend=False,
                customdata=[[q.get(s, np.nan) for s in states] for q in seg],
                text=[str(b)] * len(seg), meta=stab, hovertemplate=hover))
            owners.append("*")
            start = i
    for stab, color, dash in (("stable", "black", "solid"), ("unstable", "red", "dash")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=stab, legendgroup=stab,
                                 line=dict(color=color, width=2.5, dash=dash)))
        owners.append("*")
    cyc = [r for r in cycle_rows if f"{plot_state}_max" in r]
    if cyc:
        for key in ("max", "min"):
            fig.add_trace(go.Scatter(
                x=[r[param] for r in cyc], y=[r[f"{plot_state}_{key}"] for r in cyc], mode="markers",
                marker=dict(size=4, color=["green" if r["stability"] == "stable" else "lightgreen" for r in cyc]),
                name=f"cycles ({key})", customdata=[[r.get("period")] for r in cyc],
                hovertemplate=f"{param} = %{{x:.6g}}<br>{plot_state} {key} = %{{y:.6g}}<br>"
                              "period = %{customdata[0]:.6g}<extra>cycle</extra>"))
            owners.append(plot_state)
    for pt in points:
        full = pt.get("full_state") or {}
        fig.add_trace(go.Scatter(
            x=[pt[param]], y=[full.get(plot_state, pt.get(plot_state))], mode="markers+text",
            text=[pt["type"]], textposition="top right", name=pt["type"], showlegend=False,
            marker=dict(size=13, color="gold", line=dict(color="black", width=1.5),
                        symbol={"LP": "square", "BP": "triangle-up", "HB": "circle"}.get(pt["type"], "x")),
            customdata=[[full.get(s, np.nan) for s in states]],
            hovertemplate=f"<b>{pt['type']}</b> {param} = %{{x:.6g}}<br>"
                          + "<br>".join(f"{s} = %{{customdata[{i}]:.6g}}" for i, s in enumerate(states))
                          + "<extra></extra>"))
        owners.append("*")

    def y_for(state):
        ys = []
        for tr, own in zip(fig.data, owners):
            if tr.customdata is not None and own == "*" and len(tr.customdata) and len(tr.customdata[0]) == len(states):
                ys.append([row[states.index(state)] for row in tr.customdata])
            else:
                ys.append(list(tr.y) if tr.y is not None else [None])
        return ys

    menus = []
    if len(states) > 1:
        menus.append(dict(
            buttons=[dict(label=s, method="update",
                          args=[{"y": y_for(s), "visible": [own in ("*", s) for own in owners]},
                                {"yaxis.title.text": s, "yaxis.autorange": True}]) for s in states],
            direction="down", x=0.0, xanchor="left", y=1.02, yanchor="bottom", showactive=True))
    if points:
        xs = [r[param] for r in branch_rows] or [p[param] for p in points]
        span = (max(xs) - min(xs)) or 1.0
        steps = [dict(label="whole diagram", method="relayout",
                      args=[{"xaxis.autorange": True, "yaxis.autorange": True}])]
        for pt in sorted(points, key=lambda p: p[param]):
            x = pt[param]
            near = [r[plot_state] for r in branch_rows if abs(r[param] - x) <= 0.06 * span
                    and np.isfinite(r.get(plot_state, np.nan))]
            lo, hi = (min(near), max(near)) if near else (None, None)
            pad = 0.1 * ((hi - lo) or abs(hi) or 1.0) if near else 0
            view = {"xaxis.range": [x - 0.06 * span, x + 0.06 * span]}
            view.update({"yaxis.range": [lo - pad, hi + pad]} if near else {"yaxis.autorange": True})
            # the y range is for the y variable drawn first; after switching it, double-click resets
            steps.append(dict(label=f"{pt['type']} {param}={x:.5g}", method="relayout", args=[view]))
        menus.append(dict(buttons=steps, direction="down", x=0.25, xanchor="left", y=1.02, yanchor="bottom"))
    fig.update_layout(template="plotly_white", title=title or None, updatemenus=menus, hovermode="closest",
                      xaxis_title=param, yaxis_title=plot_state, height=620,
                      title_y=0.97, margin=dict(t=120), legend=dict(itemclick="toggle"))
    if x_range is not None:
        inside = [r[plot_state] for r in branch_rows if x_range[0] <= r[param] <= x_range[1]
                  and np.isfinite(r.get(plot_state, np.nan))]
        inside += [r[f"{plot_state}_{k}"] for r in cyc for k in ("max", "min")
                   if x_range[0] <= r[param] <= x_range[1]]
        fig.update_xaxes(range=list(x_range))
        if inside:
            lo, hi = min(inside), max(inside)
            pad = 0.05 * ((hi - lo) or abs(hi) or 1.0)
            fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig
