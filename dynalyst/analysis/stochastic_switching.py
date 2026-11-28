"""Switching events in stochastic runs, and a plot that shows them clearly.

The distribution plot shows every run at once, so a single jump between two stable
states is hard to see. This module finds each jump in the recorded runs (the read-out
leaves the core of one peak of the distribution and enters the core of another), and
draws one run in full with every jump marked, plus a zoom on up to three jumps. When the
recorded grid is too coarse to resolve a jump (fewer than eight points across it), a few
new Gillespie runs are started from the state the run had just before the jump, on a fine
time grid, so the jump itself is resolved event by event.
"""
from __future__ import annotations

import numpy as np


def _cores(mode_info):
    """[lo, hi] of the core of each peak: closer to the peak than half-way to the valley."""
    ms = mode_info["modes"]
    centers = np.array([m["at"] for m in ms])
    vall = np.array(mode_info["valleys"])
    lo = np.array([centers[i] - 0.5 * (centers[i] - (vall[i - 1] if i > 0 else -np.inf))
                   for i in range(len(ms))])
    hi = np.array([centers[i] + 0.5 * ((vall[i] if i < len(vall) else np.inf) - centers[i])
                   for i in range(len(ms))])
    return centers, lo, hi


def find_switches(series, mode_info, grid):
    """Every jump between peak cores in every run.

    series: (runs, time) read-out in molecules. Returns a list of dicts with the run,
    the last grid index in the old core (i_leave), the first in the new core
    (i_arrive), the two peaks (0-based, in order of the read-out) and the times."""
    if not mode_info or len(mode_info.get("modes", [])) < 2:
        return []
    _, lo, hi = _cores(mode_info)
    events = []
    for r, s in enumerate(series):
        state, last_in = None, None
        for i, x in enumerate(s):
            if not np.isfinite(x):
                break
            hit = np.flatnonzero((x >= lo) & (x <= hi))
            if not hit.size:
                continue
            k = int(hit[0])
            if state is not None and k != state:
                events.append({"run": r, "i_leave": last_in, "i_arrive": i,
                               "from_peak": state, "to_peak": k,
                               "t_leave": float(grid[last_in]), "t_arrive": float(grid[i]),
                               "duration_upper_bound": float(grid[i] - grid[last_in])})
            state, last_in = k, i
    return events


def replay_jump(a_fn, V, x_start, j, target_lo, target_hi, window, ssa_run,
                n_points=400, tries=30, keep=3, max_events=2_000_000, seed=1000):
    """New Gillespie runs from x_start (molecules) over [0, window]: keep up to `keep`
    that reach the target core of read-out j, preferring those that get there first.
    Returns (traces, tries_used, skipped)."""
    a0 = float(np.sum(np.clip(a_fn(np.asarray(x_start, dtype=float)), 0, None)))
    if a0 * window > max_events:
        return [], 0, f"skipped: about {a0 * window:.2g} events per replay"
    fine = np.linspace(0.0, window, n_points)
    out, used = [], 0
    for k in range(tries):
        used += 1
        traj, _, _, truncated = ssa_run(a_fn, V, x_start, fine,
                                        np.random.default_rng(seed + k), max_events)
        x = traj[:, j]
        inside = np.flatnonzero((x >= target_lo) & (x <= target_hi))
        if inside.size and not truncated:
            out.append((int(inside[0]), fine, x))
            if len(out) >= 3 * keep:
                break
    out.sort(key=lambda o: o[0])            # the most direct crossings first
    return [(t, x) for _, t, x in out[:keep]], used, None


def _ode_modes(entry):
    """The two (or more) states to switch between, taken from the ODE's stable steady states
    when the pooled distribution has fewer than two peaks -- as when every run escaped one way,
    out of a state that is about to disappear near a fold. Valleys: the unstable steady states
    between them, else the midpoint. In molecules, like the series."""
    om = float(entry.get("omega") or 1.0)
    att = entry.get("attractors") or []
    st = sorted(a["readout"] * om for a in att
                if a.get("kind") == "steady state" and a.get("stability") == "stable" and a.get("readout") is not None)
    if len(st) < 2:
        return None
    un = sorted(a["readout"] * om for a in att
                if a.get("kind") == "steady state" and a.get("stability") != "stable" and a.get("readout") is not None)
    valleys = []
    for lo, hi in zip(st[:-1], st[1:]):
        mid = [u for u in un if lo < u < hi]
        valleys.append(mid[0] if mid else 0.5 * (lo + hi))
    return {"modes": [{"at": x} for x in st], "valleys": valleys, "n_modes": len(st), "from": "ODE stable states"}


def plot(entry, grid, label, path, a_fn=None, V=None, ssa_run=None, max_zooms=3):
    """One run in full with each jump marked, and a zoom on up to `max_zooms` jumps.

    entry: a distribution entry (series in molecules, modes, attractors, omega, readout);
    with a_fn, V, ssa_run and entry['states'] (runs, time, species; molecules) the zooms
    add new Gillespie runs started just before each jump. Returns a summary dict, or None
    when no run switched."""
    md = entry.get("modes")
    if not md or len(md.get("modes", [])) < 2:       # one peak: every run ended in the same state
        md = _ode_modes(entry) or md
    events = find_switches(entry["series"], md, grid)
    if not events:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    counts = np.bincount([e["run"] for e in events])
    run = int(np.argmax(counts))
    mine = [e for e in events if e["run"] == run]
    # zoom on jumps spread over the run, both directions if possible
    picks = []
    for e in mine:
        if len(picks) < max_zooms and (not picks or e["to_peak"] != picks[-1]["to_peak"]
                                       or len(mine) <= max_zooms):
            picks.append(e)
    picks = picks or mine[:max_zooms]
    centers, lo, hi = _cores(md)
    om = entry["omega"]
    stable = [a["readout"] * om for a in entry.get("attractors", [])
              if a.get("kind") == "steady state" and a.get("stability") == "stable"]
    saddle = [a["readout"] * om for a in entry.get("attractors", [])
              if a.get("kind") == "steady state" and a.get("stability") == "unstable"]
    dt = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    names = entry.get("state_names")
    j = names.index(entry["readout"]) if names and entry["readout"] in names else None
    can_replay = (a_fn is not None and V is not None and ssa_run is not None
                  and entry.get("states") is not None and j is not None)

    n = len(picks)
    fig = plt.figure(figsize=(max(11, 4.0 * n), 8.0))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.32)
    top = fig.add_subplot(gs[0, :])
    s = entry["series"][run]
    top.step(grid, s, where="post", color="tab:blue", lw=0.8)
    for k_, (a, b) in enumerate(zip(lo, hi)):
        top.axhspan(max(a, np.nanmin(s)), min(b, np.nanmax(s)), color="0.85", alpha=0.5,
                    lw=0, zorder=0)
    for v in stable:
        top.axhline(v, color="navy", ls="--", lw=1.1)
    for v in saddle:
        top.axhline(v, color="tab:red", ls=":", lw=1.1)
    for k_, e in enumerate(mine):
        top.axvline(e["t_arrive"], color="tab:red", lw=0.8, alpha=0.6)
    for k_, e in enumerate(picks):
        top.annotate(str(k_ + 1), (e["t_arrive"], np.nanmax(s)), color="tab:red",
                     fontsize=9, ha="center", va="bottom", fontweight="bold")
    top.set_xlabel("time")
    top.set_ylabel(f"{entry['readout']} (molecules)")
    top.set_title(f"One {entry['method']} run at Ω = {om:g}: {len(mine)} jump(s) (red lines); "
                  f"the numbered ones are enlarged below", fontsize=10, pad=14)

    summary = {"run_shown": run, "jumps_in_run_shown": len(mine),
               "jumps_in_all_runs": len(events), "zoomed": []}
    for k_, e in enumerate(picks):
        ax = fig.add_subplot(gs[1, k_])
        i0, i1 = e["i_leave"], e["i_arrive"]
        span = max(grid[i1] - grid[i0], dt)
        pad = max(2.0 * span, 5 * dt)
        a_t, b_t = grid[i0] - pad, grid[i1] + pad
        sel = (grid >= a_t) & (grid <= b_t)
        ax.step(grid[sel], s[sel], where="post", color="tab:blue", lw=1.4,
                label="recorded run", zorder=3)
        ax.plot(grid[sel], s[sel], "o", color="tab:blue", ms=2.5, zorder=3)
        ax.axvspan(grid[i0], grid[i1], color="tab:red", alpha=0.08, lw=0)
        info = {"jump": k_ + 1, "t_leave": e["t_leave"], "t_arrive": e["t_arrive"],
                "from": float(centers[e["from_peak"]]), "to": float(centers[e["to_peak"]]),
                "duration_upper_bound": e["duration_upper_bound"]}
        coarse = (i1 - i0) < 8          # the recorded grid barely resolves this jump
        if can_replay and coarse:
            # start just before the run crosses the barrier (the saddle, or the valley of
            # the distribution), so the new runs show the jump rather than the long wait
            up = e["to_peak"] > e["from_peak"]
            a_, b_ = sorted((centers[e["from_peak"]], centers[e["to_peak"]]))
            between = [v for v in saddle if a_ < v < b_]   # the first barrier on the way
            barrier = ((min(between) if up else max(between)) if between
                       else md["valleys"][min(e["from_peak"], e["to_peak"])])
            seg = s[i0:i1 + 1]
            before = [i0 + q for q, x in enumerate(seg) if (x < barrier if up else x > barrier)]
            i_s = before[-1] if before else i0
            window = float(b_t - grid[i_s])
            traces, used, skipped = replay_jump(
                a_fn, V, entry["states"][run, i_s], j, lo[e["to_peak"]], hi[e["to_peak"]],
                window, ssa_run, seed=1000 + 37 * k_)
            for m_, (tt, xx) in enumerate(traces):
                hit = np.flatnonzero((xx >= lo[e["to_peak"]]) & (xx <= hi[e["to_peak"]]))
                stop = min(len(tt), hit[0] + max(20, len(tt) // 10)) if hit.size else len(tt)
                tt, xx = tt[:stop], xx[:stop]
                ax.step(grid[i_s] + tt, xx, where="post", lw=0.9, alpha=0.85,
                        color=plt.cm.Oranges(0.45 + 0.18 * m_),
                        label="new Gillespie runs from the state before the barrier" if m_ == 0 else None)
            info["replays"] = {"kept": len(traces), "tries": used, "skipped": skipped}
        for v in stable:
            ax.axhline(v, color="navy", ls="--", lw=1.0)
        for v in saddle:
            ax.axhline(v, color="tab:red", ls=":", lw=1.0)
        ax.set_xlim(a_t, b_t)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useMathText=True)
        ax.set_xlabel("time")
        ax.set_title(f"Jump {k_ + 1}: {centers[e['from_peak']]:.4g} → {centers[e['to_peak']]:.4g}"
                     f" molecules\nat t ≈ {e['t_arrive']:.4g}, transit ≤ {e['duration_upper_bound']:.3g}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        if k_ == 0:
            ax.set_ylabel(f"{entry['readout']} (molecules)")
        summary["zoomed"].append(info)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="tab:blue", lw=1.4, marker="o", ms=3, label="recorded run"),
               Line2D([], [], color="navy", ls="--", label="stable ODE state"),
               Line2D([], [], color="tab:red", ls=":", label="saddle (unstable state)"),
               Patch(color="0.85", label="core of each peak"),
               Patch(color="tab:red", alpha=0.15, label="transit")]
    if any(z.get("replays", {}).get("kept") for z in summary["zoomed"]):
        handles.append(Line2D([], [], color=plt.cm.Oranges(0.6), lw=1,
                              label="new Gillespie runs from just before the barrier"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{label}: switching between the stable states", fontsize=11)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    summary["path"] = path
    # counts over every jump: the list below is capped at 50
    first = {}
    for e in events:
        first.setdefault(e["run"], e["t_leave"])
    summary["first_jump_t_per_run"] = list(first.values())
    summary["up"] = sum(e["to_peak"] > e["from_peak"] for e in events)
    summary["down"] = len(events) - summary["up"]
    summary["jumps"] = [{k: v for k, v in e.items() if k not in ("i_leave", "i_arrive")}
                        for e in events[:50]]
    return summary
