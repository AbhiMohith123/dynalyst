"""Notebook helper: send a query to an agent and display, inline, every plot
and table produced during the call.

Formerly agent/ask_updated.py, which did os.chdir('/E-drive/...') and only
knew one agent. Output files are found by comparing snapshots of
config.WORKDIR (LHS / AUTO pipeline) and config.PLOTS_DIR (research tools) taken
before and after the call
rather than by parsing the agent's own paraphrased text for a path
(unreliable, since gpt-4o-mini doesn't always repeat exact paths verbatim).
"""
import glob
import os
import re
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Image, Markdown, display

from dynalyst import config
from dynalyst.regime import plot_registry
from dynalyst.report import save_report  # noqa: F401  (the last answer as HTML + PDF)

AGENTS = ("unified", "research")
LAST_RUN = None  # the unified agent's last RunRecord (plan, steps, follow_ups, ...)

# Math in an answer as \( .. \) and \[ .. \] (the model's habit) is rendered by Jupyter but
# not by VS Code's notebook viewer, which reads only $ .. $ and $$ .. $$ (both read those)
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.S)


def markdown_math(text: str) -> str:
    """An answer with its math delimiters written as $ .. $ / $$ .. $$, outside code."""
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text or "", flags=re.S)
    for i in range(0, len(parts), 2):          # even parts are outside code
        p = _DISPLAY_MATH.sub(lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n", parts[i])
        parts[i] = _INLINE_MATH.sub(lambda m: "$" + m.group(1).strip() + "$", p)
    return "".join(parts)


_IMAGE_LINK = re.compile(r"!\[(.*?)\]\((.+?\.(?:png|jpe?g|gif))\)", re.I)  # paths may hold spaces or brackets
_SHOWN = set()        # the plots the last answer already displayed


def _same_file(path) -> str:
    return os.path.normcase(os.path.abspath(path))


def show_answer(text):
    """An answer with its plots displayed where they stand. A Markdown image link is read relative
    to the notebook's folder, and viewers differ on what they load, so it can show as a broken
    image: the file itself is displayed instead, or a line says it is missing."""
    _SHOWN.clear()
    parts = _IMAGE_LINK.split(text or "")
    for i in range(0, len(parts), 3):
        if parts[i].strip():
            display(Markdown(markdown_math(parts[i])))
        if i + 2 < len(parts):
            path = Path(parts[i + 2])
            path = path if path.is_absolute() else config.PROJECT_ROOT / path
            if _same_file(path) in _SHOWN:
                continue
            if path.is_file():
                show_plot(path)
                _SHOWN.add(_same_file(path))
            else:
                display(Markdown(f"*(plot not found: {parts[i + 2]})*"))


def show_plot(png) -> None:
    """The PNG inline, and a link to its interactive twin (<name>.html, written next to every PNG)
    when there is one. The page opens in its own tab and loads plotly.js from its own folder, so it
    works offline; drawn inside the notebook, plotly.js would have to come from a CDN."""
    display(Image(filename=str(png)))
    html = Path(png).with_suffix(".html")
    if html.is_file():
        try:
            href = os.path.relpath(html, os.getcwd()).replace(os.sep, "/")
        except ValueError:          # another drive: no relative path
            href = html.as_posix()
        display(HTML(f'<a href="{href}" target="_blank">Open the interactive version</a> '
                     f'<span style="color:#888">(zoom, pan, hover; {html.name})</span>'))


def show_figure(fig):
    """An interactive Plotly figure as HTML, plotly.js loaded from its CDN. fig.show() sends a Plotly
    mime type that only some front ends can draw: without a renderer the figure stays blank."""
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))


def _invoke(agent, query, history, verbose):
    if agent == "unified":
        global LAST_RUN
        from dynalyst.agents.unified_agent import run as agent_run
        # a person is working in the notebook: the run may pause and ask before AUTO
        LAST_RUN = agent_run(query, conversation_history=history, verbose=verbose,
                             approval="ask")
        show_answer(LAST_RUN.final_answer)
        return LAST_RUN.final_answer
    elif agent == "research":
        from dynalyst.agents.research_agent import ask as agent_ask
    else:
        raise ValueError(f"agent must be one of {AGENTS}, got {agent!r}")
    answer = agent_ask(query, conversation_history=history, verbose=verbose)
    show_answer(answer)
    return answer


def ask(query, show_csv=True, show_xlsx=True, agent="unified", history=None, verbose=True):
    """Send a natural-language query to an agent and display inline
    whatever plot(s)/CSV(s)/Excel file(s) actually got written during this
    call, PLUS any interactive Plotly figure(s) pushed to plot_registry
    during this call (e.g. by categorized_lhs_scan's
    plot_categorized_interactive) -- those never touch disk, so they can't
    be found by the file snapshot below and have to be popped from the
    registry explicitly.

    agent: "unified" (default) -- the agent: orchestrator, teams,
    sub-agents, execution guard and final check over every capability. The
    older single-purpose agent stays available as "research" (RAG + analysis
    tools). history is passed to the agent as conversation_history.
    Returns the answer text.
    A person is working here, so the unified agent may pause before an AUTO-07p
    run with no named parameter and ask which candidate to run.
    """
    # omnipath_network_check writes its network plot and Excel report outside PLOTS_DIR
    # and the LHS scans write their plots, summary and Excel files under lhs
    roots = [str(config.WORKDIR), str(config.PLOTS_DIR), str(config.OUTPUT_DIR / "omnipath"),
             str(config.OUTPUT_DIR / "lhs")]
    exts = ('png', 'csv', 'xlsx')

    def _snapshot():
        """{path: (mtime_ns, size)} of every output file now on disk."""
        files = {}
        for root in roots:
            for ext in exts:
                for f in glob.glob(os.path.join(root, '**', f'*.{ext}'), recursive=True):
                    try:
                        st = os.stat(f)
                    except OSError:
                        continue
                    files[os.path.abspath(f)] = (st.st_mtime_ns, st.st_size)
        return files

    # files new or rewritten during this call: compared with a snapshot taken before
    # it, not with the clock (a mounted disk's clock, or a coarse timestamp, can differ)
    before = _snapshot()
    result = _invoke(agent, query, history, verbose)
    after = _snapshot()
    changed = [f for f, sig in after.items() if before.get(f) != sig]

    def _new_files(ext):
        return sorted((f for f in changed if f.lower().endswith(f'.{ext}')),
                      key=lambda f: after[f][0])

    for png in _new_files('png'):
        if _same_file(png) not in _SHOWN:
            show_plot(png)

    if show_csv:
        for csv_path in _new_files('csv'):
            df = pd.read_csv(csv_path)
            print(f"\nSummary ({len(df)} samples) - {os.path.basename(csv_path)}:")
            display(df)

    if show_xlsx:
        # categorize_lhs.py writes one .xlsx per bifurcation category (e.g.
        # bistable.xlsx, ultrasensitive.xlsx) -- read_excel needs openpyxl
        # installed (pip install openpyxl), same as writing them did.
        for xlsx_path in _new_files('xlsx'):
            df = pd.read_excel(xlsx_path)
            print(f"\n{os.path.basename(xlsx_path)} ({len(df)} samples):")
            display(df)

    # Interactive Plotly figures (categorized or combined) never get
    # written to disk -- pop_all() clears the registry so a figure from
    # THIS call never bleeds into the next ask() call's output.
    for fig in plot_registry.pop_all():
        show_figure(fig)

    return result
