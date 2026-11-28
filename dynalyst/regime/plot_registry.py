"""
LangChain @tool functions can only RETURN TEXT (that's what the LLM reads
back). But since agent.invoke(...) runs the tools in-process -- not via a
subprocess -- the tool functions and the notebook that called agent.invoke
share the same Python process and the same imported modules.

So instead of writing a Plotly figure to an HTML file and re-reading it
back into the notebook (extra disk I/O, ~3-4MB of embedded JS per file),
each plotting call just pushes its live Figure object into this
module-level list. The notebook's ask() then pops whatever showed up and
calls fig.show() directly, which renders as a fully interactive
zoom/pan/hover widget right in the output cell.
"""
FIGURES = []


def push(fig):
    FIGURES.append(fig)


def pop_all():
    figs = list(FIGURES)
    FIGURES.clear()
    return figs
