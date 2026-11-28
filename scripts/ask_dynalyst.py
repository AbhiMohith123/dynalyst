"""Chat with Dynalyst from a JupyterLab cell: the answer and every new plot appear in the
cell, and each plot is also saved as a PNG under outputs/ask/<time>/ with a download link.

    import sys; sys.path.insert(0, "scripts")
    from ask_dynalyst import ask, new_chat
    ask("Run a Gillespie simulation of Markevich2004.xml at MAPKK = 48 ...")
    ask("yes")            # answer a question the agent asked (e.g. approval before AUTO-07P)
    new_chat()            # start a fresh conversation

This file is self-contained: deleting it removes the feature and changes nothing else.
"""
import glob
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

_EXTS = ("png", "svg", "pdf", "csv")
_SKIP = ("/ask/", "/showcase/", "/records/", "/__pycache__/")
_state = {"agent": None, "history": []}


def _files():
    from dynalyst import config
    out = {}
    for root in {str(config.OUTPUT_DIR), str(config.PLOTS_DIR)}:
        for ext in _EXTS:
            for f in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
                if any(s in f.replace("\\", "/") for s in _SKIP):
                    continue
                try:
                    st = os.stat(f)
                except OSError:
                    continue
                out[os.path.abspath(f)] = (st.st_mtime_ns, st.st_size)
    return out


def _agent(approval):
    if _state["agent"] is None:
        os.environ.setdefault("DYNALYST_LESSONS", "0")
        from dynalyst.orchestration.agent import Agent
        from dynalyst.orchestration.state import SessionState
        _state["agent"] = Agent(approval=approval, verbose=False, save_records=True,
                                state=SessionState(persist=False))
    return _state["agent"]


def new_chat():
    """Forget the conversation and start a new session."""
    _state["agent"], _state["history"] = None, []
    print("New conversation started.")


def ask(prompt, approval="ask", width=900, show_files=True):
    """Send one message; show the answer and the new plots in the cell; save the plots."""
    from IPython.display import FileLink, Image, Markdown, display
    agent = _agent(approval)
    before, t0 = _files(), time.time()
    rec = agent.run(prompt, _state["history"])
    _state["history"] += [{"role": "user", "content": prompt},
                          {"role": "assistant", "content": rec.final_answer}]
    after = _files()
    new = sorted((f for f, sig in after.items() if before.get(f) != sig), key=lambda f: after[f][0])
    # only the files this request made: its answer or its tool results name the file or its folder
    # (other processes, e.g. a batch of evaluations, may write plots into outputs/ at the same time)
    text = (rec.final_answer or "") + "".join(str(st.get("output") or "") for st in (rec.steps or []))
    text = text.replace("\\", "/")
    def _mine(f):
        par = Path(f).parent.name           # a per-run folder, e.g. Markevich2004_20261120_120527_54b3ff25c9
        return Path(f).name in text or (len(par) >= 16 and any(c.isdigit() for c in par) and par in text)
    new = [f for f in new if _mine(f)]
    out = ROOT / "outputs" / "ask" / time.strftime("%Y%m%d_%H%M%S")
    saved = []
    for f in new:
        out.mkdir(parents=True, exist_ok=True)
        tgt = out / Path(f).name
        if tgt.exists():
            tgt = out / f"{Path(f).stem}_{len(saved)}{Path(f).suffix}"
        shutil.copy(f, tgt)
        saved.append(tgt)
    display(Markdown(rec.final_answer or "_(no answer)_"))
    for f in saved:
        if f.suffix == ".png":
            display(Markdown(f"**{f.name}**"))
            display(Image(filename=str(f), width=width))
    if show_files and saved:
        display(Markdown(f"*Saved in* `{os.path.relpath(out, ROOT)}` *({round(time.time() - t0)} s):*"))
        for f in saved:
            display(FileLink(os.path.relpath(f, Path.cwd()) if Path.cwd() in f.parents else str(f)))
    if getattr(rec, "awaiting_approval", False):
        display(Markdown("**Dynalyst is waiting for your answer.** Reply with `ask(\"yes\")`, "
                         "`ask(\"no\")`, or name another choice, e.g. `ask(\"use MKP3\")`."))
    return [str(f) for f in saved]
