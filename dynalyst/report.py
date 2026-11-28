"""One run as a report: the question, the answer with its plots, and how the answer was
reached (plan, every tool call, the checks), written as HTML and PDF.

From a notebook: ask(...), then save_report("name"); or save_report(run=rec) for a run record
from run(...). The PDF is drawn with PyMuPDF (already a dependency), the HTML opens in any
browser. Plots are copied next to the report, so the folder can be moved or sent as it is.
"""
import html
import re
import shutil
from datetime import datetime
from pathlib import Path

from dynalyst import config

_IMG_MD = re.compile(r"!\[(.*?)\]\((.+?\.(?:png|jpe?g|gif))\)", re.I)
_IMG_TAG = re.compile(r"<img([^>]*?)src=\"([^\"]+)\"([^>]*)>", re.I)
_ARGS_SHOWN = ("model_name_or_path", "model", "param_name", "bif_param", "param_min", "param_max",
               "parameters", "omega", "t_end", "n_runs", "n_samples", "query", "pdf")
_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.4; color: #1d2a33; }
h1 { font-size: 18pt; color: #0e6e77; margin-bottom: 2pt; }
h2 { font-size: 13pt; color: #0e6e77; margin-top: 14pt; }
h3 { font-size: 11pt; }
.meta { color: #5a6b78; font-size: 9pt; }
.question { background-color: #e6edec; padding: 8pt; }
table { border-collapse: collapse; margin: 6pt 0; }
th, td { border: 1px solid #c8d3d6; padding: 3pt 5pt; font-size: 9pt; text-align: left; }
th { background-color: #e6edec; }
code { font-family: monospace; font-size: 9pt; }
"""


def _to_html(md: str) -> str:
    """Markdown to HTML with tables: python-markdown, else mistune (comes with Jupyter), else
    the text as it is."""
    try:
        import markdown
        return markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        pass
    try:
        import mistune
        return mistune.html(md)
    except ImportError:
        return f"<pre>{html.escape(md)}</pre>"


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def _steps_table(rec) -> str:
    rows = []
    for i, s in enumerate([s for s in rec.steps if s.get("tool")], 1):
        args = s.get("arguments") or {}
        shown = ", ".join(f"{k}={args[k]}" for k in _ARGS_SHOWN if k in args)
        review = (s.get("reviewer") or {}).get("verdict") or ""
        rows.append(f"<tr><td>{i}</td><td>{html.escape(str(s.get('tool')))}</td>"
                    f"<td>{html.escape(shown[:160])}</td><td>{html.escape(str(s.get('status') or ''))}</td>"
                    f"<td>{html.escape(review)}</td></tr>")
    if not rows:
        return "<p>No tool was called.</p>"
    return ("<table><tr><th>Step</th><th>Tool</th><th>Key arguments</th><th>Status</th>"
            "<th>Reviewer</th></tr>" + "".join(rows) + "</table>")


def save_report(name: str | None = None, run=None, pdf: bool = True) -> dict:
    """Write the last answer (or `run`, a run record) as <name>.html and <name>.pdf in
    outputs/reports/, with its plots copied into <name>_files/. Returns the paths."""
    if run is None:
        from dynalyst import notebook
        run = notebook.LAST_RUN
    if run is None or not getattr(run, "final_answer", ""):
        raise ValueError("No answer to report yet: ask a question first, or pass run=<run record>.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or f"report_{stamp}").strip("_") or f"report_{stamp}"
    folder = config.OUTPUT_DIR / "reports"
    files = folder / f"{name}_files"
    files.mkdir(parents=True, exist_ok=True)

    # every plot the answer links, then any other plot of the run, copied next to the report
    copied = {}

    def copy(path: str) -> str | None:
        src = _resolve(path)
        if not src.is_file():
            return None
        if str(src) not in copied:
            dst = files / f"{len(copied) + 1:02d}_{src.name}"
            shutil.copyfile(src, dst)
            copied[str(src)] = f"{files.name}/{dst.name}"
        return copied[str(src)]

    answer = run.final_answer
    answer = _IMG_MD.sub(lambda m: f"![{m.group(1)}]({copy(m.group(2)) or m.group(2)})", answer)
    extra = [p for p in (run.plots() if hasattr(run, "plots") else [])
             if str(_resolve(p)) not in copied and copy(p)]
    body = _to_html(answer)
    if extra:
        body += "<h2>Other plots of this run</h2>" + "".join(
            f'<p><img src="{copied[str(_resolve(p))]}"></p>' for p in extra)
    # a fixed width: the PDF page is narrow and an image wider than it is cut
    body = _IMG_TAG.sub(lambda m: f'<img{m.group(1)}src="{m.group(2)}" width="460"{m.group(3)}>', body)

    model = run.model if isinstance(run.model, str) else ", ".join(
        sorted({str(v) for v in (run.model or {}).values()}))
    page = f"""<h1>Dynalyst report</h1>
<p class="meta">{html.escape(str(run.started))} · model: {html.escape(model)} · {run.seconds} s</p>
<h2>Question</h2><p class="question">{html.escape(run.query)}</p>
<h2>Answer</h2>{body}
<h2>How the answer was reached</h2>
<p>Plan: {html.escape('; '.join(map(str, run.plan)) or '(none written)')}</p>
{_steps_table(run)}
<p class="meta">Full run record: {html.escape(str(run.record_path or '(not saved)'))}</p>"""

    html_path = folder / f"{name}.html"
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><title>{html.escape(name)}</title>"
                         f"<style>{_CSS} body {{ max-width: 900px; margin: auto; }}</style>{page}",
                         encoding="utf-8")
    out = {"html": str(html_path), "files": str(files)}
    if pdf:
        import pymupdf
        pdf_path = folder / f"{name}.pdf"
        story = pymupdf.Story(html=page, user_css=_CSS, archive=str(folder))
        writer = pymupdf.DocumentWriter(str(pdf_path))
        box = pymupdf.paper_rect("a4")
        more = True
        while more:
            device = writer.begin_page(box)
            more, _ = story.place(box + (50, 50, -50, -50))
            story.draw(device)
            writer.end_page()
        writer.close()
        out["pdf"] = str(pdf_path)
    return out
