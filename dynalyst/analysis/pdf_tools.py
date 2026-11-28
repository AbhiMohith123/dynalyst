"""
Ad-hoc PDF reading, for any PDF the user points at (uploaded file, a path
under data/papers, or elsewhere) -- as opposed to build_index.py/ingest_pdf.py,
which are an OFFLINE batch pipeline that pre-chunks a fixed corpus into the
RAG vector index. This tool is for "read this PDF and summarize/extract
things from it" on demand, for a paper that may not be (or need to be) in
that index at all.

The tool itself only EXTRACTS text (full text, or organized by detected
section/page) -- the actual summarization is left to the calling agent (the
LLM), consistent with every other tool in this project returning raw
structured data rather than pre-written prose.
"""
import json
from pathlib import Path

try:  # PyMuPDF (its old name "fitz" prints a deprecation warning)
    import pymupdf as fitz
except ImportError:
    import fitz

from dynalyst import config
from dynalyst.rag.ingest_pdf import detect_section, split_into_paragraphs

MAX_CHARS_DEFAULT = 60_000  # keep a single tool response reasonably sized


def _resolve_pdf_path(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate

    search_dirs = [config.PROJECT_ROOT / "data" / "papers", Path("/mnt/user-data/uploads")]
    query = path_or_name.lower()
    if query.endswith(".pdf"):
        query = query[:-4]
    for d in search_dirs:
        if not d.exists():
            continue
        match = config.best_name_match(query, d.rglob("*.pdf"))
        if match:
            return match

    there = sorted({p.name for d in search_dirs if d.exists() for p in d.rglob("*.pdf")})
    raise FileNotFoundError(
        f"No PDF matching '{path_or_name}' found (checked the given path, "
        f"{[str(d) for d in search_dirs]}). Pass a full path instead"
        + (f", or one of the PDFs there are: {', '.join(there)}." if there else ".")
    )


def read_pdf(path_or_name: str, page_range: list[int] | None = None,
             max_chars: int = MAX_CHARS_DEFAULT) -> str:
    """
    Tool: extract text from any PDF (by path, or by name searched in
    data/papers and uploaded files), organized by page and detected section
    heading, for the calling model to read, summarize, or answer questions
    about. page_range (1-indexed, inclusive [start, end]) limits extraction
    to part of a long document.
    """
    try:
        path = _resolve_pdf_path(path_or_name)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        return json.dumps({"error": f"Could not open '{path}': {e}"})

    n_pages = doc.page_count
    start, end = 1, n_pages
    if page_range:
        start, end = page_range[0], page_range[1]
        start = max(1, start)
        end = min(n_pages, end)

    metadata = doc.metadata or {}
    pages_out = []
    current_section = "unknown"
    total_chars = 0
    truncated = False

    for page_num in range(start, end + 1):
        page_text = doc[page_num - 1].get_text("text")
        for line in page_text.split("\n"):
            detected = detect_section(line)
            if detected:
                current_section = detected

        if total_chars + len(page_text) > max_chars:
            truncated = True
            break

        pages_out.append({
            "page": page_num,
            "section": current_section,
            "text": page_text.strip(),
        })
        total_chars += len(page_text)

    doc.close()

    return json.dumps({
        "path": str(path),
        "title": metadata.get("title") or path.stem,
        "author": metadata.get("author"),
        "n_pages_total": n_pages,
        "pages_extracted": [start, min(end, pages_out[-1]["page"] if pages_out else start)],
        "truncated": truncated,
        "pages": pages_out,
        "note": (
            "Truncated -- call again with a page_range to continue reading the rest "
            "of the document." if truncated else None
        ),
    }, indent=2, default=str)


def list_available_pdfs() -> str:
    """Tool: list PDFs already available locally (data/papers + any uploads),
    so the model doesn't have to guess a filename."""
    found = []
    for d in [config.PROJECT_ROOT / "data" / "papers", Path("/mnt/user-data/uploads")]:
        if d.exists():
            found.extend(str(p) for p in d.rglob("*.pdf"))
    # papers downloaded by fetch_open_access_paper are named by PMCID: say which paper each is
    index = config.PROJECT_ROOT / "data" / "papers" / "index.json"
    try:
        known = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    except ValueError:
        known = {}
    titles = {p: known[k]["title"] for p in found for k in known
              if Path(p).name.startswith(k) and not Path(p).name[len(k):len(k) + 1].isdigit()
              and (known[k] or {}).get("title")}
    return json.dumps({"pdfs": found, **({"titles": titles} if titles else {})}, indent=2)


TOOL_DEFINITIONS = [
    {
        "name": "read_pdf",
        "description": (
            "Extract text from a PDF (any paper -- by path, or by name searched in "
            "data/papers and uploaded files), organized by page and detected section "
            "heading (abstract/introduction/methods/results/discussion/etc). Use this "
            "to read a paper and then summarize it, extract specific information "
            "(e.g. reported parameter values, model equations, experimental results), "
            "or answer questions about its content. For a long PDF, call again with "
            "page_range to continue past a truncation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path_or_name": {"type": "string", "description": "Full path to a PDF, or a name/partial name to search for."},
                "page_range": {"type": "array", "items": {"type": "integer"}, "description": "[start_page, end_page], 1-indexed inclusive. Omit to read from the start."},
                "max_chars": {"type": "integer", "description": "Cap on extracted characters per call. Default 60000."},
            },
            "required": ["path_or_name"],
        },
    },
    {
        "name": "list_available_pdfs",
        "description": "List PDF files already available locally (the project's papers directory and any files the user has uploaded), so a filename doesn't have to be guessed.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_FUNCTIONS = {
    "read_pdf": lambda **kwargs: read_pdf(**kwargs),
    "list_available_pdfs": lambda **kwargs: list_available_pdfs(**kwargs),
}
