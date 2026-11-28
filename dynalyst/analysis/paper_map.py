"""Map a paper to a model: which of the model's parameters and species the paper
states, with what values.

Deterministic, so nothing is invented: every parameter / species id (and name)
of the model is looked up as a whole word in the PDF's text; a number written
right after it ("k1 = 0.02", "k1: 0.02", a table row "k1  0.02  nM/s") is read
as the paper's value, with the page and the surrounding text to check it.
Species are also matched through their UniProt annotation (protein / gene
names), so "ERK2" in the paper finds species M. Units are NOT converted: a
value that differs may be the same quantity in other units -- the context says.
save_sbml applies the paper's values to a copy of the model.
"""
import json
import re

_MANTISSA = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUM = (rf"({_MANTISSA}(?:\s*[x×]\s*10\s*\^?\s*\(?[+-]?\d+\)?|[eE][+-]?\d+)?)"
        r"(?!\w|\.\d|\s*[x×]\s*10)")
# "=" may come out of a PDF as a control character; a unit may sit in parentheses
_SEP = r"\s*(?:[=:≈~\x01-\x08]|\([^)]{0,25}\))?\s*"


def _term_pattern(term):
    """The name as typeset: PDF text splits sub/superscripts onto their own line
    ("k1cat" -> "k1\ncat", "Km_1" -> "K m1"), so letters / digits may be apart."""
    parts = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d_]", term)
    return r"[\s_]*".join(re.escape(p) for p in parts)


def _num(text):
    t = re.sub(r"\s+", "", text)
    m = re.fullmatch(rf"({_MANTISSA})(?:[x×]10\^?\(?([+-]?\d+)\)?|[eE]([+-]?\d+))?", t)
    if not m:
        return None
    exp = m.group(2) or m.group(3)
    return float(m.group(1)) * (10 ** int(exp) if exp else 1)


def _clean(page):
    return (page.replace("−", "-").replace("–", "-").replace("ﬁ", "fi")
            .replace(" ", " "))


def _find_values(term, pages, max_hits=4):
    """[(value, page, context)] where `term` is followed by a number."""
    pat = re.compile(rf"(?<![\w.]){_term_pattern(term)}(?![\w]){_SEP}{_NUM}")
    hits = []
    for no, page in pages:
        for m in pat.finditer(page):
            v = _num(m.group(1))
            if v is None:
                continue
            a, b = max(0, m.start() - 50), min(len(page), m.end() + 40)
            hits.append({"value": v, "page": no,
                         "context": re.sub(r"\s+", " ", page[a:b]).strip()})
            if len(hits) >= max_hits:
                return hits
    return hits


def _mentions(term, pages):
    pat = re.compile(rf"(?<![\w]){re.escape(term)}(?![\w])", re.I if len(term) > 3 else 0)
    return sorted({no for no, page in pages if pat.search(page)})


def map_paper_to_model(pdf: str, model_name_or_path: str, resolve_proteins: bool = True) -> str:
    """Tool: the model's parameters and species as the paper states them."""
    from dynalyst.analysis.pdf_tools import _resolve_pdf_path
    from dynalyst.analysis.literature import model_references, uniprot_record
    from dynalyst.rag.ingest_pdf import extract_pages_text
    try:
        pdf_path = _resolve_pdf_path(pdf)
        pages = [(i + 1, _clean(t)) for i, t in enumerate(extract_pages_text(pdf_path))]
        refs = model_references(model_name_or_path)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    if sum(len(t) for _, t in pages) < 500:
        return json.dumps({"error": f"Almost no text could be read from {pdf_path.name} "
                                    "(scanned pages?), so nothing can be matched."})
    model = refs["model"]

    params = []
    for i in range(model.getNumParameters()):
        p = model.getParameter(i)
        params.append((p.getId(), p.getName(), p.getValue() if p.isSetValue() else None, None))
    for i in range(model.getNumReactions()):
        kl = model.getReaction(i).getKineticLaw()
        if kl is None:
            continue
        for lst in (kl.getListOfLocalParameters(), kl.getListOfParameters()):
            for p in lst:
                params.append((p.getId(), p.getName(), p.getValue() if p.isSetValue() else None,
                               model.getReaction(i).getId()))

    rows, differ = [], {}
    for pid, pname, val, reaction in params:
        hits = []
        for term in dict.fromkeys(t for t in (pid, pname) if t and len(t) >= 2):
            hits += _find_values(term, pages)
        status = "not stated in the paper"
        if hits:
            v = hits[0]["value"]
            if val is None:
                status = "paper value; model has none"
            elif abs(v - val) <= 1e-2 * max(abs(val), 1e-12):
                status = "same"
            else:
                status = f"differs (paper/model = {v / val:.3g})" if val else "differs"
                if reaction is None:
                    differ[pid] = v
        rows.append({"parameter": pid, **({"reaction": reaction} if reaction else {}),
                     "model_value": val, "paper": hits[:3], "status": status})

    proteins = {}
    for acc, sids in refs["proteins"].items():
        terms = [acc]
        if resolve_proteins:
            try:
                rec = uniprot_record(acc)
                terms += [rec["protein"]] + rec["genes"]
            except Exception:
                pass
        for s in sids:
            proteins.setdefault(s, []).extend(t for t in terms if t)
    species = []
    for i in range(model.getNumSpecies()):
        s = model.getSpecies(i)
        terms = [t for t in dict.fromkeys([s.getId(), s.getName(), *proteins.get(s.getId(), [])])
                 if t and len(t) >= 2]
        found = {t: _mentions(t, pages) for t in terms}
        found = {t: pg for t, pg in found.items() if pg}
        start = []
        for t in terms[:2]:
            start += _find_values(t, pages, max_hits=2)
        species.append({"species": s.getId(), "name": s.getName() or None,
                        "annotated_as": proteins.get(s.getId(), [])[1:3] or None,
                        "mentioned_as": found or None,
                        "values_next_to_it": start[:2] or None})

    stated = [r for r in rows if r["paper"]]
    res = {"pdf": pdf_path.name, "model": refs["path"].name,
           "summary": {"parameters": len(rows), "stated_in_paper": len(stated),
                       "same": sum(r["status"] == "same" for r in rows),
                       "different": sum(r["status"].startswith("differs") for r in rows),
                       "species_mentioned": sum(1 for s in species if s["mentioned_as"])},
           "parameters": stated + [{"parameter": r["parameter"], "status": r["status"]}
                                   for r in rows if not r["paper"]],
           "species": species,
           "method": "whole-word match of each model id / name (and a species' UniProt protein "
                     "and gene names) in the PDF text, with the number written right after it. "
                     "Units are not converted; check each context."}
    if differ:
        res["apply_with"] = {"tool": "save_sbml", "parameters": differ,
                             "note": "only if the contexts confirm these are the same quantities "
                                     "in the same units"}
    return json.dumps(res, indent=2, default=float)


TOOL_DEFINITIONS = [
    {"name": "map_paper_to_model",
     "description": "Map an uploaded paper (PDF) to a model: which of the model's parameters the "
                    "paper states and with what value (same / different, page and context), and "
                    "where its species / proteins are mentioned. The differing values can then be "
                    "applied with save_sbml.",
     "input_schema": {"type": "object", "properties": {
         "pdf": {"type": "string", "description": "PDF name or path (list_available_pdfs)."},
         "model_name_or_path": {"type": "string"},
         "resolve_proteins": {"type": "boolean", "description": "Also match species by their "
                                                                "UniProt protein / gene names."}},
         "required": ["pdf", "model_name_or_path"]}},
]
TOOL_FUNCTIONS = {"map_paper_to_model": map_paper_to_model}
