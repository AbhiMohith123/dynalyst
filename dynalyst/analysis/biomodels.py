"""BioModels: search, model information, download, and the models behind an
uploaded article.

Search and model information use EBI Search's BioModels index
(www.ebi.ac.uk/ebisearch, fields such as name, description, curation status
and the publication's title / authors / year / DOI / PubMed ID). Files are
downloaded through the BioModels REST API (``DYNALYST_BIOMODELS_API``,
default https://www.ebi.ac.uk/biomodels, which redirects to biomodels.org)
into data/sbml_models/, where every other tool finds them by name.
"""
import json
import os
import re
from pathlib import Path

import requests

from dynalyst import config

EBI_SEARCH = "https://www.ebi.ac.uk/ebisearch/ws/rest/biomodels"
BIOMODELS_API = os.environ.get("DYNALYST_BIOMODELS_API", "https://www.ebi.ac.uk/biomodels")
TIMEOUT = float(os.environ.get("DYNALYST_BIOMODELS_TIMEOUT", "30"))
FIELDS = ["name", "curationstatus", "modelformat", "publication_title", "publication_year",
          "first_author", "publication_pubmed", "publication_doi", "last_modification_date"]
_ID = re.compile(r"^(BIOMD|MODEL)\d{10}$")


RETRIES = int(os.environ.get("DYNALYST_BIOMODELS_RETRIES", "3"))


def _get(url, params=None, timeout=TIMEOUT, as_json=True):
    """GET with retries: BioModels / EBI answer 5xx, 429 or drop the connection now
    and then for a few seconds; a 4xx (bad ID, missing file) is final at once."""
    import time
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"Accept": "application/json" if as_json else "*/*"})
            r.raise_for_status()
            return r.json() if as_json else r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status is not None and status < 500 and status != 429) or attempt == RETRIES - 1:
                raise
            time.sleep(2 * 2 ** attempt)     # 2 s, 4 s, ...


def _one(fields, key):
    v = fields.get(key) or []
    return v[0] if v else None


def _summary(entry):
    f = entry.get("fields", {})
    pmid, doi = _one(f, "publication_pubmed"), _one(f, "publication_doi")
    return {
        "id": entry["id"], "name": _one(f, "name"),
        "curation": _one(f, "curationstatus"), "format": _one(f, "modelformat"),
        "article": {"title": _one(f, "publication_title"), "first_author": _one(f, "first_author"),
                    "year": _one(f, "publication_year"), "pubmed": pmid, "doi": doi},
        "page": f"https://www.ebi.ac.uk/biomodels/{entry['id']}",
    }


def search_biomodels(query: str, n_results: int = 10, curated_only: bool = False) -> str:
    """Tool: search BioModels (names, descriptions, articles, annotations)."""
    q = f"({query}) AND curationstatus:\"Manually curated\"" if curated_only else query
    try:
        d = _get(EBI_SEARCH, {"query": q, "size": max(1, min(int(n_results), 50)),
                              "format": "json", "fields": ",".join(FIELDS)})
    except requests.RequestException as e:
        return json.dumps({"error": f"BioModels search failed ({type(e).__name__}: {e})."})
    hits = [_summary(e) for e in d.get("entries", [])]
    return json.dumps({"query": query, "total_hits": d.get("hitCount", len(hits)),
                       "results": hits}, indent=2)


def get_biomodels_info(model_id: str) -> str:
    """Tool: a BioModels entry's description, article and files."""
    model_id = model_id.strip()
    if not _ID.match(model_id):
        return json.dumps({"error": f"'{model_id}' is not a BioModels ID (BIOMD0000000027 or "
                                    "MODEL1234567890); search_biomodels finds IDs."})
    try:
        d = _get(f"{EBI_SEARCH}/entry/{model_id}",
                 {"format": "json", "fields": ",".join(FIELDS + ["description", "submitter",
                                                                 "submissionid", "levelversion"])})
    except requests.RequestException as e:
        return json.dumps({"error": f"BioModels lookup failed ({type(e).__name__}: {e})."})
    if not d.get("entries"):
        return json.dumps({"error": f"No BioModels entry {model_id}."})
    entry = d["entries"][0]
    out = _summary(entry)
    f = entry.get("fields", {})
    desc = re.sub(r"\s+", " ", _one(f, "description") or "").strip()
    out.update(description=desc[:2500] + ("..." if len(desc) > 2500 else ""),
               submitter=_one(f, "submitter"), sbml_level_version=_one(f, "levelversion"),
               submission_id=_one(f, "submissionid"))
    art = out["article"]
    out["article_links"] = [u for u in (
        f"https://pubmed.ncbi.nlm.nih.gov/{art['pubmed']}/" if art["pubmed"] else None,
        f"https://doi.org/{art['doi']}" if art["doi"] else None) if u]
    try:
        files = _get(f"{BIOMODELS_API}/model/files/{model_id}", {"format": "json"}, timeout=15)
        out["files"] = {"main": [x.get("name") for x in files.get("main", [])],
                        "additional": [x.get("name") for x in files.get("additional", [])][:20]}
    except requests.RequestException:
        out["files"] = "file list unavailable (BioModels file server did not answer)"
    return json.dumps(out, indent=2)


def fetch_biomodels_model(model_id: str, filename: str | None = None) -> str:
    """Tool: download a BioModels entry's SBML into data/sbml_models/."""
    import libsbml
    model_id = model_id.strip()
    filename = (filename or "").strip() or None      # an empty name means the main file
    if not _ID.match(model_id):
        return json.dumps({"error": f"'{model_id}' is not a BioModels ID."})
    manual = (f"Download it from https://www.ebi.ac.uk/biomodels/{model_id} and upload the SBML "
              "file in the notebook instead.")
    path = config.LOCAL_SBML_MODELS_DIR / f"{model_id}.xml"
    if filename is None and path.exists() and libsbml.readSBMLFromFile(str(path)).getModel():
        m = libsbml.readSBMLFromFile(str(path)).getModel()   # already downloaded: no server needed
        return json.dumps({
            "model_id": model_id, "saved_as": config.display_path(path), "name": m.getName() or m.getId(),
            "use_as": path.stem, "species": m.getNumSpecies(), "reactions": m.getNumReactions(),
            "parameters": _n_parameters(m), "rules": m.getNumRules(), "events": m.getNumEvents(),
            "note": f"Already downloaded; every tool finds it as '{path.stem}'.",
        }, indent=2)
    try:
        if filename is None:
            try:
                files = _get(f"{BIOMODELS_API}/model/files/{model_id}", {"format": "json"})
                sbml = [x["name"] for x in files.get("main", [])
                        if x.get("name", "").lower().endswith(".xml")]
                if not sbml:
                    return json.dumps({"error": f"{model_id} has no SBML main file (main files: "
                                                f"{[x.get('name') for x in files.get('main', [])]})."})
                filename = sbml[0]
            except requests.RequestException:
                filename = f"{model_id}_url.xml"   # file list down: the standard curated file name
        r = _get(f"{BIOMODELS_API}/model/download/{model_id}", {"filename": filename},
                 as_json=False, timeout=max(TIMEOUT, 60))
    except requests.RequestException as e:
        return json.dumps({"error": f"Could not download {model_id} from BioModels after {RETRIES} "
                                    f"tries ({type(e).__name__}: {e}). BioModels' server is having "
                                    f"trouble; try again in a few minutes. {manual}"})
    config.LOCAL_SBML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    doc = libsbml.readSBMLFromString(r.text)
    m = doc.getModel()
    if m is None or any(doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
                        for i in range(doc.getNumErrors())):
        return json.dumps({"error": f"The downloaded file {filename} is not readable SBML. {manual}"})
    import tempfile
    from pathlib import Path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(r.content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return json.dumps({
        "model_id": model_id, "saved_as": config.display_path(path), "name": m.getName() or m.getId(),
        "use_as": path.stem, "source_file": filename,
        "species": m.getNumSpecies(), "reactions": m.getNumReactions(),
        "parameters": _n_parameters(m), "rules": m.getNumRules(), "events": m.getNumEvents(),
        "note": f"Saved; every tool now finds it as '{path.stem}'.",
    }, indent=2)


# -- a paper name -> its BioModels entries ------------------------------------------
_ID_LOOSE = re.compile(r"\bBIOMD\s*[-_#]?\s*0*(\d{1,10})\b", re.I)
# non-curated IDs only in full (MODEL + 10 digits): a loose form would read 'model 2' or
# 'model1.xml' as an ID
_ID_MODEL = re.compile(r"\bMODEL(\d{10})\b", re.I)
_ID_WORDS = re.compile(r"\bbio\s*-?\s*models?\s*(?:id|entry|no\.?|number|#)?\s*:?\s*0*(\d{1,10})\b",
                       re.I)


def biomodels_ids_in(text: str) -> set:
    """Every BioModels ID written in a text, normalised: 'BIOMD0000000012', 'BIOMD12',
    'biomodel 12', 'BioModels #12' -> BIOMD0000000012; MODEL ids only as MODEL+10 digits."""
    out = set()
    for num in _ID_LOOSE.findall(text or ""):
        out.add(f"BIOMD{int(num):010d}")
    for num in _ID_MODEL.findall(text or ""):
        out.add(f"MODEL{num}")
    for num in _ID_WORDS.findall(text or ""):
        out.add(f"BIOMD{int(num):010d}")
    return out



def _paper_tokens(name: str):
    """'Smith2004', 'Smith et al. 2004', 'smith 2004 kinase' ->
    (['smith', 'kinase'], '2004')."""
    s = re.sub(r"([A-Za-z])(\d{4})\b", r"\1 \2", name or "")
    s = re.sub(r"\bet\s*al\.?", " ", s, flags=re.I)
    year = next(iter(re.findall(r"\b(19\d{2}|20\d{2})\b", s)), None)
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", s)
             if w.lower() not in {"the", "model", "models", "paper", "of", "and", "biomodels",
                                  "biomodel", "fetch", "load", "download"}]
    return words, year


def _local_matches(words, year):
    """Model files already in data/sbml_models whose file or SBML name has the author
    (and year, when given)."""
    hits = []
    folder = config.LOCAL_SBML_MODELS_DIR
    if not words or not folder.exists():
        return hits
    author = words[0]
    for p in sorted(folder.glob("*.xml")):
        name = p.stem
        try:
            head = p.read_text(errors="replace")[:4000]
            m = re.search(r"<model[^>]*\bname=\"([^\"]+)\"", head)
            if m:
                name = f"{p.stem} ({m.group(1)})"
        except OSError:
            pass
        low = name.lower()
        if author in low and (year is None or year in low):
            hit = {"file": p.name, "use_as": p.stem, "sbml_name": name}
            try:      # the BioModels ID the file itself declares (bqmodel:is)
                from dynalyst.analysis.annotations import model_identity
                ident = model_identity(str(p)) or {}
                if ident.get("biomodels_id"):
                    hit["biomodels_id"] = ident["biomodels_id"]
            except Exception:
                pass
            hits.append(hit)
    return hits


def find_biomodels_by_paper(paper: str, n_results: int = 20) -> str:
    """Tool: the BioModels entries built from a paper named by author and year
    ('Smith 2004', 'Smith2004'), checked against each entry's article
    (first author, year) rather than trusting the search ranking."""
    words, year = _paper_tokens(paper)
    if not words:
        return json.dumps({"error": f"No author name in '{paper}'. Give it as 'Author Year', "
                                    "e.g. 'Smith 2004'."})
    # local first: a model file named exactly as the request ('Markevich2004') is the model
    # the user means; BioModels is searched only when there is none
    local = _local_matches(words, year)
    key = re.sub(r"[^a-z0-9]", "", paper.lower())
    exact = [m for m in local if re.sub(r"[^a-z0-9]", "", m["use_as"].lower()) == key]
    if exact:
        return json.dumps({
            "paper": paper, "local_file": exact[0]["file"], "use_as": exact[0]["use_as"],
            "other_local_files": [m["file"] for m in local if m is not exact[0]],
            "note": f"A local model file named {exact[0]['file']} exists: it is the model this "
                    "name refers to. Use it as it is; download a BioModels entry only if the "
                    "user asked for the BioModels version."}, indent=2)
    author = words[0]
    query = f"{author} {year}" if year else author
    try:
        d = _get(EBI_SEARCH, {"query": query, "size": max(1, min(int(n_results), 50)),
                              "format": "json", "fields": ",".join(FIELDS)})
    except requests.RequestException as e:
        return json.dumps({"error": f"BioModels search failed ({type(e).__name__}: {e}).",
                           "local_files": _local_matches(words, year)})
    hits = [_summary(e) for e in d.get("entries", [])]

    def matches(h):
        art = h["article"]
        first = (art.get("first_author") or "").lower()
        name = (h.get("name") or "").lower()
        by_author = author in first or name.startswith(author)
        by_year = year is None or str(art.get("year") or "") == year or year in name
        return by_author and by_year

    good = [h for h in hits if matches(h)]
    good.sort(key=lambda h: (h.get("curation") != "Manually curated", h["id"]))
    extra = [w for w in words[1:] if len(w) > 2]
    if extra:      # 'Smith 2004 <variant words>' -> the variant whose name has those words
        scored = sorted(good, key=lambda h: -sum(w in (h.get("name") or "").lower() for w in extra))
        if scored and sum(w in (scored[0].get("name") or "").lower() for w in extra):
            good = scored
    out = {"paper": paper, "author": author, "year": year, "query_used": query,
           "n_matching": len(good),
           "models": [{"id": h["id"], "name": h["name"], "curation": h["curation"],
                       "article": h["article"]} for h in good],
           "local_files": _local_matches(words, year)}
    if not good:
        out["note"] = (f"No BioModels entry whose article is by {author}"
                       + (f" ({year})" if year else "") + f". {len(hits)} search hit(s) were "
                       "about other papers and are not listed. Do not fetch any of them; ask the "
                       "user for the paper title, PubMed ID or DOI.")
    elif len(good) == 1:
        out["best"] = good[0]["id"]
        out["note"] = f"One entry: {good[0]['id']} ({good[0]['name']})."
    else:
        out["best"] = good[0]["id"]
        out["ambiguous"] = True
        out["note"] = (f"{len(good)} entries come from this paper (model variants): "
                       + "; ".join(f"{h['id']} {h['name']}" for h in good)
                       + f". Use {good[0]['id']} unless the user's words pick another variant, "
                       "and name the others in the answer.")
    return json.dumps(out, indent=2)


_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>\\]+[^\s\"<>\\.,;)\]])", re.I)
_PMID = re.compile(r"\bPMID[:\s]*(\d{6,9})\b", re.I)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def _pubmed_for_doi(doi):
    """(PMID, exact title) for a DOI via NCBI, only if NCBI's record carries
    that same DOI."""
    try:
        ids = _get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": f"{doi}[doi]",
                                              "retmode": "json"}, timeout=20)["esearchresult"]["idlist"]
        if not ids:
            return None, None
        rec = _get(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": ids[0], "retmode": "json"},
                   timeout=20)["result"][ids[0]]
    except (requests.RequestException, KeyError, ValueError):
        return None, None
    dois = [a["value"].lower() for a in rec.get("articleids", []) if a.get("idtype") == "doi"]
    return (ids[0], rec.get("title", "").rstrip(".")) if doi.lower() in dois else (None, None)


def _words(t):
    return set(re.findall(r"[a-z]{4,}", t.lower()))


_COMMON = {"with", "from", "that", "this", "their", "which", "these", "were", "have", "been",
           "into", "than", "also", "such", "between", "through", "using", "model", "models"}


def _norm(t):
    """Lowercase letters/digits only, line-break hyphenation joined."""
    t = re.sub(r"-\s*\n\s*", "", t.lower())
    return re.sub(r"[^a-z0-9]+", " ", t).split()


def _bigrams(words):
    return set(zip(words, words[1:]))


def _title_share(title, text_bigrams):
    """Share of the title's consecutive word pairs printed in the text: the
    title must appear as a phrase, not as scattered words."""
    b = _bigrams(_norm(title or ""))
    return (len(b & text_bigrams) / len(b)) if len(b) >= 3 else 0.0


def find_models_for_article(path_or_name: str, n_results: int = 5, min_share: float = 0.75) -> str:
    """Tool: BioModels entries built from an uploaded article.

    Every DOI / PubMed ID on the article's first pages is tried (a DOI is
    resolved to its PubMed record, since BioModels often stores only the
    PubMed ID); then a search on the page's distinctive words. A BioModels
    entry is accepted only if its article title is printed as a phrase on the
    first pages (at least `min_share` of the title's consecutive word pairs),
    so neither a cited paper's identifier nor scattered words can match."""
    from dynalyst.analysis.pdf_tools import read_pdf
    raw = read_pdf(path_or_name, page_range=[1, 2])
    try:
        data = json.loads(raw)
    except ValueError:
        data = raw
    if isinstance(data, dict) and data.get("error"):
        return json.dumps({"error": data["error"]})
    text = "\n".join(_strings(data.get("pages") if isinstance(data, dict) else data))
    # the PDF's own metadata title, when it is a real title (not a typesetting
    # label like "Fujita 1..10"), is the article's title: a match must agree
    # with it, which rules out papers the article merely cites
    meta = (data.get("title") or "").strip() if isinstance(data, dict) else ""
    meta_title = meta if len(_norm(meta)) >= 5 and not re.search(r"\d+\.\.\d+", meta) else None
    page_words = _norm(text)
    printed = _bigrams(page_words)
    readable = sum(1 for w in page_words if len(w) > 3) >= 80
    page = set(page_words)
    dois = list(dict.fromkeys(m.group(1) for m in _DOI.finditer(text)))[:6]
    pmids = list(dict.fromkeys(_PMID.findall(text)))[:3]
    queries = [(f"PubMed ID {p}", f"publication_pubmed:{p}") for p in pmids]
    for doi in dois:
        queries.append((f"DOI {doi}", f'publication_doi:"{doi}"'))
        pmid, _ = _pubmed_for_doi(doi)
        if pmid:
            queries.append((f"PubMed ID {pmid} (from DOI {doi})", f"publication_pubmed:{pmid}"))
    if meta_title:
        queries.insert(0, ("title in the PDF's metadata", f'publication_title:"{meta_title}"'))
        printed = _bigrams(_norm(meta_title))
    distinctive = sorted((w for w in page if len(w) >= 7 and w not in _COMMON),
                         key=len, reverse=True)[:40]
    if distinctive:
        queries.append(("title printed in the article",
                        "publication_title:(" + " OR ".join(distinctive) + ")"))
    tried = []
    for label, query in queries:
        tried.append(label)
        res = json.loads(search_biomodels(query, 50))
        scored = [(_title_share(h["article"]["title"], printed), h) for h in res.get("results", [])]
        hits = [h for sh, h in sorted(scored, key=lambda x: -x[0]) if sh >= min_share]
        if hits:
            return json.dumps({"article": path_or_name, "matched_by": label,
                               "models": hits[:n_results]}, indent=2)
    note = ("No BioModels entry whose article title appears in this PDF." if readable else
            "The PDF's text could not be read (scanned pages or encoded fonts), so no match can "
            "be confirmed. Search BioModels by the article's title or authors instead.")
    return json.dumps({"article": path_or_name, "dois_found": dois, "tried": tried, "models": [],
                       "note": note}, indent=2)


_SCHEMA_ID = {"model_id": {"type": "string", "description": (
    "BioModels ID: BIOMD and the entry number in 10 digits. An entry the user names by its "
    "number ('biomodel 12', 'BioModels entry 12') is BIOMD0000000012.")}}
TOOL_DEFINITIONS = [
    {"name": "search_biomodels",
     "description": "Search the BioModels database (model names, descriptions, articles, "
                    "annotations) and list matching models with their article.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "n_results": {"type": "integer", "description": "Default 10, max 50."},
         "curated_only": {"type": "boolean", "description": "Only manually curated models."}},
         "required": ["query"]}},
    {"name": "get_biomodels_info",
     "description": "Information on one BioModels entry: description, curation, article "
                    "(title, authors, year, PubMed / DOI links) and files.",
     "input_schema": {"type": "object", "properties": _SCHEMA_ID, "required": ["model_id"]}},
    {"name": "fetch_biomodels_model",
     "description": "Download a BioModels entry's SBML into the local model folder, so every "
                    "other tool (simulation, bifurcation, CRNT...) can use it by its ID.",
     "input_schema": {"type": "object", "properties": {
         **_SCHEMA_ID, "filename": {"type": "string", "description": "Specific file (default: "
                                                                     "the main SBML file)."}},
         "required": ["model_id"]}},
    {"name": "find_biomodels_by_paper",
     "description": "The BioModels entries of a paper named by author and year ('Smith 2004', "
                    "'Smith2004', 'Smith 2004 <variant words>'), checked against each "
                    "entry's article; lists all variants, picks a default ('best'), and lists "
                    "matching local model files. Use this BEFORE fetching whenever the user names "
                    "a paper or model rather than giving a BioModels ID.",
     "input_schema": {"type": "object", "properties": {
         "paper": {"type": "string", "description": "Author and year, optionally variant words."},
         "n_results": {"type": "integer"}}, "required": ["paper"]}},
    {"name": "find_models_for_article",
     "description": "Find the BioModels entries built from an uploaded article (PDF in "
                    "data/papers), by the article's DOI, PubMed ID or title.",
     "input_schema": {"type": "object", "properties": {
         "path_or_name": {"type": "string", "description": "PDF name or path."},
         "n_results": {"type": "integer"}}, "required": ["path_or_name"]}},
]
TOOL_FUNCTIONS = {"search_biomodels": search_biomodels, "get_biomodels_info": get_biomodels_info,
                  "fetch_biomodels_model": fetch_biomodels_model,
                  "find_biomodels_by_paper": find_biomodels_by_paper,
                  "find_models_for_article": find_models_for_article}


def _n_parameters(m) -> int:
    """Global parameters plus the local ones of every reaction (SBML keeps both)."""
    local = 0
    for i in range(m.getNumReactions()):
        law = m.getReaction(i).getKineticLaw()
        if law is not None:
            local += law.getNumLocalParameters() or law.getNumParameters()
    return m.getNumParameters() + local
