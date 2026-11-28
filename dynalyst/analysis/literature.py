"""PubMed and UniProt: the article behind a model and the proteins in it.

PubMed through NCBI E-utilities (esummary for the record, efetch for the
abstract); UniProt through rest.uniprot.org. Neither needs a key. Requests go
through biomodels._get, so a passing 5xx / dropped connection is retried.
model_proteins reads a model's own UniProt annotations (identifiers.org /
urn:miriam / purl.uniprot.org URIs) and resolves them.
"""
import json
import re
import xml.etree.ElementTree as ET

import requests

from dynalyst.analysis.biomodels import _get, search_biomodels

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
_PMID = re.compile(r"^\s*(?:PMID[:\s]*)?(\d{5,9})\s*$", re.I)
_DOI = re.compile(r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/\S+)\s*$", re.I)
# UniProt's own accession format
ACCESSION = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
                       r"(?:-\d+)?\b")
_UNIPROT_URI = re.compile(r"uniprot[/:]([A-Z0-9]{6,10})(?:-\d+)?", re.I)
_PUBMED_URI = re.compile(r"pubmed[/:](\d{5,9})", re.I)


def _err(msg):
    return json.dumps({"error": msg})


# -- PubMed -------------------------------------------------------------------

def _pmid_for(ref: str):
    """(PMID, None) for a PMID or DOI, else (None, error)."""
    m = _PMID.match(str(ref))
    if m:
        return m.group(1), None
    m = _DOI.match(str(ref))
    if not m:
        return None, f"'{ref}' is neither a PubMed ID (e.g. 15302873) nor a DOI (10.xxxx/...)."
    ids = _get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": f"{m.group(1)}[doi]",
                                         "retmode": "json"}, timeout=20)["esearchresult"]["idlist"]
    return (ids[0], None) if ids else (None, f"PubMed has no article with DOI {m.group(1)}.")


def _abstract(pmid: str) -> str:
    r = _get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": pmid, "rettype": "abstract",
                                       "retmode": "xml"}, timeout=20, as_json=False)
    root = ET.fromstring(r.content)
    parts = []
    for node in root.iter("AbstractText"):
        text = "".join(node.itertext()).strip()
        label = node.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(p for p in parts if p)


def pubmed_record(pmid: str, with_abstract: bool = True) -> dict:
    """The PubMed record as a dict (raises requests errors)."""
    rec = _get(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": pmid, "retmode": "json"},
               timeout=20)["result"].get(pmid)
    if not rec or rec.get("error"):
        raise KeyError(f"No PubMed record {pmid}.")
    doi = next((a["value"] for a in rec.get("articleids", []) if a.get("idtype") == "doi"), None)
    out = {"pmid": pmid, "title": (rec.get("title") or "").rstrip("."),
           "authors": [a["name"] for a in rec.get("authors", [])][:12],
           "journal": rec.get("fulljournalname") or rec.get("source"),
           "year": (rec.get("pubdate") or "")[:4] or None, "doi": doi,
           "links": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"]
                    + ([f"https://doi.org/{doi}"] if doi else [])}
    if with_abstract:
        try:
            out["abstract"] = _abstract(pmid) or "(no abstract in PubMed)"
        except (requests.RequestException, ET.ParseError):
            out["abstract"] = "(abstract unavailable: PubMed did not answer)"
    return out


def get_pubmed_article(pmid_or_doi: str) -> str:
    """Tool: a PubMed article (title, authors, journal, year, DOI, abstract)."""
    try:
        pmid, problem = _pmid_for(pmid_or_doi)
        if problem:
            return _err(problem)
        return json.dumps(pubmed_record(pmid), indent=2)
    except KeyError as e:
        return _err(str(e).strip("'\""))
    except (requests.RequestException, ValueError) as e:
        return _err(f"PubMed lookup failed ({type(e).__name__}: {e}); try again in a few minutes.")


def find_models_for_pubmed(pmid_or_doi: str, n_results: int = 10) -> str:
    """Tool: BioModels entries built from one article (by its PMID, else its DOI)."""
    try:
        pmid, problem = _pmid_for(pmid_or_doi)
    except (requests.RequestException, ValueError, KeyError) as e:
        return _err(f"PubMed lookup failed ({type(e).__name__}: {e}).")
    doi = _DOI.match(str(pmid_or_doi))
    tried, hits = [], []
    if pmid:
        tried.append(f"PubMed ID {pmid}")
        hits = json.loads(search_biomodels(f"publication_pubmed:{pmid}", n_results)).get("results", [])
    if not hits and doi:
        tried.append(f"DOI {doi.group(1)}")
        hits = json.loads(search_biomodels(f'publication_doi:"{doi.group(1)}"', n_results)).get(
            "results", [])
    if not pmid and not doi:
        return _err(problem)
    return json.dumps({"article": pmid_or_doi, "pmid": pmid, "tried": tried, "models": hits,
                       **({} if hits else {"note": "No BioModels entry cites this article."})},
                      indent=2)


# -- open-access full text ------------------------------------------------------
# PMC's open-access subset as served by its public bucket: no key and no browser check
# (the PMC and publisher pages refuse scripted downloads). Only open-access articles are there.
PMC_OA = "https://pmc-oa-opendata.s3.amazonaws.com"
_PMCID = re.compile(r"^\s*(?:PMCID[:\s]*)?(PMC\d{4,9})\s*$", re.I)
MAX_FILE_MB = 50


def _pmcid_for(ref: str):
    """(PMCID, None) for a PMCID, PMID or DOI, else (None, why)."""
    m = _PMCID.match(str(ref))
    if m:
        return m.group(1).upper(), None
    pm, doi = _PMID.match(str(ref)), _DOI.match(str(ref))
    if not pm and not doi:
        return None, f"'{ref}' is not a PMCID (PMC5874525), PubMed ID or DOI."
    q = f"EXT_ID:{pm.group(1)} AND SRC:MED" if pm else f'DOI:"{doi.group(1)}"'
    hits = _get(EUROPE_PMC, {"query": q, "format": "json", "resultType": "lite"},
                timeout=20).get("resultList", {}).get("result", [])
    pmcid = next((h["pmcid"] for h in hits if h.get("pmcid")), None)
    return (pmcid, None) if pmcid else (None, f"{ref} has no PubMed Central copy, so no open-access "
                                              "full text: only its abstract (get_pubmed_article).")


def _oa_record(pmcid: str):
    """The bucket's metadata of the newest version of an article, or None."""
    rec = None
    for v in range(1, 10):
        r = requests.get(f"{PMC_OA}/{pmcid}.{v}/{pmcid}.{v}.json", timeout=20)
        if r.status_code != 200:
            break
        rec = r.json()
    return rec


def fetch_open_access_paper(pmid_or_doi: str, supplements: bool = True) -> str:
    """Tool: download an open-access article's PDF and its supplementary files (PubMed
    Central's open-access subset) into data/papers/, to read with read_pdf."""
    from dynalyst import config
    try:
        pmcid, problem = _pmcid_for(pmid_or_doi)
        if problem:
            return _err(problem)
        rec = _oa_record(pmcid)
        if not rec:
            return _err(f"{pmcid} is in PubMed Central but not in its open-access subset, so it "
                        "cannot be downloaded: only its abstract (get_pubmed_article).")
        urls = [("article", rec["pdf_url"])] if rec.get("pdf_url") else []
        if supplements:   # figures are images inside the article PDF already
            urls += [("supplement", u) for u in rec.get("media_urls") or []
                     if not re.search(r"\.(jpe?g|gif|png|tiff?)\?", u + "?", re.I)]
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        saved, skipped = [], []
        for kind, url in urls:
            key = url.split("?")[0].replace("s3://pmc-oa-opendata/", "")
            name = key.split("/")[-1]
            dest = config.PAPERS_DIR / (name if name.startswith(pmcid) else f"{pmcid}_{name}")
            if not dest.exists():
                r = requests.get(f"{PMC_OA}/{key}", timeout=60)
                r.raise_for_status()
                if len(r.content) > MAX_FILE_MB * 2 ** 20:
                    skipped.append(f"{name} (over {MAX_FILE_MB} MB)")
                    continue
                dest.write_bytes(r.content)
            saved.append({"kind": kind, "path": dest.relative_to(config.PROJECT_ROOT).as_posix(),
                          "readable_with_read_pdf": dest.suffix.lower() == ".pdf"})
    except (requests.RequestException, ValueError, KeyError) as e:
        return _err(f"Download failed ({type(e).__name__}: {e}); try again in a few minutes.")
    # the files are named by PMCID only: the index lets list_available_pdfs say which paper
    index = config.PAPERS_DIR / "index.json"
    try:
        known = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    except ValueError:
        known = {}
    known[pmcid] = {"title": rec.get("title"), "doi": rec.get("doi"), "pmid": rec.get("pmid")}
    index.write_text(json.dumps(known, indent=2), encoding="utf-8")
    return json.dumps({"pmcid": pmcid, "pmid": rec.get("pmid"), "doi": rec.get("doi"),
                       "title": rec.get("title"), "license": rec.get("license_code"),
                       "files": saved, **({"skipped": skipped} if skipped else {}),
                       "note": "Read each PDF with read_pdf(path). Model equations and parameter "
                               "values are often only in a supplement."}, indent=2)


EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


# words that carry no topic: a request's own phrasing, not what a paper is about
_NOT_TOPIC = {"a", "an", "and", "about", "any", "are", "by", "few", "find", "for", "from", "get",
              "give", "in", "is", "key", "latest", "look", "me", "new", "of", "on", "or",
              "paper", "papers", "please", "publication", "publications", "recent", "research",
              "show", "some", "studies", "study", "the", "to", "up", "what", "with", "work"}


_SYNTAX = re.compile(r"\b[A-Z_]{2,}:|\b(?:AND|OR|NOT)\b|[()\[\]\"]")


def _plain_query(query: str) -> str:
    """A query written in search syntax (field tags, AND/OR, brackets, date ranges, years
    listed) reduced to its topic words: this search finds the topic itself and ranks by
    title; 'recent' is the sort option, not years in the query. A plain query (an author
    and a year included) is kept as it is."""
    q = query or ""
    if not _SYNTAX.search(q):
        return q
    q = re.sub(r"\[[^\]]*\]", " ", q)                        # date ranges
    q = re.sub(r"\b[A-Z_]{2,}:", " ", q)                       # field tags
    q = re.sub(r"\b(?:AND|OR|NOT)\b|[()\"]", " ", q)
    q = re.sub(r"\b(?:19|20)\d{2}\b", " ", q)                 # years listed as words
    return re.sub(r"\s+", " ", q).strip()


def search_literature(query: str, n_results: int = 10, open_access_only: bool = False,
                      sort: str = "cited") -> str:
    """Tool: search papers by topic / author / title in Europe PMC (covers PubMed). The
    topic words must all appear in a paper's title or abstract (a plain full-text search
    returns papers that merely mention a word); when nothing matches that way (e.g. an
    author and year), the plain search is used. sort: 'cited' (most cited first) or
    'recent' (newest first)."""
    query = _plain_query(query)
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query or "")
             if w.lower() not in _NOT_TOPIC]
    oa = " AND OPEN_ACCESS:y" if open_access_only else ""
    order = {"cited": "CITED desc", "recent": "P_PDATE_D desc"}.get(str(sort).lower(), "CITED desc")
    page = max(1, min(int(n_results), 50))

    def fetch(q, size, how=order):
        return _get(EUROPE_PMC, {"query": q, "format": "json", "resultType": "lite",
                                 "sort": how, "pageSize": size}, timeout=20)

    d = {}
    try:
        # an author and a year ("Smith 2004", "Smith et al. 2004", "Smith's 2004"): that
        # author's papers of that year, ranked by how many topic words their titles share
        for author, year in re.findall(r"\b([A-Z][A-Za-z\-]+)(?:['\u2019]s)?\s+(?:et\s+al\.?\s+)?"
                                       r"((?:19|20)\d{2})\b", query or ""):
            topic = {t.lower() for w in words if w not in (author, year)
                     for t in w.split("-") if len(t) > 2}
            # "Smith-Jones": both authors
            authors = [x for x in author.split("-") if x]
            found = {}
            for join in (" AND ", " OR "):        # both named authors first, then either
                base = f"({join.join(f'AUTH:{chr(34)}{x}{chr(34)}' for x in authors)}) AND PUB_YEAR:{year}"
                for q in ([f"{base} AND TITLE_ABS:({' OR '.join(sorted(topic))}){oa}"] if topic else [])                         + [f"{base}{oa}"]:
                    found = fetch(q, 50, "CITED desc")
                    if found.get("resultList", {}).get("result"):
                        break
                if found.get("resultList", {}).get("result") or len(authors) == 1:
                    break
            hits = found.get("resultList", {}).get("result", [])
            if hits:
                share = lambda r: len(topic & set(re.findall(r"[a-z0-9]+", (r.get("title") or "").lower())))
                hits = sorted(hits, key=share, reverse=True)
                # the author and year match: kept, best topic match first
                d = {"hitCount": found.get("hitCount"), "resultList": {"result": hits[:page]}}
                break
        if not d:
            # a topic: words that stand next to each other in the request as phrases
            # ("cell cycle"), then all words in the title, then in the title or abstract, then the
            # plain search; the first giving enough papers, else the first giving any
            runs = [r.split() for r in re.split(r"\b(?:%s)\b|[^A-Za-z0-9\- ]" % "|".join(_NOT_TOPIC),
                                                 query or "", flags=re.I) if r.split()]
            phrases = [f'"{" ".join(r)}"' if len(r) > 1 else r[0] for r in runs]
            queries = ([f"TITLE_ABS:({' AND '.join(phrases)}){oa}"] if len(phrases) < len(words) else []) \
                + ([f"TITLE:({' AND '.join(words)}){oa}", f"TITLE_ABS:({' AND '.join(words)}){oa}"]
                   if words else []) + [f"({query}){oa}"]
            first = {}
            for q in queries:
                d = fetch(q, max(page, 50))
                n = len(d.get("resultList", {}).get("result") or [])
                if n and not first:
                    first = d
                if n >= page:
                    break
            else:
                d = first or d
            # papers whose title carries more of the topic terms first ("oscill" matches
            # oscillation and oscillatory); the requested order is kept among equals
            terms = ([p.strip('"').lower() for p in phrases if " " in p]
                     + [w.lower()[:6] for w in words if len(w) > 2])
            hits = d.get("resultList", {}).get("result") or []
            latin = str.maketrans("αβγδκ", "abgdk")   # Greek letters as written in Latin
            in_title = lambda r: sum(t in (r.get("title") or "").lower().translate(latin)
                                     for t in terms)
            hits = sorted(hits, key=in_title, reverse=True)[:page]
            if order.startswith("P_PDATE"):    # the best matches, shown newest first as asked
                hits = sorted(hits, key=lambda r: r.get("firstPublicationDate")
                              or str(r.get("pubYear") or ""), reverse=True)
            d.setdefault("resultList", {})["result"] = hits
    except (requests.RequestException, ValueError) as e:
        d = {"error": f"Europe PMC search failed ({type(e).__name__}: {e})"}
    rows = []
    for r in d.get("resultList", {}).get("result", []):
        rows.append({"title": (r.get("title") or "").rstrip("."), "authors": r.get("authorString"),
                     "journal": r.get("journalTitle"), "year": r.get("pubYear"),
                     "pmid": r.get("pmid"), "pmcid": r.get("pmcid"), "doi": r.get("doi"),
                     "open_access": r.get("isOpenAccess") == "Y", "source": "Europe PMC",
                     "link": (f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" if r.get("pmid")
                              else f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}")})
    # Europe PMC indexes only part of some journals (Current Opinion in Systems Biology, ...):
    # when no paper it found carries the request's words in its title, Crossref (every DOI)
    # is asked too, and its close title matches come first
    out = {"query": query, "total_hits": d.get("hitCount"), "results": rows}
    if not rows or max(_title_match(r["title"], words) for r in rows) < 0.8:
        extra = _crossref(query, page)
        if extra:
            close = [r for r in extra if _title_match(r["title"], words) >= 0.6]
            known = {(r.get("doi") or "").lower() for r in rows}
            new = [r for r in (close or (extra if not rows else [])) if r["doi"].lower() not in known]
            if new:
                out["results"] = (new + rows)[:page]
                out["crossref_note"] = ("Europe PMC has no paper whose title matches the request; the "
                                        "Crossref entries are DOI records: no abstract, no open-access flag.")
    if not out["results"] and d.get("error"):
        return _err(d["error"] + "; try again in a few minutes.")
    return json.dumps(out, indent=2)


def _title_match(title: str, words: list) -> float:
    """The share of the request's topic words (3+ letters) that a title contains."""
    terms = {w.lower() for w in words if len(w) > 2}
    if not terms:
        return 0.0
    have = set(re.findall(r"[a-z0-9]+", (title or "").lower().replace("-", " ")))
    return sum(any(h.startswith(t[:6]) for h in have) for t in terms) / len(terms)


def _crossref(query: str, rows: int) -> list:
    """Papers from Crossref's bibliographic search, as search_literature rows ([] when it fails)."""
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query.bibliographic": query, "rows": max(1, min(rows, 20)),
                                 "select": "DOI,title,author,container-title,issued,type"},
                         headers={"User-Agent": "dynalyst-literature-search/1.0"}, timeout=20)
        r.raise_for_status()
        items = r.json()["message"]["items"]
    except (requests.RequestException, ValueError, KeyError):
        return []
    out = []
    for it in items:
        title = ((it.get("title") or [""])[0] or "").strip().rstrip(".")
        if not title or not it.get("DOI"):
            continue
        authors = ", ".join(" ".join(x for x in (a.get("family"), (a.get("given") or "")[:1]) if x)
                            for a in (it.get("author") or [])[:6])
        year = ((it.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        out.append({"title": title, "authors": authors or None,
                    "journal": (it.get("container-title") or [None])[0],
                    "year": str(year) if year else None, "pmid": None, "pmcid": None,
                    "doi": it["DOI"], "open_access": None, "source": "Crossref",
                    "link": f"https://doi.org/{it['DOI']}"})
    return out


# -- UniProt ------------------------------------------------------------------

def _protein_name(d):
    desc = d.get("proteinDescription", {})
    for key in ("recommendedName", "submissionNames"):
        v = desc.get(key)
        v = v[0] if isinstance(v, list) and v else v
        if v and v.get("fullName", {}).get("value"):
            return v["fullName"]["value"]
    return None


def uniprot_record(accession: str) -> dict:
    d = _get(f"{UNIPROT}/{accession}.json", timeout=20)
    function = next((t.get("value") for c in d.get("comments", [])
                     if c.get("commentType") == "FUNCTION" for t in c.get("texts", [])), None)
    genes = [g["geneName"]["value"] for g in d.get("genes", []) if g.get("geneName")]
    return {"accession": d.get("primaryAccession", accession), "protein": _protein_name(d),
            "genes": genes, "organism": d.get("organism", {}).get("scientificName"),
            "reviewed": "Swiss-Prot" in (d.get("entryType") or ""),
            "length": d.get("sequence", {}).get("length"),
            "function": (function[:1200] + "...") if function and len(function) > 1200 else function,
            "page": f"https://www.uniprot.org/uniprotkb/{d.get('primaryAccession', accession)}"}


def get_uniprot_entry(accession: str) -> str:
    """Tool: one UniProt entry (protein, gene, organism, function, length)."""
    acc = str(accession).strip().upper()
    if not ACCESSION.fullmatch(acc.split("-")[0]):
        return _err(f"'{accession}' is not a UniProt accession (e.g. P28482); search_uniprot "
                    "finds accessions by protein or gene name.")
    try:
        return json.dumps(uniprot_record(acc.split("-")[0]), indent=2)
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) in (400, 404):
            return _err(f"No UniProt entry {acc}.")
        return _err(f"UniProt lookup failed ({e}); try again in a few minutes.")
    except (requests.RequestException, ValueError) as e:
        return _err(f"UniProt lookup failed ({type(e).__name__}: {e}).")


def search_uniprot(query: str, organism: str | None = None, n_results: int = 5) -> str:
    """Tool: UniProt entries for a protein / gene name, reviewed (Swiss-Prot) first."""
    q = f"({query})" + (f' AND organism_name:"{organism}"' if organism else "")
    try:
        d = _get(f"{UNIPROT}/search", {"query": q, "size": max(1, min(int(n_results), 25)) * 2,
                                       "format": "json"}, timeout=20)
    except (requests.RequestException, ValueError) as e:
        return _err(f"UniProt search failed ({type(e).__name__}: {e}).")
    rows = [{"accession": r.get("primaryAccession"), "protein": _protein_name(r),
             "genes": [g["geneName"]["value"] for g in r.get("genes", []) if g.get("geneName")],
             "organism": r.get("organism", {}).get("scientificName"),
             "reviewed": "Swiss-Prot" in (r.get("entryType") or "")}
            for r in d.get("results", [])]
    rows.sort(key=lambda r: not r["reviewed"])
    return json.dumps({"query": query, "organism": organism, "results": rows[:n_results]}, indent=2)


# -- a model's own annotations ---------------------------------------------------

def model_references(model_name_or_path: str) -> dict:
    """{'proteins': {accession: [species ids]}, 'pmids': [...], 'model': SBML model}
    from the model's MIRIAM annotations (no network)."""
    import libsbml
    from dynalyst.analysis.annotations import _cv_terms, resolve_sbml_path
    path = resolve_sbml_path(str(model_name_or_path))
    model = libsbml.readSBMLFromFile(str(path)).getModel()
    if model is None:
        raise ValueError(f"{path.name} is not readable SBML.")
    proteins, pmids = {}, []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        for term in _cv_terms(sp):
            for uri in term["resources"]:
                m = _UNIPROT_URI.search(uri)
                if m:
                    proteins.setdefault(m.group(1).upper(), []).append(sp.getId())
    for term in _cv_terms(model):
        if term["qualifier"] in ("bqmodel:isDescribedBy", "bqmodel:is"):
            pmids += [m.group(1) for uri in term["resources"] for m in _PUBMED_URI.finditer(uri)]
    return {"proteins": proteins, "pmids": list(dict.fromkeys(pmids)), "model": model, "path": path}


def model_proteins(model_name_or_path: str, max_proteins: int = 30) -> str:
    """Tool: the proteins a model's species are annotated with, resolved in UniProt."""
    try:
        refs = model_references(model_name_or_path)
    except (FileNotFoundError, ValueError) as e:
        return _err(str(e))
    model = refs["model"]
    names = {model.getSpecies(i).getId(): model.getSpecies(i).getName()
             for i in range(model.getNumSpecies())}
    if not refs["proteins"]:
        return json.dumps({"model": refs["path"].name, "proteins": [],
                           "note": "The model's species carry no UniProt annotation; "
                                   "search_uniprot can look a species' protein up by name."})
    out, failed = [], []
    for acc, species in list(refs["proteins"].items())[:max_proteins]:
        try:
            rec = uniprot_record(acc)
        except (requests.RequestException, ValueError):
            failed.append(acc)
            rec = {"accession": acc, "protein": None}
        rec["species"] = [f"{s} ({names[s]})" if names.get(s) and names[s] != s else s for s in species]
        out.append(rec)
    res = {"model": refs["path"].name, "proteins": out, "article_pmids": refs["pmids"]}
    if failed:
        res["unresolved"] = f"UniProt did not answer for {failed}; their accessions are listed."
    if len(refs["proteins"]) > max_proteins:
        res["note"] = f"First {max_proteins} of {len(refs['proteins'])} annotated proteins."
    return json.dumps(res, indent=2)


TOOL_DEFINITIONS = [
    {"name": "search_literature",
     "description": "Search published papers by topic, author or title (Europe PMC, which "
                    "includes PubMed): title, authors, journal, year, PMID, DOI, open access.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": (
             "Plain topic, author or title words, e.g. 'hysteresis cell cycle' or 'Smith 2004'. "
             "No AND/OR, field tags or year lists: use sort='recent' for recent work.")},
         "n_results": {"type": "integer"},
         "open_access_only": {"type": "boolean"},
         "sort": {"type": "string", "enum": ["cited", "recent"],
                  "description": "'cited' (default): most cited first; 'recent': newest first."}},
         "required": ["query"]}},
    {"name": "get_pubmed_article",
     "description": "A PubMed article by PMID or DOI: title, authors, journal, year, DOI and "
                    "abstract (e.g. the paper a model comes from).",
     "input_schema": {"type": "object", "properties": {
         "pmid_or_doi": {"type": "string", "description": "PubMed ID (15302873) or DOI."}},
         "required": ["pmid_or_doi"]}},
    {"name": "fetch_open_access_paper",
     "description": "Download an open-access paper's full PDF and its supplementary files "
                    "(PubMed Central open-access subset) into data/papers/; then read them with "
                    "read_pdf. Use when the full text, model equations or parameter values are "
                    "needed (often only in a supplement). Paywalled papers cannot be downloaded.",
     "input_schema": {"type": "object", "properties": {
         "pmid_or_doi": {"type": "string", "description": "PMCID (PMC5874525), PubMed ID or DOI."},
         "supplements": {"type": "boolean", "description": "Also the supplementary files "
                                                           "(default true)."}},
         "required": ["pmid_or_doi"]}},
    {"name": "find_models_for_pubmed",
     "description": "The BioModels entries built from one article, given its PubMed ID or DOI.",
     "input_schema": {"type": "object", "properties": {
         "pmid_or_doi": {"type": "string"}, "n_results": {"type": "integer"}},
         "required": ["pmid_or_doi"]}},
    {"name": "get_uniprot_entry",
     "description": "One UniProt protein entry by accession (e.g. P28482): protein name, genes, "
                    "organism, function, sequence length.",
     "input_schema": {"type": "object", "properties": {"accession": {"type": "string"}},
                      "required": ["accession"]}},
    {"name": "search_uniprot",
     "description": "Search UniProt by protein or gene name (optionally one organism), "
                    "reviewed Swiss-Prot entries first.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "organism": {"type": "string",
                                                   "description": "e.g. 'Homo sapiens'."},
         "n_results": {"type": "integer"}}, "required": ["query"]}},
    {"name": "model_proteins",
     "description": "Which proteins a model's species are (from the model's UniProt "
                    "annotations, resolved in UniProt), plus the PubMed IDs the model cites.",
     "input_schema": {"type": "object", "properties": {
         "model_name_or_path": {"type": "string"}}, "required": ["model_name_or_path"]}},
]
TOOL_FUNCTIONS = {"search_literature": search_literature, "get_pubmed_article": get_pubmed_article,
                  "fetch_open_access_paper": fetch_open_access_paper,
                  "find_models_for_pubmed": find_models_for_pubmed,
                  "get_uniprot_entry": get_uniprot_entry, "search_uniprot": search_uniprot,
                  "model_proteins": model_proteins}
