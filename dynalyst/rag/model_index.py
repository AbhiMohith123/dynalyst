"""The knowledge base learns every model the user saves.

index_model(model) turns a model into retrieval chunks -- what it is (name, file,
BioModels ID), its description (SBML notes), its species with the proteins their
UniProt annotations name, its parameters, and its article (PubMed title,
authors, abstract) -- and adds them to the RAG at once:
  * data/chunks_models.jsonl  (always; a full rebuild with build_index.py
    --chunks ... data/chunks_models.jsonl keeps them),
  * the Chroma collection config.COLLECTION_MODELS and the BM25 index (live,
    when the RAG stack is installed; a model indexed again replaces its chunks).
Network lookups (PubMed / UniProt) are best effort: without them the chunks
still hold everything the file itself says.
"""
import json
import hashlib
import pickle
import re

from dynalyst import config

CHUNKS_FILE = config.DATA_DIR / "chunks_models.jsonl"
PAPER_CHUNKS_FILE = config.DATA_DIR / "chunks_uploaded_papers.jsonl"   # PDFs indexed live
_WORDS_PER_CHUNK = 300


def _pieces(text, n=_WORDS_PER_CHUNK):
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)] or [""]


def model_chunks(model_name_or_path: str, online: bool = True) -> list[dict]:
    """Retrieval records {id, text, metadata} describing one SBML model."""
    from dynalyst.analysis.annotations import _notes_text, model_identity
    from dynalyst.analysis.literature import model_references, pubmed_record, uniprot_record
    refs = model_references(model_name_or_path)
    model, path = refs["model"], refs["path"]
    ident = model_identity(str(path)) or {}
    name = model.getName() or model.getId()
    head = f"Model {name} (file {path.name}" + (
        f", BioModels {ident['biomodels_id']})" if ident.get("biomodels_id") else ")")
    meta = {"source_type": "model", "source_name": path.name, "model_name": name,
            "source_id": hashlib.sha256(str(path.resolve()).encode()).hexdigest(),
            "source_path": str(path.resolve()),
            "biomodels_id": ident.get("biomodels_id") or ""}

    proteins = {}
    if online:
        for acc in list(refs["proteins"])[:30]:
            try:
                rec = uniprot_record(acc)
                proteins[acc] = f"{rec['protein']} ({', '.join(rec['genes'][:2]) or 'no gene'}; " \
                                f"{rec['organism']}; UniProt {acc})"
            except Exception:
                proteins[acc] = f"UniProt {acc}"
    by_species = {s: proteins.get(a, f"UniProt {a}") for a, ss in refs["proteins"].items() for s in ss}
    species = []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        label = sp.getId() + (f" ({sp.getName()})" if sp.getName() and sp.getName() != sp.getId() else "")
        species.append(label + (f" = {by_species[sp.getId()]}" if sp.getId() in by_species else ""))
    params = [f"{p.getId()}={p.getValue():g}" for p in model.getListOfParameters() if p.isSetValue()]

    sections = [("overview",
                 f"{head}. {model.getNumSpecies()} species, {model.getNumReactions()} reactions, "
                 f"{model.getNumParameters()} parameters. Species: {'; '.join(species)}. "
                 f"Parameters: {', '.join(params[:120])}.")]
    notes = re.sub(r"\s+", " ", _notes_text(model) or "").strip()
    if notes:
        sections += [("description", f"{head} -- description: {p}") for p in _pieces(notes)]
    if online:
        for pmid in refs["pmids"][:3]:
            try:
                art = pubmed_record(pmid)
            except Exception:
                continue
            meta.setdefault("pmid", pmid)
            text = (f"{head} -- article (PMID {pmid}): {art['title']}. "
                    f"{', '.join(art['authors'][:6])}. {art['journal']} {art['year']}. "
                    f"DOI {art['doi']}. Abstract: {art.get('abstract', '')}")
            sections += [("article", p) for p in _pieces(text)]
    return [{"id": f"model::{meta['source_id']}::{sec}::{i}", "text": text,
             "metadata": {**meta, "section": sec}}
            for i, (sec, text) in enumerate(sections)]


def _save_chunks(name, records, path=None):
    """Replace one source's records in a JSONL file (kept for full rebuilds)."""
    path = path or CHUNKS_FILE
    field = "source_id" if records and records[0]["metadata"].get("source_id") else "source_name"
    identity = records[0]["metadata"][field] if records else name
    kept = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["metadata"].get(field) != identity:
                kept.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept + [json.dumps(r) for r in records]) + "\n", encoding="utf-8")


def _update_live_index(name, records, collection=None):
    """Upsert into Chroma + BM25. Returns None, or why the live index was skipped."""
    try:
        from rank_bm25 import BM25Okapi
        from dynalyst.rag import retrieve
    except ImportError as e:
        return f"RAG stack not installed ({e.name}); the model is saved for the next index build"
    client = retrieve._get_chroma_client()
    coll = client.get_or_create_collection(collection or config.COLLECTION_MODELS,
                                           metadata={"hnsw:space": "cosine"})
    field = "source_id" if records[0]["metadata"].get("source_id") else "source_name"
    identity = records[0]["metadata"][field]
    emb = retrieve._get_embed_model(config.DEVICE).encode(
        [r["text"] for r in records], normalize_embeddings=True, convert_to_numpy=True)
    coll.delete(where={field: identity})
    coll.add(ids=[r["id"] for r in records], embeddings=emb.tolist(),
             documents=[r["text"] for r in records], metadatas=[r["metadata"] for r in records])

    payload = {"ids": [], "texts": [], "metadatas": []}
    if config.BM25_INDEX_PATH.exists():
        with open(config.BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
    keep = [i for i, m in enumerate(payload["metadatas"]) if m.get(field) != identity]
    payload = {k: [payload[k][i] for i in keep] for k in ("ids", "texts", "metadatas")}
    for r in records:
        payload["ids"].append(r["id"])
        payload["texts"].append(r["text"])
        payload["metadatas"].append(r["metadata"])
    payload["bm25"] = BM25Okapi([t.lower().split() for t in payload["texts"]])
    config.BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.BM25_INDEX_PATH.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(config.BM25_INDEX_PATH)
    retrieve._get_bm25.cache_clear()           # the next retrieve() sees the new model
    return None


def index_model(model_name_or_path: str) -> str:
    """Tool: add (or refresh) a model in the knowledge base so retrieve_knowledge finds it."""
    try:
        records = model_chunks(model_name_or_path)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    name = records[0]["metadata"]["source_name"]
    _save_chunks(name, records)
    try:
        skipped = _update_live_index(name, records)
    except Exception as e:            # never lose the saved model over an index problem
        skipped = f"live index update failed ({type(e).__name__}: {e}); saved for the next build"
    sections = sorted({r["metadata"]["section"] for r in records})
    return json.dumps({"model": name, "chunks": len(records), "sections": sections,
                       "knowledge_base": "updated" if skipped is None else "saved, not yet searchable",
                       **({"note": skipped} if skipped else {})}, indent=2)


def auto_index(model_name_or_path: str) -> dict:
    """index_model for hooks: a short status dict, never raises."""
    try:
        res = json.loads(index_model(model_name_or_path))
    except Exception as e:
        return {"knowledge_base": f"not updated ({type(e).__name__}: {e})"}
    if res.get("error"):
        return {"knowledge_base": f"not updated ({res['error']})"}
    return {"knowledge_base": res["knowledge_base"], **({"note": res["note"]} if "note" in res else {})}


# -- uploaded PDFs ---------------------------------------------------------------

def index_pdf(path_or_name: str) -> str:
    """Tool: add (or refresh) a PDF in the knowledge base, searchable at once."""
    from dynalyst.analysis.pdf_tools import _resolve_pdf_path
    from dynalyst.rag.ingest_pdf import process_pdf
    try:
        path = _resolve_pdf_path(path_or_name)
        records = process_pdf(path, config.PDF_CHUNK_TOKENS, config.PDF_CHUNK_OVERLAP_TOKENS)
    except Exception as e:
        return json.dumps({"error": f"Could not read {path_or_name}: {type(e).__name__}: {e}"})
    if not records:
        return json.dumps({"error": f"No text could be read from {path_or_name} (scanned pages?)."})
    name = records[0]["metadata"]["source_name"]
    _save_chunks(name, records, PAPER_CHUNKS_FILE)
    try:
        skipped = _update_live_index(name, records, config.COLLECTION_PAPERS)
    except Exception as e:
        skipped = f"live index update failed ({type(e).__name__}: {e}); saved for the next build"
    return json.dumps({"pdf": path.name, "chunks": len(records),
                       "pages": max(r["metadata"]["page"] for r in records),
                       "knowledge_base": "updated" if skipped is None else "saved, not yet searchable",
                       **({"note": skipped} if skipped else {})}, indent=2)


def _indexed_sources() -> set:
    # Saved JSONL is a retry source, not evidence that search indexing committed.
    if config.BM25_INDEX_PATH.exists():
        try:
            from dynalyst.rag import retrieve
            sparse = {m.get("source_name") for m in retrieve._get_bm25()["metadatas"]}
            collection = retrieve._get_chroma_client().get_collection(config.COLLECTION_PAPERS)
            dense = {m.get("source_name") for m in collection.get(include=["metadatas"])["metadatas"]}
            return sparse & dense
        except Exception:
            pass
    return set()


def sync_papers() -> list:
    """Index every PDF in data/papers the knowledge base does not hold yet, so an
    upload is searchable from the next question on. Returns the names indexed."""
    if not config.PAPERS_DIR.exists():
        return []
    have = _indexed_sources()
    new = []
    for pdf in sorted(config.PAPERS_DIR.glob("*.pdf")):
        if pdf.stem not in have:
            result = json.loads(index_pdf(str(pdf)))
            if result.get("knowledge_base") == "updated":
                new.append(pdf.name)
    return new


TOOL_DEFINITIONS = [
    {"name": "index_pdf",
     "description": "Add an uploaded PDF (paper, supplement) to the knowledge base so "
                    "retrieve_knowledge searches it at once. New PDFs in data/papers are added "
                    "automatically before each knowledge-base search.",
     "input_schema": {"type": "object", "properties": {
         "path_or_name": {"type": "string"}}, "required": ["path_or_name"]}},
    {"name": "add_model_to_knowledge_base",
     "description": "Save a model into the knowledge base (description, species and their "
                    "proteins, parameters, its article's abstract) so later questions about it "
                    "are answered from it. Models downloaded from BioModels are added "
                    "automatically; use this for a model the user wants remembered / saved.",
     "input_schema": {"type": "object", "properties": {
         "model_name_or_path": {"type": "string"}}, "required": ["model_name_or_path"]}},
]
TOOL_FUNCTIONS = {"add_model_to_knowledge_base": index_model, "index_pdf": index_pdf}
