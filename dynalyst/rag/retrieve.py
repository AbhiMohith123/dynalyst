"""
Hybrid retrieval: dense (embedding) search + BM25 (keyword) search, merged with
reciprocal rank fusion, optionally reranked with a cross-encoder.

Import `retrieve(query, ...)` from agent.py — this module is not meant to be run
standalone, though `python retrieve.py "some query"` works for quick testing.
"""
import pickle
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from dynalyst import config


@lru_cache(maxsize=1)
def _get_embed_model(device: str = config.DEVICE):
    return SentenceTransformer(config.TEXT_EMBED_MODEL, device=device)


@lru_cache(maxsize=1)
def _get_reranker(device: str = config.DEVICE):
    return CrossEncoder(config.RERANKER_MODEL, device=device)


@lru_cache(maxsize=1)
def _get_chroma_client():
    return chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))


@lru_cache(maxsize=1)
def _get_bm25():
    with open(config.BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def _dense_search(query: str, collection_name: str, top_k: int, device: str):
    client = _get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []
    model = _get_embed_model(device)
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_rank": i + 1,
        })
    return hits


def _bm25_search(query: str, top_k: int):
    payload = _get_bm25()
    bm25 = payload["bm25"]
    scores = bm25.get_scores(query.lower().split())
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for rank, idx in enumerate(ranked_idx, start=1):
        hits.append({
            "id": payload["ids"][idx],
            "text": payload["texts"][idx],
            "metadata": payload["metadatas"][idx],
            "bm25_rank": rank,
        })
    return hits


def _reciprocal_rank_fusion(dense_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for hit in dense_hits:
        scores[hit["id"]] = scores.get(hit["id"], 0) + 1.0 / (k + hit["dense_rank"])
        doc_lookup[hit["id"]] = hit
    for hit in bm25_hits:
        scores[hit["id"]] = scores.get(hit["id"], 0) + 1.0 / (k + hit["bm25_rank"])
        doc_lookup.setdefault(hit["id"], hit)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_lookup[doc_id] for doc_id, _ in fused]


def retrieve(
    query: str,
    collections: list[str] | None = None,
    top_k_final: int = config.TOP_K_FINAL,
    device: str = config.DEVICE,
    use_reranker: bool = config.USE_RERANKER,
) -> list[dict]:
    """
    Main entry point. Returns a list of {id, text, metadata} dicts, best first.
    `collections` restricts dense search to specific Chroma collections
    (e.g. ["jkm_code"] when the query is clearly about implementation details).
    """
    if collections is None:
        collections = [config.COLLECTION_PAPERS, config.COLLECTION_CODE, config.COLLECTION_DOCS,
                       config.COLLECTION_MODELS]

    dense_hits = []
    for coll in collections:
        dense_hits.extend(_dense_search(query, coll, config.TOP_K_DENSE, device))
    # re-rank dense_hits' dense_rank field within the merged pool
    for i, hit in enumerate(sorted(dense_hits, key=lambda h: h["dense_rank"]), start=1):
        hit["dense_rank"] = i

    bm25_hits = _bm25_search(query, config.TOP_K_BM25)

    fused = _reciprocal_rank_fusion(dense_hits, bm25_hits)

    if use_reranker and fused:
        reranker = _get_reranker(device)
        pairs = [(query, hit["text"]) for hit in fused[: max(top_k_final * 3, 10)]]
        rerank_scores = reranker.predict(pairs)
        scored = list(zip(fused[: len(pairs)], rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        fused = [hit for hit, _ in scored]

    return fused[:top_k_final]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "what solver does jaxkineticmodel use"
    for hit in retrieve(q, device="cpu", use_reranker=False):
        print(f"\n--- {hit['id']} ({hit['metadata']['source_type']}) ---")
        print(hit["text"][:300], "...")
