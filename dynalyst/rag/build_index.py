"""
Embed chunks (on GPU) and build:
  1. A ChromaDB collection (dense/semantic search) with metadata filtering
  2. A BM25 index (sparse/keyword search), pickled to disk

Usage:
    python build_index.py \
        --chunks data/chunks_papers.jsonl data/chunks_code.jsonl \
        --persist_dir chroma_db \
        --device cuda
"""
import argparse
import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dynalyst import config


def load_chunks(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def build_dense_index(records: list[dict], client: chromadb.Client, device: str):
    model = SentenceTransformer(config.TEXT_EMBED_MODEL, device=device)
    print(f"Loaded embedding model {config.TEXT_EMBED_MODEL} on {device}")

    # Split papers vs code vs docs into separate collections so retrieval can be
    # filtered/weighted per source type
    papers = [r for r in records if r["metadata"]["source_type"] == "paper"]
    code = [r for r in records if r["metadata"]["source_type"] == "code"]
    docs = [r for r in records if r["metadata"]["source_type"] == "docs"]
    models = [r for r in records if r["metadata"]["source_type"] == "model"]   # rag/model_index.py

    collections_to_build = [
        (config.COLLECTION_PAPERS, papers),
        (config.COLLECTION_CODE, code),
        (config.COLLECTION_DOCS, docs),
        (config.COLLECTION_MODELS, models),
    ]
    for name, subset in collections_to_build:
        if not subset:
            continue
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

        texts = [r["text"] for r in subset]
        embeddings = model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection.add(
            ids=[r["id"] for r in subset],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[r["metadata"] for r in subset],
        )
        print(f"Indexed {len(subset)} chunks into Chroma collection '{name}'")


def build_bm25_index(records: list[dict], output_path: Path):
    tokenized = [tokenize_for_bm25(r["text"]) for r in records]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "ids": [r["id"] for r in records],
        "texts": [r["text"] for r in records],
        "metadatas": [r["metadata"] for r in records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 index ({len(records)} docs) saved to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True, type=Path,
                     help="One or more JSONL chunk files from ingest_pdf.py / ingest_code.py")
    ap.add_argument("--persist_dir", default=str(config.CHROMA_PERSIST_DIR), type=Path)
    ap.add_argument("--device", default=config.DEVICE, choices=["cuda", "cpu"],
                     help=f"Defaults to auto-detected device (currently: {config.DEVICE})")
    args = ap.parse_args()

    records = load_chunks(args.chunks)
    print(f"Loaded {len(records)} chunks total from {len(args.chunks)} file(s)")

    client = chromadb.PersistentClient(path=str(args.persist_dir))
    build_dense_index(records, client, args.device)
    build_bm25_index(records, args.persist_dir / "bm25_index.pkl")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()
