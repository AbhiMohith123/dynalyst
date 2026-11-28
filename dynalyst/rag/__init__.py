"""Knowledge base: ingest PDFs/code/docs into chunks, build the Chroma + BM25
index, and hybrid retrieval. Run the ingest/index steps as modules, e.g.
`python -m dynalyst.rag.ingest_pdf --input_dir data/papers --output data/chunks_papers.jsonl`."""
