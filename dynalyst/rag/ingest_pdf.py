"""
Parse PDFs (the jaxkineticmodel paper, its S1 text, and referenced papers) into
metadata-tagged text chunks, saved as JSONL for embedding in build_index.py.

Usage:
    python ingest_pdf.py --input_dir data/papers --output data/chunks_papers.jsonl
"""
import argparse
import json
import re
from pathlib import Path

try:  # PyMuPDF (its old name "fitz" prints a deprecation warning)
    import pymupdf as fitz
except ImportError:
    import fitz

SECTION_HEADERS = [
    "abstract", "introduction", "background", "methods", "materials and methods",
    "design and implementation", "implementation", "results", "discussion",
    "conclusion", "references", "supporting information", "author summary",
]


def approx_token_count(text: str) -> int:
    # Cheap approximation: ~4 chars/token for English scientific text.
    return max(1, len(text) // 4)


def detect_section(line: str) -> str | None:
    stripped = line.strip().lower().rstrip(":")
    if stripped in SECTION_HEADERS:
        return stripped
    return None


def extract_pages_text(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def split_into_paragraphs(text: str) -> list[str]:
    # Collapse hyphenation artifacts from PDF line wraps, then split on blank lines.
    text = re.sub(r"-\n(?=[a-z])", "", text)   # de-hyphenate wrapped words
    text = re.sub(r"\n(?=[a-z])", " ", text)   # join wrapped lines mid-sentence
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if len(p.strip()) > 20]


def chunk_paragraphs(paragraphs: list[str], chunk_tokens: int, overlap_tokens: int) -> list[str]:
    chunks, current, current_tokens = [], [], 0
    for para in paragraphs:
        para_tokens = approx_token_count(para)
        if current_tokens + para_tokens > chunk_tokens and current:
            chunks.append(" ".join(current))
            # keep the tail of the previous chunk for overlap
            overlap_text = " ".join(current)[-overlap_tokens * 4:]
            current = [overlap_text, para] if overlap_text else [para]
            current_tokens = approx_token_count(" ".join(current))
        else:
            current.append(para)
            current_tokens += para_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def process_pdf(pdf_path: Path, chunk_tokens: int, overlap_tokens: int) -> list[dict]:
    pages = extract_pages_text(pdf_path)
    source_name = pdf_path.stem

    records = []
    current_section = "unknown"
    for page_num, page_text in enumerate(pages, start=1):
        lines = page_text.split("\n")
        # crude section detection: scan lines on this page for a header match
        for line in lines:
            detected = detect_section(line)
            if detected:
                current_section = detected

        paragraphs = split_into_paragraphs(page_text)
        chunks = chunk_paragraphs(paragraphs, chunk_tokens, overlap_tokens)

        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < 40:
                continue
            records.append({
                "id": f"{source_name}_p{page_num}_c{i}",
                "text": chunk_text,
                "metadata": {
                    "source_type": "paper",
                    "source_name": source_name,
                    "page": page_num,
                    "section": current_section,
                },
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--chunk_tokens", type=int, default=400)
    ap.add_argument("--overlap_tokens", type=int, default=60)
    args = ap.parse_args()

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {args.input_dir}")

    all_records = []
    for pdf_path in pdf_files:
        print(f"Parsing {pdf_path.name} ...")
        records = process_pdf(pdf_path, args.chunk_tokens, args.overlap_tokens)
        print(f"  -> {len(records)} chunks")
        all_records.extend(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(all_records)} total chunks to {args.output}")


if __name__ == "__main__":
    main()
