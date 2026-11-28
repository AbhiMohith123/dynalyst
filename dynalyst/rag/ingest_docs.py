"""
Chunk the documentation pages in data/docs_html/*.md (either the seed files
already included, or freshly scraped ones from scrape_docs.py) into JSONL for
embedding in build_index.py.

Usage:
    python ingest_docs.py --input_dir data/docs_html --output data/chunks_docs.jsonl
"""
import argparse
import json
import re
from pathlib import Path


def split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split markdown into (header, section_text) chunks on lines starting with '#'."""
    lines = text.split("\n")
    sections = []
    current_header = "intro"
    current_lines = []

    for line in lines:
        if re.match(r"^#{1,3}\s+", line):
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return [(h, t) for h, t in sections if len(t) > 30]


def process_doc(md_path: Path) -> list[dict]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    source_name = md_path.stem
    sections = split_by_headers(text)

    records = []
    for i, (header, body) in enumerate(sections):
        records.append({
            "id": f"{source_name}_sec{i}",
            "text": f"[{source_name} — {header}]\n\n{body}",
            "metadata": {
                "source_type": "docs",
                "source_name": source_name,
                "section": header,
            },
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    md_files = sorted(args.input_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(
            f"No .md files found in {args.input_dir}. Run scrape_docs.py first, "
            f"or check the seed files (home.md, sbml_loading.md) are there."
        )

    all_records = []
    for md_path in md_files:
        print(f"Parsing {md_path.name} ...")
        records = process_doc(md_path)
        print(f"  -> {len(records)} chunks")
        all_records.extend(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(all_records)} total doc chunks to {args.output}")


if __name__ == "__main__":
    main()
