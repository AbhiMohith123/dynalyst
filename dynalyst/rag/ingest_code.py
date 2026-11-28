"""
Parse the jaxkineticmodel source (.py files) into chunks split by function/class
so each chunk is a complete, semantically whole unit (with its docstring).

Usage:
    python ingest_code.py --input_dir data/code --output data/chunks_code.jsonl
"""
import argparse
import ast
import json
from pathlib import Path

SKIP_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", "tests", "build", "dist"}


def get_source_segment(source: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(source, node)
    return seg if seg else ""


def extract_units(pyfile: Path, min_chars: int) -> list[dict]:
    try:
        source = pyfile.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  skip (parse error): {pyfile} -> {e}")
        return []

    units = []
    module_docstring = ast.get_docstring(tree)
    if module_docstring and len(module_docstring) >= min_chars:
        units.append({
            "kind": "module_docstring",
            "name": pyfile.stem,
            "text": module_docstring,
            "lineno": 1,
        })

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            code = get_source_segment(source, node)
            if not code or len(code) < min_chars:
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            units.append({
                "kind": kind,
                "name": node.name,
                "text": code,
                "lineno": node.lineno,
            })
    return units


def process_repo(input_dir: Path, min_chars: int) -> list[dict]:
    records = []
    py_files = [
        p for p in input_dir.rglob("*.py")
        if not any(part in SKIP_DIR_NAMES for part in p.parts)
    ]

    for pyfile in py_files:
        rel_path = pyfile.relative_to(input_dir)
        print(f"Parsing {rel_path} ...")
        units = extract_units(pyfile, min_chars)
        for i, unit in enumerate(units):
            records.append({
                "id": f"{rel_path.as_posix().replace('/', '_')}_{unit['kind']}_{unit['name']}_{i}",
                "text": f"# File: {rel_path}\n# {unit['kind']}: {unit['name']}\n\n{unit['text']}",
                "metadata": {
                    "source_type": "code",
                    "source_name": rel_path.as_posix(),
                    "unit_kind": unit["kind"],
                    "unit_name": unit["name"],
                    "lineno": unit["lineno"],
                },
            })
        print(f"  -> {len(units)} units")

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path,
                     help="Path to the cloned jaxkineticmodel repo (or subfolder)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min_chars", type=int, default=40)
    args = ap.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"{args.input_dir} does not exist — clone the repo first")

    records = process_repo(args.input_dir, args.min_chars)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} total code chunks to {args.output}")


if __name__ == "__main__":
    main()
