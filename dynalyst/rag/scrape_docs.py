"""
Scrapes the full jaxkineticmodel documentation site (mkdocs pages) into
data/docs_html/*.md for the knowledge base. Run this ON YOUR SERVER (this
sandbox's network doesn't allow abeellab.github.io) — it needs normal internet
access, not the GPU.

Usage:
    python scrape_docs.py
"""
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from dynalyst import config

BASE = "https://abeellab.github.io/jaxkineticmodel"
PAGES = {
    "home": "/",
    "building_models": "/building_models/",
    "sbml_loading": "/SBML/",
    "training_models": "/training_models/",
    "simulated_dbtl": "/simulated-DBTL/",
    "custom_models_glycolysis": "/glycolysis/",
}


def fetch_and_clean(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # mkdocs (readthedocs theme) puts the actual content in <div role="main">
    main = soup.find("div", attrs={"role": "main"}) or soup.find("article") or soup.body
    if main is None:
        return resp.text

    # Strip nav/sidebar leftovers if any slipped through, keep headings/code/text
    for tag in main.find_all(["nav", "script", "style"]):
        tag.decompose()

    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main():
    out_dir = config.DOCS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in PAGES.items():
        url = BASE + path
        print(f"Fetching {url} ...")
        try:
            content = fetch_and_clean(url)
        except requests.RequestException as e:
            print(f"  FAILED: {e}")
            continue

        out_path = out_dir / f"{name}.md"
        out_path.write_text(f"# {name}\nSource: {url}\n\n{content}", encoding="utf-8")
        print(f"  -> saved {out_path} ({len(content)} chars)")

    print(f"\nDone. Pages saved to {out_dir}")
    print("Note: home.md and sbml_loading.md may already exist as curated seed "
          "files — this will overwrite them with freshly scraped versions, which "
          "is fine (they cover the same content).")


if __name__ == "__main__":
    main()
