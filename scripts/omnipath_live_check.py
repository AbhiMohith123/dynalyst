"""Run the OmniPath network check on a real model against the live services
(omnipathdb.org and rest.uniprot.org). No language model and no API key needed.

    python scripts/omnipath_live_check.py my_model.xml                  # human (default)
    python scripts/omnipath_live_check.py my_model.xml --organism 10090  # mouse
    python scripts/omnipath_live_check.py my_model.xml --species-map '{"S1": "GENE1"}'
    python scripts/omnipath_live_check.py my_model.xml --raw            # also raw OmniPath rows

my_model.xml: any SBML model name or path the agent can read (data/sbml_models, models/,
or a path); its species need UniProt annotations, or give --species-map.

Prints the species mapping, every model edge with its OmniPath status and
evidence, what the model lacks, and where the plot and Excel report went.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynalyst.analysis import omnipath_check as oc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("model", help="SBML model name or path")
    ap.add_argument("--organism", type=int, default=9606, choices=sorted(oc.ORGANISMS))
    ap.add_argument("--species-map", default=None, help="JSON: species id -> gene symbol(s)")
    ap.add_argument("--raw", action="store_true", help="print the first raw OmniPath rows")
    a = ap.parse_args()

    r = json.loads(oc.omnipath_network_check(
        a.model, species_map=json.loads(a.species_map) if a.species_map else None,
        organism=a.organism))
    if "error" in r:
        print("ERROR:", r["error"])
        if r.get("species_mapping"):
            print(json.dumps(r["species_mapping"], indent=2))
        sys.exit(1)

    print(f"\n== {r['model']} ({r['model_file']}), {r['organism']}\n")
    print("Species -> proteins")
    for sid, m in r["species_mapping"].items():
        print(f"  {sid:12s} {', '.join(m['genes']) or '-':18s} {m['how']}"
              + (f"  [{m['note']}]" if m.get("note") else ""))
    print("\nModel edges")
    for e in r["edges"]:
        ev = e["evidence"]
        extra = f" sites {ev['sites']}" if ev.get("sites") else ""
        print(f"  {e['source']} -> {e['target']} ({e['modification'] or e['type']}, "
              f"reactions {e['reactions']}): {e['status']}{extra}")
    print("\nIn OmniPath, not in the model")
    for m in r["missing_internal_edges"][:10] or [{"source": "(none)", "target": ""}]:
        print(f"  {m['source']} -> {m['target']}"
              + (f"  {m.get('kind')}, {m.get('n_references')} refs"
                 + (", closes a loop" if m.get("closes_loop") else "") if m.get("kind") else ""))
    print("\nLoop candidates through other proteins")
    for l in r["loop_candidates"][:8] or [{"path": "(none)", "kind": "", "signs": ""}]:
        print(f"  {l['path']:32s} {l['kind']}  signs: {l['signs']}")
    print("\nConclusion:", r["conclusion"])
    print("Plot:  ", r.get("plot_path", r.get("plot_error")))
    print("Excel: ", (r.get("excel_paths") or {}).get("report", r.get("excel_error")))
    print(f"(retrieved {r['n_interactions_retrieved']} interactions, "
          f"{r['n_enzsub_retrieved']} enzyme-substrate rows)")

    if a.raw:
        data = oc.query_omnipath(r["model_genes"], a.organism)
        for k in ("interactions", "enzsub", "complexes"):
            print(f"\n-- raw {k}: {len(data[k])} rows; first 2:")
            print(json.dumps(data[k][:2], indent=2)[:1500])


if __name__ == "__main__":
    main()
