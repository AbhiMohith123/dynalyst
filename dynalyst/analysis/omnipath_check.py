"""OmniPath network check: does prior knowledge support a model's reactions,
and what does the model leave out?

For an SBML model:
  1. MAP each species to a gene symbol -- a species_map the caller gives first,
     else the model's own UniProt annotations resolved in UniProt (a non-human
     protein (e.g. a Xenopus or yeast entry) is matched to its human orthologue
     by gene symbol, and the result says so). Unmapped species are listed.
  2. READ the model's protein-level edges from its reactions:
       modification  enzyme -> substrate (the enzyme is a modifier, a species in
                     the rate law that is neither reactant nor product, or a
                     protein on both sides); phosphorylation / dephosphorylation
                     from the reaction's GO / SBO terms, else its name
       binding       two proteins forming a complex
  3. CHECK every edge against OmniPath (omnipathdb.org web service):
       enzyme-substrate table   site-level support (residues, e.g. T185, Y187)
       interaction table        directed / signed literature-curated edges
       complexes table          binding partners in a known complex
  4. FIND what the model leaves out: OmniPath edges between model proteins that
     no reaction represents, external proteins closing a loop A -> X -> B
     (candidate feedback / cross-talk), top external regulators and targets.

OmniPath gives topology, direction, sign and references -- never rate
constants or concentrations. Requests are cached under
OUTPUT_DIR/omnipath/cache for a week; the HTTP layer is _fetch, which the
tests replace.
"""
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from dynalyst import config

OMNIPATH_URL = "https://omnipathdb.org"
CACHE_DAYS = 7
ORGANISMS = {9606: "human", 10090: "mouse", 10116: "rat"}

_UNIPROT_URI = re.compile(r"uniprot[/:]([A-Z0-9]{6,10})(?:-\d+)?", re.I)
_GO_URI = re.compile(r"GO[:_](\d{7})", re.I)
_SBO_URI = re.compile(r"SBO[:_](\d{7})", re.I)
# GO / SBO terms that name the modification a reaction performs
_PHOS = {"GO:0006468", "GO:0004672", "GO:0004674", "GO:0004713", "GO:0004708", "GO:0016310",
         "SBO:0000216"}
_DEPHOS = {"GO:0006470", "GO:0004721", "GO:0004722", "GO:0004725", "GO:0008138", "GO:0016311",
           "SBO:0000330"}


def _err(msg, **extra):
    return json.dumps({"error": msg, **extra})


def _out_dir():
    d = config.OUTPUT_DIR / "omnipath"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- reading the SBML file (plain XML: works for L2 and L3) --------------------------

def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _children(node, name):
    return [c for c in node if _local(c.tag) == name]


def _first(node, name):
    return next(iter(_children(node, name)), None)


def _rdf_terms(elem):
    """[(qualifier, uri)] from an element's <annotation> RDF block."""
    ann = _first(elem, "annotation")
    out = []
    if ann is None:
        return out
    for desc in ann.iter():
        if _local(desc.tag) != "Description":
            continue
        for q in desc:
            qual = _local(q.tag)
            for li in q.iter():
                uri = next((v for k, v in li.attrib.items() if _local(k) == "resource"), None)
                if uri:
                    out.append((qual, uri))
    return out


def read_network(path) -> dict:
    """{'model', 'species': {id: {...}}, 'reactions': [...]} from an SBML file."""
    root = ET.parse(str(path)).getroot()
    model = next((e for e in root.iter() if _local(e.tag) == "model"), None)
    if model is None:
        raise ValueError(f"{Path(path).name} has no SBML <model>.")
    species = {}
    for sp in (e for e in model.iter() if _local(e.tag) == "species"):
        terms = _rdf_terms(sp)
        accs, parts = [], []
        for qual, uri in terms:
            m = _UNIPROT_URI.search(uri)
            if m:
                (parts if qual == "hasPart" else accs).append(m.group(1).upper())
        species[sp.get("id")] = {
            "id": sp.get("id"), "name": sp.get("name") or None,
            "uniprot": list(dict.fromkeys(accs)), "has_part": list(dict.fromkeys(parts)),
            "constant": sp.get("constant") == "true" or sp.get("boundaryCondition") == "true"}
    reactions = []
    for r in (e for e in model.iter() if _local(e.tag) == "reaction"):
        def refs(list_name, item):
            lst = _first(r, list_name)
            return [] if lst is None else [x.get("species") for x in _children(lst, item)]
        law = _first(r, "kineticLaw")
        law_names = set()
        if law is not None:
            law_names = {(c.text or "").strip() for c in law.iter() if _local(c.tag) == "ci"}
        terms = set()
        for _, uri in _rdf_terms(r):
            for rx, pre in ((_GO_URI, "GO:"), (_SBO_URI, "SBO:")):
                m = rx.search(uri)
                if m:
                    terms.add(pre + m.group(1))
        if r.get("sboTerm"):
            terms.add(r.get("sboTerm").upper().replace("_", ":"))
        reactions.append({
            "id": r.get("id"), "name": r.get("name") or "",
            "reactants": refs("listOfReactants", "speciesReference"),
            "products": refs("listOfProducts", "speciesReference"),
            "modifiers": refs("listOfModifiers", "modifierSpeciesReference"),
            "law_species": sorted(n for n in law_names if n in species),
            "terms": sorted(terms)})
    return {"model": model.get("name") or model.get("id"), "model_id": model.get("id"),
            "species": species, "reactions": reactions}


# -- species -> gene symbols ------------------------------------------------------------

def _cache_path(kind, key):
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = _out_dir() / "cache"
    d.mkdir(exist_ok=True)
    return d / f"{kind}_{h}.json"


def _cached(kind, key, compute):
    p = _cache_path(kind, key)
    if p.exists() and time.time() - p.stat().st_mtime < CACHE_DAYS * 86400:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    value = compute()
    p.write_text(json.dumps(value), encoding="utf-8")
    return value


def _uniprot_genes(accession) -> dict:
    """{'genes': [...], 'organism': ...} for a UniProt accession (cached)."""
    def compute():
        from dynalyst.analysis.literature import uniprot_record
        rec = uniprot_record(accession)
        return {"genes": rec.get("genes") or [], "organism": rec.get("organism")}
    return _cached("uniprot", accession, compute)


def _symbol(gene, organism):
    """A gene name in the target organism's convention, orthologue suffixes
    removed (Xenopus 'abc1.L' / 'abc1-a' -> ABC1; mouse/rat -> Abc1)."""
    g = re.sub(r"(\.[LS]|-[ab])$", "", str(gene).strip(), flags=re.I)
    return g.upper() if organism == 9606 else g[:1].upper() + g[1:].lower()


_TARGET_ORGANISM_NAMES = {9606: "Homo sapiens", 10090: "Mus musculus", 10116: "Rattus norvegicus"}


def map_species(net, species_map=None, organism=9606) -> dict:
    """{species_id: {'genes': [...], 'how': ..., 'note': ...}}."""
    species_map = species_map or {}
    out = {}
    for sid, sp in net["species"].items():
        given = species_map.get(sid) or (species_map.get(sp["name"]) if sp["name"] else None)
        if given:
            genes = [given] if isinstance(given, str) else list(given)
            out[sid] = {"genes": [_symbol(g, organism) for g in genes], "how": "species_map"}
            continue
        accs = sp["has_part"] or sp["uniprot"]
        if not accs:
            out[sid] = {"genes": [], "how": "unmapped",
                        "note": "no UniProt annotation; give it in species_map"}
            continue
        genes, organisms, failed = [], set(), []
        for acc in accs:
            try:
                rec = _uniprot_genes(acc)
            except Exception:
                failed.append(acc)
                continue
            if rec["genes"]:
                genes.append(_symbol(rec["genes"][0], organism))
            if rec.get("organism"):
                organisms.add(rec["organism"])
        genes = list(dict.fromkeys(genes))
        entry = {"genes": genes, "how": "uniprot annotation", "uniprot": accs}
        foreign = sorted(o for o in organisms if o != _TARGET_ORGANISM_NAMES.get(organism))
        if foreign and genes:
            entry["note"] = (f"annotated with {', '.join(foreign)} protein(s); matched to the "
                             f"{ORGANISMS[organism]} orthologue by gene symbol")
        if failed:
            entry["note"] = (entry.get("note", "") + f"; UniProt did not answer for {failed}").lstrip("; ")
        if not genes:
            entry["how"] = "unmapped"
        out[sid] = entry
    return out


# -- the model's own edges ------------------------------------------------------------------

def _modification(rx):
    terms = set(rx["terms"])
    if terms & _DEPHOS or re.search(r"dephosphorylat", rx["name"], re.I):
        return "dephosphorylation"
    if terms & _PHOS or re.search(r"phosphorylat", rx["name"], re.I):
        return "phosphorylation"
    return "modification"


def model_edges(net, mapping) -> list:
    """Protein-level edges the model's reactions encode (one per enzyme/substrate
    or binding pair, listing every reaction that encodes it)."""
    genes = lambda sids: [g for s in sids for g in mapping.get(s, {}).get("genes", [])]
    edges = {}
    for rx in net["reactions"]:
        reac, prod = set(rx["reactants"]), set(rx["products"])
        catalysts = set(rx["modifiers"]) | (set(rx["law_species"]) - reac - prod) | (reac & prod)
        changed = (reac | prod) - catalysts
        subs = set(genes(changed))
        # another form of the substrate in the rate law (e.g. the unmodified protein slowing
        # its own modification) is an inhibitor, not an enzyme
        enz = set(genes(catalysts)) - subs
        if enz and subs:
            kind = _modification(rx)
            for e in sorted(enz):
                for s in sorted(subs):
                    key = ("modification", e, s, kind)
                    edges.setdefault(key, {"type": "modification", "source": e, "target": s,
                                           "modification": kind, "reactions": []})
                    edges[key]["reactions"].append(rx["id"])
        # binding: a product species carrying >= 2 proteins formed from separate reactants
        for p in prod - reac:
            pg = sorted(set(mapping.get(p, {}).get("genes", [])))
            if len(pg) >= 2 and len(reac) >= 2:
                for i, a in enumerate(pg):
                    for b in pg[i + 1:]:
                        key = ("binding", a, b, None)
                        edges.setdefault(key, {"type": "binding", "source": a, "target": b,
                                               "modification": None, "reactions": []})
                        edges[key]["reactions"].append(rx["id"])
    return list(edges.values())


# -- OmniPath web service -------------------------------------------------------------------

def _fetch(endpoint, params) -> list:
    """GET omnipathdb.org/<endpoint> as JSON rows (cached, retried)."""
    from dynalyst.analysis.biomodels import _get
    params = {**params, "format": "json"}
    key = endpoint + json.dumps(params, sort_keys=True)
    return _cached("omnipath", key,
                   lambda: _get(f"{OMNIPATH_URL}/{endpoint}", params, timeout=60))


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return [x for x in re.split(r"[;,]", str(v)) if x]


def _truthy(v):
    return v in (True, 1, "1", "True", "true", "yes")


def _pmids(refs):
    return sorted({r.split(":")[-1] for r in _as_list(refs) if r.split(":")[-1].isdigit()})


def query_omnipath(gene_symbols, organism=9606) -> dict:
    """Interactions, enzyme-substrate relations and complexes touching the genes."""
    partners = ",".join(sorted(gene_symbols))
    common = {"genesymbols": "yes", "organisms": organism}
    interactions = _fetch("interactions", {**common, "partners": partners,
                                           "fields": "sources,references,curation_effort"})
    enzsub = _fetch("enzsub", {**common, "partners": partners,
                               "fields": "sources,references,curation_effort"})
    try:
        complexes = _fetch("complexes", {"proteins": partners})
    except Exception:          # complexes are optional evidence
        complexes = []
    if not interactions and not enzsub:
        # the service did not filter by these names: take the whole tables once
        # (cached for CACHE_DAYS) and filter them here
        interactions = _fetch("interactions", {**common, "fields": "sources,references,curation_effort"})
        enzsub = _fetch("enzsub", {**common, "fields": "sources,references,curation_effort"})
    genes = set(gene_symbols)
    keep = lambda a, b: a in genes or b in genes
    interactions = [r for r in interactions
                    if keep(r.get("source_genesymbol"), r.get("target_genesymbol"))]
    enzsub = [r for r in enzsub if keep(r.get("enzyme_genesymbol"), r.get("substrate_genesymbol"))]
    complexes = [c for c in complexes
                 if genes & set(re.split(r"[_;,]", str(c.get("components_genesymbols") or "")))]
    return {"interactions": interactions, "enzsub": enzsub, "complexes": complexes}


def _edge_row(r):
    return {"source": r.get("source_genesymbol"), "target": r.get("target_genesymbol"),
            "directed": _truthy(r.get("consensus_direction") or r.get("is_directed")),
            "sign": ("activation" if _truthy(r.get("consensus_stimulation") or r.get("is_stimulation"))
                     else "inhibition" if _truthy(r.get("consensus_inhibition") or r.get("is_inhibition"))
                     else "unknown"),
            "n_references": len(_pmids(r.get("references"))),
            "sources": _as_list(r.get("sources"))[:8],
            "pmids": _pmids(r.get("references"))[:8]}


def _site_row(r):
    res = f"{r.get('residue_type') or ''}{r.get('residue_offset') or ''}" or None
    return {"enzyme": r.get("enzyme_genesymbol"), "substrate": r.get("substrate_genesymbol"),
            "modification": r.get("modification"), "residue": res,
            "n_references": len(_pmids(r.get("references"))), "sources": _as_list(r.get("sources"))[:8]}


# -- comparison -------------------------------------------------------------------------------

def _check_edge(edge, ints, sites, complexes):
    a, b = edge["source"], edge["target"]
    fwd = [i for i in ints if i["source"] == a and i["target"] == b]
    rev = [i for i in ints if i["source"] == b and i["target"] == a]
    ev = {"interaction_forward": fwd[:3], "interaction_reverse": rev[:3]}
    if edge["type"] == "binding":
        cx = [c.get("name") for c in complexes
              if {a, b} <= set(re.split(r"[_;,]", str(c.get("components_genesymbols") or "")))]
        ev["complexes"] = cx[:5]
        status = ("supported: known complex" if cx else
                  "supported: interaction" if fwd or rev else "not found in OmniPath")
        return status, ev
    s = [x for x in sites if x["enzyme"] == a and x["substrate"] == b]
    same = [x for x in s if edge["modification"] in ("modification", x["modification"])]
    ev["sites"] = sorted({x["residue"] for x in same if x["residue"]})
    ev["site_references"] = max([x["n_references"] for x in same] or [0])
    if same:
        status = "supported: site-level"
    elif s:
        status = f"supported: site-level, but OmniPath records {sorted({x['modification'] for x in s})}"
    elif any(i["directed"] for i in fwd):
        status = "supported: directed interaction"
    elif fwd:
        status = "supported: undirected interaction"
    elif rev:
        status = "opposite direction only"
    else:
        status = "not found in OmniPath"
    return status, ev


def _reaches(edges, start, goal):
    seen, todo = {start}, [start]
    while todo:
        n = todo.pop()
        if n == goal:
            return True
        for e in edges:
            if e["source"] == n and e["target"] not in seen:
                seen.add(e["target"])
                todo.append(e["target"])
    return False


def compare(edges, mapping, data, max_neighbours=15) -> dict:
    model_genes = sorted({g for m in mapping.values() for g in m["genes"]})
    gset = set(model_genes)
    ints = [_edge_row(r) for r in data["interactions"]]
    sites = [_site_row(r) for r in data["enzsub"]]
    checked = []
    for e in edges:
        status, ev = _check_edge(e, ints, sites, data["complexes"])
        checked.append({**e, "status": status, "evidence": ev})

    represented = {(e["source"], e["target"]) for e in edges} | \
                  {(e["target"], e["source"]) for e in edges if e["type"] == "binding"}
    missing = {}
    for i in ints:
        if (i["source"] in gset and i["target"] in gset and i["source"] != i["target"]
                and i["directed"] and (i["source"], i["target"]) not in represented):
            missing.setdefault((i["source"], i["target"]), {**i, "kind": "interaction", "residues": []})
    for x in sites:
        if (x["enzyme"] in gset and x["substrate"] in gset and x["enzyme"] != x["substrate"]
                and (x["enzyme"], x["substrate"]) not in represented):
            m = missing.setdefault((x["enzyme"], x["substrate"]),
                                   {"source": x["enzyme"], "target": x["substrate"], "directed": True,
                                    "sign": "unknown", "n_references": 0, "sources": [], "pmids": [],
                                    "kind": "enzyme-substrate", "residues": []})
            m["kind"] = "enzyme-substrate"
            if x["residue"]:
                m["residues"] = sorted(set(m["residues"]) | {f"{x['modification']} {x['residue']}"})
            m["n_references"] = max(m["n_references"], x["n_references"])
    missing = sorted(missing.values(), key=lambda m: -m["n_references"])
    for m in missing:   # would it close a loop in the model?
        m["closes_loop"] = _reaches(edges, m["target"], m["source"])

    directed = [i for i in ints if i["directed"]]
    out_of = {}
    for i in directed:
        if i["source"] in gset and i["target"] not in gset:
            out_of.setdefault(i["target"], []).append(i)
    loops = []
    for i in directed:
        x, b = i["source"], i["target"]
        if x in out_of and b in gset:
            for first in out_of[x]:
                a = first["source"]
                kind = "feedback" if a == b or _reaches(edges, b, a) else "cross-talk / feed-forward"
                loops.append({"path": f"{a} -> {x} -> {b}", "via": x, "kind": kind,
                              "signs": f"{first['sign']}, {i['sign']}",
                              "evidence": min(first["n_references"], i["n_references"])})
    loops = sorted({l["path"]: l for l in loops}.values(),
                   key=lambda l: (l["kind"] != "feedback", -l["evidence"]))[:max_neighbours]

    regulators = sorted((i for i in directed if i["target"] in gset and i["source"] not in gset),
                        key=lambda i: -i["n_references"])[:max_neighbours]
    targets = sorted((i for i in directed if i["source"] in gset and i["target"] not in gset),
                     key=lambda i: -i["n_references"])[:max_neighbours]
    return {"model_genes": model_genes, "edges": checked, "missing_internal_edges": missing,
            "loop_candidates": loops, "external_regulators": regulators,
            "external_targets": targets, "n_interactions_retrieved": len(ints),
            "n_enzsub_retrieved": len(sites)}


# -- report files -----------------------------------------------------------------------------

def _plot(result, path):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    genes = result["model_genes"]
    loops = [l for l in result["loop_candidates"] if l["kind"] == "feedback"][:6]
    extra = list(dict.fromkeys(l["via"] for l in loops))
    pos = {g: (math.cos(2 * math.pi * k / max(len(genes), 1)), math.sin(2 * math.pi * k / max(len(genes), 1)))
           for k, g in enumerate(genes)}
    for k, x in enumerate(extra):
        a = 2 * math.pi * (k + 0.5) / max(len(extra), 1)
        pos[x] = (1.9 * math.cos(a), 1.9 * math.sin(a))
    colour = lambda st: ("#2e8b57" if st.startswith("supported: site") else
                         "#e69f00" if st.startswith("supported") else "#c0392b")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    def arrow(a, b, **kw):
        if a in pos and b in pos and a != b:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-|>", shrinkA=22, shrinkB=22,
                                        connectionstyle="arc3,rad=0.15", **kw))
    for e in result["edges"]:
        arrow(e["source"], e["target"], color=colour(e["status"]), lw=2.4)
    for m in result["missing_internal_edges"]:
        arrow(m["source"], m["target"], color="#555555", lw=1.4, ls="--")
    for l in loops:
        a, x, b = [s.strip() for s in l["path"].split("->")]
        arrow(a, x, color="#7b68ee", lw=1.0, ls=":")
        arrow(x, b, color="#7b68ee", lw=1.0, ls=":")
    for g, (x, y) in pos.items():
        inside = g in genes
        ax.scatter([x], [y], s=1500 if inside else 800, zorder=3,
                   color="#dfe7fb" if inside else "#f1edfd",
                   edgecolors="#4b4ba8" if inside else "#7b68ee", linewidths=1.5)
        ax.text(x, y, g, ha="center", va="center", fontsize=10 if inside else 8, zorder=4,
                weight="bold" if inside else "normal")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="#2e8b57", lw=2.4, label="model edge, site-level support"),
                       Line2D([], [], color="#e69f00", lw=2.4, label="model edge, interaction support"),
                       Line2D([], [], color="#c0392b", lw=2.4, label="model edge, not in OmniPath"),
                       Line2D([], [], color="#555555", lw=1.4, ls="--", label="in OmniPath, not in model"),
                       Line2D([], [], color="#7b68ee", lw=1.0, ls=":", label="feedback via external protein")],
              loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"OmniPath check: {Path(result['model_file']).stem}", fontsize=13)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _excel(result, path):
    import pandas as pd
    flat = lambda rows: [{k: (", ".join(map(str, v)) if isinstance(v, list) else
                              json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
                         for r in rows]
    sheets = {
        "species_mapping": [{"species": s, **m} for s, m in result["species_mapping"].items()],
        "model_edges": [{**{k: v for k, v in e.items() if k != "evidence"},
                         "sites": e["evidence"].get("sites"),
                         "complexes": e["evidence"].get("complexes"),
                         "pmids": (e["evidence"]["interaction_forward"] or [{}])[0].get("pmids")}
                        for e in result["edges"]],
        "missing_in_model": result["missing_internal_edges"],
        "loop_candidates": result["loop_candidates"],
        "external_regulators": result["external_regulators"],
        "external_targets": result["external_targets"],
    }
    with pd.ExcelWriter(path) as xw:
        for name, rows in sheets.items():
            pd.DataFrame(flat(rows) or [{"note": "none"}]).to_excel(xw, sheet_name=name, index=False)


def _conclusion(result):
    edges = result["edges"]
    if not edges:
        return ("No protein-level edge could be read from the model's reactions (species not "
                "mapped to proteins, or no enzyme in any rate law).")
    sup = [e for e in edges if e["status"].startswith("supported")]
    site = [e for e in edges if e["status"] == "supported: site-level"]
    parts = [f"{len(sup)} of {len(edges)} model edge(s) are supported by OmniPath, "
             f"{len(site)} at site level"]
    ex = next((e for e in site if e["evidence"]["sites"]), None)
    if ex:
        parts.append(f"(e.g. {ex['source']} -> {ex['target']} {ex['modification']} at "
                     f"{', '.join(ex['evidence']['sites'][:4])})")
    text = " ".join(parts) + "."
    bad = [f"{e['source']} -> {e['target']}" for e in edges if not e["status"].startswith("supported")]
    if bad:
        text += f" Not supported: {', '.join(bad)}."
    miss = result["missing_internal_edges"][:3]
    if miss:
        text += (" OmniPath also lists " + "; ".join(
            f"{m['source']} -> {m['target']}" + (" (would close a loop)" if m["closes_loop"] else "")
            for m in miss) + ", which the model does not contain.")
    fb = [l for l in result["loop_candidates"] if l["kind"] == "feedback"][:2]
    if fb:
        text += " Candidate feedback through other proteins: " + "; ".join(l["path"] for l in fb) + "."
    return text


# -- the tool ------------------------------------------------------------------------------------

def omnipath_network_check(model_name_or_path: str, species_map: dict | None = None,
                           organism: int = 9606, include_neighbours: bool = True,
                           max_neighbours: int = 15) -> str:
    """Tool: check a model's reactions against OmniPath prior knowledge."""
    organism = int(organism)
    if organism not in ORGANISMS:
        return _err(f"organism must be one of {ORGANISMS} (NCBI taxonomy ID); OmniPath "
                    "covers human, mouse and rat only.")
    try:
        from dynalyst.analysis.generic_sbml import resolve_sbml_path
        path = resolve_sbml_path(str(model_name_or_path))
    except (FileNotFoundError, ValueError) as e:
        return _err(str(e))
    except ImportError:
        path = Path(model_name_or_path)
    if Path(path).suffix.lower() != ".xml":
        return _err(f"{Path(path).name} is not an SBML (.xml) file; convert it first.")
    try:
        net = read_network(path)
    except (ET.ParseError, ValueError) as e:
        return _err(f"Could not read {Path(path).name}: {e}")

    mapping = map_species(net, species_map, organism)
    genes = sorted({g for m in mapping.values() for g in m["genes"]})
    unmapped = [s for s, m in mapping.items() if not m["genes"]]
    if not genes:
        return _err("No species could be mapped to a protein: the model has no UniProt "
                    "annotations (or UniProt did not answer). Pass species_map, e.g. "
                    '{"S1": "GENE1", "S2": "GENE2"}.', unmapped_species=unmapped)
    edges = model_edges(net, mapping)
    try:
        data = query_omnipath(genes, organism)
    except Exception as e:
        return _err(f"OmniPath did not answer ({type(e).__name__}: {e}); try again in a few "
                    "minutes. Nothing about the model was concluded.")
    if not data["interactions"] and not data["enzsub"]:
        return _err(f"OmniPath returned nothing for {genes} ({ORGANISMS[organism]}). Check the "
                    "gene symbols in species_mapping, or pass species_map.",
                    species_mapping=mapping)
    result = {"model": net["model"], "model_file": Path(path).name,
              "organism": f"{ORGANISMS[organism]} ({organism})",
              "species_mapping": mapping, "unmapped_species": unmapped,
              **compare(edges, mapping, data, max_neighbours)}
    if not include_neighbours:
        for k in ("loop_candidates", "external_regulators", "external_targets"):
            result[k] = []
    stem = f"{Path(path).stem}_{organism}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = _out_dir()
    plot_path, xlsx = out / f"{stem}_network.png", out / f"{stem}.xlsx"
    result["conclusion"] = _conclusion(result)
    result["caveats"] = [
        "OmniPath holds topology, direction, sign and references, not kinetics: support for an "
        "edge says nothing about the model's rate laws or parameter values.",
        "A missing edge may be a deliberate simplification of the model, not an error.",
    ] + (["Species were matched to orthologues by gene symbol (see species_mapping notes)."]
         if any("orthologue" in m.get("note", "") for m in mapping.values()) else [])
    try:
        _plot(result, plot_path)
        result["plot_path"] = str(plot_path)
        result["plot_note"] = ("A real plot was saved. Embed it in the answer as markdown: "
                               f"![{Path(path).stem} OmniPath network]"
                               f"({config.display_path(plot_path)})")
    except Exception as e:
        result["plot_error"] = f"{type(e).__name__}: {e}"
    try:
        _excel(result, xlsx)
        result["excel_paths"] = {"report": str(xlsx)}
    except Exception as e:
        result["excel_error"] = f"{type(e).__name__}: {e}"
    return json.dumps(result, indent=2, default=str)


TOOL_DEFINITIONS = [{
    "name": "omnipath_network_check",
    "description": (
        "Check a model's reaction network against OmniPath prior knowledge (human / mouse / "
        "rat): maps species to proteins (UniProt annotations, or species_map), reads the "
        "enzyme -> substrate and binding edges from the reactions, and reports for each whether "
        "OmniPath supports it (site-level, e.g. phosphorylation residues; directed interaction; "
        "known complex) with references, plus edges between the model's proteins that the model "
        "lacks, candidate feedback loops through other proteins, and top external regulators "
        "and targets. Writes a network plot and an Excel report. Use for 'is this model "
        "consistent with the literature', 'what does it miss', 'which feedbacks could be added'."),
    "input_schema": {"type": "object", "properties": {
        "model_name_or_path": {"type": "string", "description": "SBML model name or path."},
        "species_map": {"type": "object", "additionalProperties": {
            "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
            "description": "Optional species id -> gene symbol(s), overriding annotations, "
                           "e.g. {\"S1\": \"GENE1\", \"S2\": \"GENE2\"}. A complex maps to "
                           "a list."},
        "organism": {"type": "integer", "enum": list(ORGANISMS),
                     "description": "NCBI taxonomy: 9606 human (default), 10090 mouse, 10116 rat."},
        "include_neighbours": {"type": "boolean",
                               "description": "Default true: also list loop candidates and "
                                              "external regulators / targets."},
        "max_neighbours": {"type": "integer", "description": "Cap per list. Default 15."},
    }, "required": ["model_name_or_path"]},
}]

TOOL_FUNCTIONS = {"omnipath_network_check": lambda **kw: omnipath_network_check(**kw)}
