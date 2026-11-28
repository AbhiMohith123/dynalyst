"""omnipath_network_check on Markevich2004, offline: OmniPath and UniProt are
replaced by a small fixture in OmniPath's JSON format.

THE FIXTURE ROWS ARE TEST DATA, NOT REAL OMNIPATH OUTPUT -- they only exercise
the logic (site-level support, a missing internal edge, a feedback loop through
an external protein). scripts/omnipath_live_check.py runs the real service.
"""
import json

import pytest

from dynalyst import config
from dynalyst.analysis import omnipath_check as oc

MARKEVICH = config.LOCAL_SBML_MODELS_DIR / "Markevich2004.xml"

# the model's (Xenopus) UniProt annotations -> gene names, as UniProt would give them
FAKE_UNIPROT = {
    "P26696": {"genes": ["mapk1.L"], "organism": "Xenopus laevis"},
    "Q05116": {"genes": ["map2k1.L"], "organism": "Xenopus laevis"},
    "Q90W58": {"genes": ["dusp6.L"], "organism": "Xenopus laevis"},
}


def _i(src, tgt, stim=False, inh=False, refs=("SIGNOR:111", "SignaLink:222")):
    return {"source_genesymbol": src, "target_genesymbol": tgt, "is_directed": 1,
            "consensus_direction": 1, "is_stimulation": int(stim), "is_inhibition": int(inh),
            "consensus_stimulation": int(stim), "consensus_inhibition": int(inh),
            "sources": ["SIGNOR", "SignaLink3"], "references": list(refs)}


def _site(enz, sub, mod, res, off):
    return {"enzyme_genesymbol": enz, "substrate_genesymbol": sub, "modification": mod,
            "residue_type": res, "residue_offset": off, "sources": ["PhosphoSite"],
            "references": ["PhosphoSite:333"]}


FAKE_OMNIPATH = {
    "interactions": [
        _i("MAP2K1", "MAPK1", stim=True),
        _i("DUSP6", "MAPK1", inh=True),
        _i("MAPK1", "MAP2K1", inh=True),          # in OmniPath, not in the model
        _i("MAPK1", "RAF1", inh=True),            # RAF1 is outside the model ...
        _i("RAF1", "MAP2K1", stim=True),          # ... and closes MAPK1 -> RAF1 -> MAP2K1
        _i("GRB2", "SOS1", stim=True),            # touches no model protein: filtered out
    ],
    "enzsub": [
        _site("MAP2K1", "MAPK1", "phosphorylation", "T", 185),
        _site("MAP2K1", "MAPK1", "phosphorylation", "Y", 187),
        _site("DUSP6", "MAPK1", "dephosphorylation", "T", 185),
        _site("MAPK1", "MAP2K1", "phosphorylation", "T", 292),
    ],
    "complexes": [],
}


@pytest.fixture
def offline(monkeypatch):
    calls = []

    def fetch(endpoint, params):
        calls.append((endpoint, params))
        return FAKE_OMNIPATH[endpoint]

    monkeypatch.setattr(oc, "_fetch", fetch)
    monkeypatch.setattr(oc, "_uniprot_genes", lambda acc: FAKE_UNIPROT[acc])
    return calls


def test_reads_four_enzyme_edges_from_markevich():
    net = oc.read_network(MARKEVICH)
    mapping = {"M": {"genes": ["MAPK1"]}, "Mp": {"genes": ["MAPK1"]}, "Mpp": {"genes": ["MAPK1"]},
               "MAPKK": {"genes": ["MAP2K1"]}, "MKP3": {"genes": ["DUSP6"]}}
    edges = {(e["source"], e["target"], e["modification"]): e["reactions"]
             for e in oc.model_edges(net, mapping)}
    # v2 also lists M as a modifier (competitive inhibition): it must not become an enzyme
    assert edges == {("MAP2K1", "MAPK1", "phosphorylation"): ["v1", "v2"],
                     ("DUSP6", "MAPK1", "dephosphorylation"): ["v3", "v4"]}


def test_xenopus_annotations_map_to_human_orthologues(offline):
    mapping = oc.map_species(oc.read_network(MARKEVICH), organism=9606)
    assert mapping["Mpp"]["genes"] == ["MAPK1"] and mapping["MKP3"]["genes"] == ["DUSP6"]
    assert "orthologue" in mapping["MAPKK"]["note"]
    mouse = oc.map_species(oc.read_network(MARKEVICH), organism=10090)
    assert mouse["MAPKK"]["genes"] == ["Map2k1"]


def test_full_check_on_markevich(offline):
    r = json.loads(oc.omnipath_network_check(str(MARKEVICH)))
    assert "error" not in r, r
    status = {(e["source"], e["target"]): e["status"] for e in r["edges"]}
    assert status[("MAP2K1", "MAPK1")] == "supported: site-level"
    assert status[("DUSP6", "MAPK1")] == "supported: site-level"
    mek = next(e for e in r["edges"] if e["source"] == "MAP2K1")
    assert mek["evidence"]["sites"] == ["T185", "Y187"]
    missing = {(m["source"], m["target"]): m for m in r["missing_internal_edges"]}
    assert ("MAPK1", "MAP2K1") in missing and missing[("MAPK1", "MAP2K1")]["closes_loop"]
    loops = {l["path"]: l["kind"] for l in r["loop_candidates"]}
    assert loops.get("MAPK1 -> RAF1 -> MAP2K1") == "feedback"
    assert all("SOS1" not in json.dumps(x) for x in r["external_regulators"])
    assert "2 of 2 model edge(s) are supported" in r["conclusion"]
    from pathlib import Path
    assert Path(r["plot_path"]).exists() and Path(r["excel_paths"]["report"]).exists()


def test_species_map_overrides_and_unmapped_model_is_explained(offline, tmp_path):
    bare = tmp_path / "bare.xml"
    bare.write_text(MARKEVICH.read_text().replace("uniprot/", "nothing/"))
    r = json.loads(oc.omnipath_network_check(str(bare)))
    assert "species_map" in r["error"]
    r = json.loads(oc.omnipath_network_check(str(bare), species_map={
        "M": "MAPK1", "Mp": "MAPK1", "Mpp": "MAPK1", "MAPKK": "MAP2K1", "MKP3": "DUSP6"}))
    assert "error" not in r and r["species_mapping"]["MKP3"]["how"] == "species_map"


def test_omnipath_down_is_an_error_not_a_conclusion(monkeypatch):
    monkeypatch.setattr(oc, "_uniprot_genes", lambda acc: FAKE_UNIPROT[acc])

    def down(endpoint, params):
        raise ConnectionError("no route")
    monkeypatch.setattr(oc, "_fetch", down)
    r = json.loads(oc.omnipath_network_check(str(MARKEVICH)))
    assert "OmniPath did not answer" in r["error"]


def test_tool_is_wired_into_the_research_sub_agent(offline):
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.guard import ExecutionGuard
    from dynalyst.orchestration.subagents import SUBAGENTS, owner_of
    assert owner_of("omnipath_network_check") == "research"
    assert "omnipath_network_check" in registry.all_tools()
    guard = ExecutionGuard()
    assert guard.check("research", "omnipath_network_check",
                       {"model_name_or_path": "Markevich2004"}).allowed
    assert not guard.check("research", "omnipath_network_check", {"model": "x"}).allowed
    out = registry.execute("omnipath_network_check", {"model_name_or_path": str(MARKEVICH)})
    assert "error" not in json.loads(out), out
    assert "omnipath_network_check" in SUBAGENTS["research"].tools
