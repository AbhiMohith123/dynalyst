"""PubMed / UniProt tools, the self-updating model knowledge base, and the
final check on invented PMIDs / accessions. No network: lookups are faked."""
import json

import numpy as np
import pytest

from dynalyst import config


def test_invented_pmids_and_accessions_are_caught_real_ones_kept():
    from dynalyst.orchestration.final_check import strip_model_ids, unsupported_reference_ids
    evidence = ['{"pmid": "14744999", "proteins": [{"accession": "P26696"}]}']
    answer = ("Markevich2004 comes from PMID 14744999 and PMID: 99999999; M is UniProt P26696, "
              "MAPKK is UniProt Q99999 (https://pubmed.ncbi.nlm.nih.gov/12345678/).")
    bad = unsupported_reference_ids(answer, evidence)
    assert sorted(b["id"] for b in bad) == ["PMID: 99999999", "UniProt Q99999",
                                             "pubmed.ncbi.nlm.nih.gov/12345678"]
    cleaned = strip_model_ids(answer, [b["id"] for b in bad])
    assert "14744999" in cleaned and "P26696" in cleaned and "99999999" not in cleaned
    assert unsupported_reference_ids("km7 = 0.16, k4cat 15", evidence) == []


def test_reference_parsing_and_bad_input():
    from dynalyst.analysis import literature as L
    assert L._pmid_for("PMID: 14744999") == ("14744999", None)
    assert L._pmid_for("not a ref")[0] is None
    assert "not a UniProt accession" in json.loads(L.get_uniprot_entry("XYZ"))["error"]


def test_model_proteins_resolves_the_models_own_annotations(monkeypatch):
    from dynalyst.analysis import literature as L
    monkeypatch.setattr(L, "uniprot_record", lambda acc: {"accession": acc, "protein": f"prot {acc}",
                                                          "genes": ["G"], "organism": "X"})
    out = json.loads(L.model_proteins("Markevich2004"))
    by_acc = {p["accession"]: p for p in out["proteins"]}
    assert set(by_acc["P26696"]["species"]) == {"M", "Mp", "Mpp"}
    assert out["article_pmids"] == ["14744999"]


def test_the_tools_are_registered_and_routed():
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.subagents import SUBAGENTS
    model_agent = {"find_models_for_pubmed", "get_uniprot_entry", "search_uniprot",
                   "model_proteins", "add_model_to_knowledge_base"}
    assert model_agent | {"get_pubmed_article"} <= set(registry.all_tools())
    assert model_agent <= set(SUBAGENTS["model_search"].objective_tools)
    assert "get_pubmed_article" in SUBAGENTS["research"].objective_tools   # papers: research
    assert "config.COLLECTION_MODELS" in __import__("inspect").getsource(
        __import__("dynalyst.rag.retrieve", fromlist=["retrieve"]).retrieve)


@pytest.fixture
def scratch_kb(tmp_path, monkeypatch):
    """A throwaway knowledge base with a tiny stand-in embedder."""
    from dynalyst.rag import model_index as MI
    from dynalyst.rag import retrieve as R
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "BM25_INDEX_PATH", tmp_path / "chroma" / "bm25_index.pkl")
    monkeypatch.setattr(MI, "CHUNKS_FILE", tmp_path / "chunks_models.jsonl")

    class Embed:
        def encode(self, texts, **kw):
            v = np.array([[t.lower().count(w) for w in ("mapk", "erk", "kinase", "markevich")] + [1.0]
                          for t in texts], dtype=float)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
    for f in (R._get_chroma_client, R._get_bm25, R._get_embed_model):
        f.cache_clear()
    monkeypatch.setattr(R, "_get_embed_model", lambda device=None: Embed())
    monkeypatch.setattr("dynalyst.analysis.literature.uniprot_record",
                        lambda acc: {"accession": acc, "protein": "Mitogen-activated protein kinase 1",
                                     "genes": ["MAPK1"], "organism": "Rattus norvegicus"})
    monkeypatch.setattr("dynalyst.analysis.literature.pubmed_record",
                        lambda pmid: {"title": "Signaling switches and bistability", "authors": ["A"],
                                      "journal": "J Cell Biol", "year": "2004", "doi": "10.1/x",
                                      "abstract": "MAPK cascades can be bistable."})
    yield MI
    R._get_chroma_client.cache_clear()
    R._get_bm25.cache_clear()


def test_a_saved_model_is_searchable_at_once_and_reindexing_replaces_it(scratch_kb):
    from dynalyst.rag import retrieve as R
    res = json.loads(scratch_kb.index_model("Markevich2004"))
    assert res["knowledge_base"] == "updated" and {"overview", "article"} <= set(res["sections"])
    hits = R.retrieve("Which protein is M in Markevich2004?", collections=[config.COLLECTION_MODELS],
                      use_reranker=False)
    assert hits and hits[0]["metadata"]["source_name"] == "Markevich2004.xml"
    assert any("Mitogen-activated protein kinase 1" in h["text"] for h in hits)
    n = res["chunks"]
    scratch_kb.index_model("Markevich2004")                  # saved again: replaced, not doubled
    lines = scratch_kb.CHUNKS_FILE.read_text().splitlines()
    assert len(lines) == n
    assert len(R._get_bm25()["ids"]) == n


def test_saving_tools_add_the_model_automatically(monkeypatch):
    from dynalyst.orchestration import tools as registry
    monkeypatch.setattr(config, "AUTO_INDEX_MODELS", True)
    seen = []
    monkeypatch.setattr("dynalyst.rag.model_index.auto_index",
                        lambda p: seen.append(p) or {"knowledge_base": "updated"})
    out = registry._index_saved_model("fetch_biomodels_model",
                                      json.dumps({"saved_as": "data/sbml_models/BIOMD0000000010.xml"}))
    assert json.loads(out)["knowledge_base"] == "updated" and seen
    err = json.dumps({"error": "Could not download"})
    assert registry._index_saved_model("fetch_biomodels_model", err) == err
    assert registry._index_saved_model("simulate", "{}") == "{}"


def test_a_paper_europe_pmc_lacks_is_found_through_crossref(monkeypatch):
    from dynalyst.analysis import literature as L
    other = {"resultList": {"result": [{"title": "Autophagy in breast cancer cells", "pmid": "1",
                                        "doi": "10.1/x", "source": "MED", "id": "1"}]},
             "hitCount": 200}
    monkeypatch.setattr(L, "_get", lambda *a, **k: other)
    paper = {"title": "ODE-based models of signaling networks in autophagy", "authors": "Galhuber M",
             "journal": "Current Opinion in Systems Biology", "year": "2024", "pmid": None,
             "pmcid": None, "doi": "10.1016/j.coisb.2024.100519", "open_access": None,
             "source": "Crossref", "link": "https://doi.org/10.1016/j.coisb.2024.100519"}
    monkeypatch.setattr(L, "_crossref", lambda query, rows: [paper])
    out = json.loads(L.search_literature("ODE based models of signaling networks in autophagy"))
    assert out["results"][0]["doi"] == "10.1016/j.coisb.2024.100519" and "crossref_note" in out
    # a good Europe PMC title match: Crossref is not asked
    monkeypatch.setattr(L, "_crossref", lambda query, rows: pytest.fail("Crossref asked"))
    out = json.loads(L.search_literature("autophagy breast cancer cells"))
    assert out["results"][0]["source"] == "Europe PMC" and "crossref_note" not in out


def test_an_open_access_paper_and_its_supplement_are_downloaded(monkeypatch, tmp_path):
    from dynalyst.analysis import literature as L
    from dynalyst.orchestration import tools as registry
    from dynalyst.orchestration.subagents import SUBAGENTS
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PAPERS_DIR", tmp_path / "data" / "papers")
    monkeypatch.setattr(L, "_get", lambda *a, **k: {"resultList": {"result": [{"pmcid": "PMC5874525"}]}})
    base = "s3://pmc-oa-opendata/PMC5874525.1/"
    meta = {"pmid": 29510589, "doi": "10.3390/antiox7030039", "title": "NRF2", "license_code": "CC BY",
            "pdf_url": base + "PMC5874525.1.pdf?md5=a",
            "media_urls": [base + "antioxidants-07-00039-g001.jpg?md5=b",
                           base + "antioxidants-07-00039-s001.pdf?md5=c"]}

    class R:
        def __init__(self, code, body=b"", js=None):
            self.status_code, self.content, self._js = code, body, js
        def json(self):
            return self._js
        def raise_for_status(self):
            assert self.status_code == 200

    def get(url, timeout=None):
        if url.endswith("PMC5874525.1.json"):
            return R(200, js=meta)
        if url.endswith(".json"):
            return R(404)
        assert not url.endswith(".jpg"), "figures are not downloaded"
        return R(200, b"%PDF-1.4 " + url.encode())
    monkeypatch.setattr(L.requests, "get", get)
    out = json.loads(L.fetch_open_access_paper("29510589"))
    assert out["pmcid"] == "PMC5874525" and out["license"] == "CC BY"
    assert [f["kind"] for f in out["files"]] == ["article", "supplement"]
    assert (tmp_path / "data" / "papers" / "PMC5874525_antioxidants-07-00039-s001.pdf").exists()
    # the files are named by PMCID: the PDF list says which paper each one is
    from dynalyst.analysis.pdf_tools import list_available_pdfs
    titles = json.loads(list_available_pdfs())["titles"]
    assert set(titles.values()) == {"NRF2"} and len(titles) == 2
    assert "not a PMCID" in json.loads(L.fetch_open_access_paper("no such paper"))["error"]
    assert "fetch_open_access_paper" in registry.all_tools()
    assert "fetch_open_access_paper" in SUBAGENTS["research"].objective_tools
