"""A model named by its paper ('Markevich2004') must resolve to that paper's
BioModels entry, never to a guessed ID (the agent once fetched BIOMD0000000010,
Kholodenko 2000, for 'Markevich2004'). No network: the EBI search is faked."""
import json

import pytest

from dynalyst.analysis import biomodels as bm


def _entry(i, name, author, year, curated=True):
    return {"id": f"BIOMD{i:010d}", "fields": {
        "name": [name], "curationstatus": ["Manually curated" if curated else "Non curated"],
        "modelformat": ["SBML"], "publication_title": ["t"], "publication_year": [str(year)],
        "first_author": [author], "publication_pubmed": [], "publication_doi": [],
        "last_modification_date": []}}


SEARCH = {"hitCount": 8, "entries": [
    _entry(10, "Kholodenko2000_MAPK_feedback", "Kholodenko BN", 2000),     # ranked first
    _entry(27, "Markevich2004_MAPK_orderedMM", "Markevich NI", 2004),
    _entry(26, "Markevich2004_MAPK_orderedElementary", "Markevich NI", 2004),
    _entry(28, "Markevich2004_MAPK_phosphoRandomElementary", "Markevich NI", 2004),
    _entry(29, "Markevich2004_MAPK_phosphoRandomMM", "Markevich NI", 2004),
    _entry(30, "Markevich2004_MAPK_AllRandomElementary", "Markevich NI", 2004),
    _entry(31, "Markevich2004_MAPK_orderedMM2kinases", "Markevich NI", 2004),
    _entry(1053, "Garde2020", "Garde A", 2020)]}


@pytest.fixture
def fake_search(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(bm, "_get", lambda url, params=None, **k: seen.append(params) or SEARCH)
    monkeypatch.setattr(bm.config, "LOCAL_SBML_MODELS_DIR", tmp_path)
    return seen


@pytest.mark.parametrize("name", ["Markevich2004", "Markevich 2004", "markevich et al. 2004",
                                  "the Markevich 2004 model"])
def test_a_paper_name_finds_that_papers_entries_not_the_top_hit(fake_search, name):
    out = json.loads(bm.find_biomodels_by_paper(name))
    ids = [m["id"] for m in out["models"]]
    assert "BIOMD0000000010" not in ids                       # Kholodenko, ranked first
    assert ids == [f"BIOMD{i:010d}" for i in (26, 27, 28, 29, 30, 31)]
    assert out["best"] == "BIOMD0000000026" and out["ambiguous"]
    assert fake_search[-1]["query"] == "markevich 2004"


def test_variant_words_pick_the_variant(fake_search):
    out = json.loads(bm.find_biomodels_by_paper("Markevich 2004 orderedMM"))
    assert out["best"] == "BIOMD0000000027"


def test_an_unknown_paper_returns_nothing_to_fetch(fake_search):
    out = json.loads(bm.find_biomodels_by_paper("Nobody 1999"))
    assert out["models"] == [] and "best" not in out and "Do not fetch" in out["note"]


def test_a_local_file_of_that_paper_is_reported(fake_search, tmp_path):
    (tmp_path / "Markevich2004.xml").write_text(
        '<sbml><model id="m" name="Markevich2004_MAPK_orderedMM"></model></sbml>')
    out = json.loads(bm.find_biomodels_by_paper("Markevich 2004"))
    assert out["use_as"] == "Markevich2004"            # found locally first, not searched


@pytest.mark.parametrize("text,ids", [
    ("fetch BIOMD0000000010", {"BIOMD0000000010"}),
    ("biomodel 10 please", {"BIOMD0000000010"}),
    ("BIOMD26 and BioModels #27", {"BIOMD0000000026", "BIOMD0000000027"}),
    ("Markevich2004", set()),
    ("MODEL1006230000", {"MODEL1006230000"}),
    ("feedback in model 2 of the paper, see model1.xml", set())])
def test_ids_are_read_however_they_are_written(text, ids):
    assert bm.biomodels_ids_in(text) == ids


def test_the_model_agent_owns_the_lookup():
    from dynalyst.orchestration.subagents import owner_of
    assert owner_of("find_biomodels_by_paper") == "model_search"


