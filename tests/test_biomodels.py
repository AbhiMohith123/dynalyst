"""BioModels search / info / article lookup (needs network to EBI; the file
download needs biomodels.org and is not tested here)."""
import json

import pytest
import requests


def _online():
    try:
        requests.get("https://www.ebi.ac.uk/ebisearch/ws/rest/biomodels?query=x&size=1&format=json",
                     timeout=10)
        return True
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _online(), reason="EBI Search not reachable")


def test_search_and_info():
    from dynalyst.analysis.biomodels import get_biomodels_info, search_biomodels
    d = json.loads(search_biomodels("Markevich MAPK bistability", 5, curated_only=True))
    assert any(r["id"] == "BIOMD0000000027" for r in d["results"])
    info = json.loads(get_biomodels_info("BIOMD0000000027"))
    assert info["article"]["pubmed"] == "14744999" and info["description"]


MARKEVICH = {f"BIOMD00000000{n}" for n in range(26, 32)}  # one article, six model variants


@pytest.mark.parametrize("pdf, models", [("garde.pdf", {"BIOMD0000001053"}),
                                         ("elowitz.pdf", {"BIOMD0000000012"}),
                                         ("Markevich.pdf", MARKEVICH)])
def test_article_to_model(pdf, models):
    from tests.conftest import need_pdf
    need_pdf(pdf)
    from dynalyst.analysis.biomodels import find_models_for_article
    ids = {m["id"] for m in json.loads(find_models_for_article(pdf, 10))["models"]}
    assert ids and ids <= models


def test_cited_paper_is_not_the_article():
    """Hass 2017 cites Chassagnole 2002 (DOI and full title in its references)."""
    from tests.conftest import need_pdf
    need_pdf("Hass.pdf")
    from dynalyst.analysis.biomodels import find_models_for_article
    ids = [m["id"] for m in json.loads(find_models_for_article("Hass.pdf", 10))["models"]]
    assert "BIOMD0000000051" not in ids


def test_name_lookup_prefers_exact_names():
    from tests.conftest import need_pdf
    need_pdf("Hass.pdf")
    from dynalyst.analysis.generic_sbml import resolve_sbml_path
    from dynalyst.analysis.pdf_tools import _resolve_pdf_path
    assert _resolve_pdf_path("Hass.pdf").name == "Hass.pdf"
    assert resolve_sbml_path("Full_model_for_SN").name == "Full_model_for_SN.xml"
