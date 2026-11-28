"""A model is named by what THIS query's results say about it: its file and the
BioModels ID the file itself declares. Reproduces the bug where BIOMD0000000010
(Kholodenko 2000), fetched earlier in the chat, was attached to Markevich2004
in every later answer."""
import json
import shutil

from langchain_core.messages import AIMessage

from dynalyst import config
from dynalyst.analysis.annotations import model_identity
from dynalyst.orchestration.final_check import (model_identities, strip_model_ids,
                                                unsupported_model_ids)
from dynalyst.regime import model_source

MARKEVICH_RESULT = json.dumps({"steady_states": [], "model_identity": {
    "file": "Markevich2004.xml", "model_name": "Markevich2004 - MAPK", "biomodels_id": "BIOMD0000000027"}},
    indent=2)
FETCH_RESULT = json.dumps({"model_id": "BIOMD0000000010", "saved_as": "data/sbml_models/BIOMD0000000010.xml",
                           "name": "Kholodenko2000 - MAPK feedback"})


def test_identity_is_read_from_the_file():
    assert model_identity("Markevich2004")["biomodels_id"] == "BIOMD0000000027"
    assert model_identity("Garde2020")["biomodels_id"] == "BIOMD0000001053"
    none = model_identity("pitchfork_model.xml")
    assert none["file"] == "pitchfork_model.xml" and none["biomodels_id"] is None
    assert model_identity("Full_model_for_SN.txt") is None             # XPP: not SBML


def test_a_fetched_file_is_known_by_the_id_it_was_saved_under(tmp_path):
    path = tmp_path / "BIOMD0000000010.xml"
    shutil.copy(config.MODELS_DIR / "pitchfork_model.xml", path)       # no annotation inside
    assert model_identity(str(path))["biomodels_id"] == "BIOMD0000000010"


def test_identity_lines_and_their_parsing():
    line = model_source.identity_line("Markevich2004")
    assert "BIOMD0000000027" in line and "Markevich2004.xml" in line
    assert "no BioModels ID" in model_source.identity_line("pitchfork_model.xml")
    assert "no BioModels ID" in model_source.identity_line("pdp_mm_bistable.txt")
    ids = model_identities([MARKEVICH_RESULT, line, model_source.identity_line("pitchfork_model.xml")])
    assert ids == {"markevich2004": "BIOMD0000000027", "pitchfork_model": None}


def test_a_borrowed_id_is_flagged():
    evidence = ["Find the steady states of Markevich2004", MARKEVICH_RESULT]
    ids = model_identities(evidence)
    bad = unsupported_model_ids("The Markevich2004 model (BIOMD0000000010) has 3 steady states.",
                                evidence, ids)
    assert [b["id"] for b in bad] == ["BIOMD0000000010"] and "no result" in bad[0]["reason"]
    assert unsupported_model_ids("Markevich2004 (BIOMD0000000027) has 3 steady states.",
                                 evidence, ids) == []


def test_a_wrong_pairing_is_flagged_even_when_both_ids_are_in_this_querys_results():
    evidence = ["fetch BIOMD0000000010 and analyse Markevich2004", FETCH_RESULT, MARKEVICH_RESULT]
    ids = model_identities(evidence)
    bad = unsupported_model_ids("Markevich2004 (BIOMD0000000010) is bistable.", evidence, ids)
    assert bad and bad[0]["model"] == "markevich2004" and "BIOMD0000000027" in bad[0]["reason"]
    ok = "BIOMD0000000010 (Kholodenko2000) was saved. Markevich2004 (BIOMD0000000027) is bistable."
    assert unsupported_model_ids(ok, evidence, ids) == []


def test_ids_are_stripped_cleanly():
    text = "The Markevich2004 model (BIOMD0000000010) is bistable; see BIOMD0000000010 too."
    assert strip_model_ids(text, ["BIOMD0000000010"]) == \
        "The Markevich2004 model is bistable; see too."


def test_every_json_analysis_result_carries_the_model_identity():
    from dynalyst.orchestration import tools as registry
    t = registry.all_tools()
    d = json.loads(t["find_steady_states"].invoke({"model_name_or_path": "Markevich2004"}))
    assert d["model_identity"]["biomodels_id"] == "BIOMD0000000027"


class _Rewriter:
    """A chat model that answers the rewrite request with a fixed text."""

    def __init__(self, text):
        self.text, self.calls = text, 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.text)


def _run_final_check(answer, rewrite, steps, history):
    from dynalyst.orchestration.agent import Agent, RunRecord
    llm = _Rewriter(rewrite)
    agent = Agent(llms={"orchestrator": llm, "subagent": llm}, verbose=False,
                  save_records=False)
    rec = RunRecord(query="Find the steady states of Markevich2004", model={})
    rec.steps, rec.final_answer = steps, answer
    agent._final_check(rec.query, history, rec)
    return rec, llm


HISTORY = [{"role": "user", "content": "fetch BIOMD0000000010"},
           {"role": "assistant", "content": "Saved BIOMD0000000010 (Kholodenko2000)."}]
STEP = [{"kind": "delegation", "tool": "find_steady_states", "output": MARKEVICH_RESULT,
         "arguments": {"model_name_or_path": "Markevich2004"}, "result": ""}]


def test_the_agent_rewrites_a_borrowed_id_and_keeps_a_correct_rewrite():
    rec, llm = _run_final_check("Markevich2004 (BIOMD0000000010): no steady states found.",
                                "Markevich2004 (BIOMD0000000027): no steady states found.",
                                STEP, HISTORY)
    assert llm.calls == 1 and rec.final_answer.startswith("Markevich2004 (BIOMD0000000027)")
    assert rec.final_check["model_ids_removed"] == []


def test_the_agent_removes_the_id_when_the_rewrite_keeps_it():
    rec, _ = _run_final_check("Markevich2004 (BIOMD0000000010): no steady states found.",
                              "Markevich2004 (BIOMD0000000010): no steady states found.",
                              STEP, HISTORY)
    assert "BIOMD0000000010" not in rec.final_answer.split("_Note")[0]
    assert rec.final_answer.startswith("Markevich2004: no steady states found.")
    assert rec.final_check["model_ids_removed"] == ["BIOMD0000000010"]


def test_a_conversational_answer_may_recall_an_id_from_history():
    rec, llm = _run_final_check("Earlier you fetched BIOMD0000000010.", "unused", [], HISTORY)
    assert llm.calls == 0 and rec.final_check["unsupported_model_ids"] == []
