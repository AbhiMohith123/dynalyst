"""A new instruction cannot approve and resume a previous model's analysis."""
import time

import pytest

from dynalyst.orchestration import approval
from dynalyst.orchestration.agent import Agent, RunRecord
from dynalyst.orchestration.state import SessionState

OLD = "Bifurcation analysis of BIOMD0000000035.xml in total_DA"
NEW = "Run a bifurcation analysis of BIOMD0000000150 in Binding_kf."
CANDS = [{"parameter": "total_DA", "range": [0.26, 1.5]},
         {"parameter": "k2", "range": [1, 2]}]


@pytest.mark.parametrize("reply", [
    NEW,
    "Please run a bifurcation analysis of BIOMD0000000150 in Binding_kf.",
    "Yes, run BIOMD0000000150 in Binding_kf instead.",
    "No, run BIOMD0000000150 in Binding_kf instead.",
    "Run BIOMD0000000035 in a different parameter.",
    "Run BIOMD0000000150 in total_DA.",
    "Plot total_DA against time.",
    "Use k2 from 3 to 4.",
    "Simulate model 2.",
    "k2 and total_DA",
])
def test_new_instructions_are_not_approval_replies(reply):
    assert approval.parse(reply, []) == ("other", None)
    assert approval.parse(reply, CANDS) == ("other", None)
    assert approval.parse_name(reply, CANDS) == ("other", None)
    assert approval.parse_crnt_wait(reply) == "other"


@pytest.mark.parametrize("reply,expected", [
    ("Yes, please!", ("approve", 0)),
    ("ok go ahead", ("approve", 0)),
    ("please run it", ("approve", 0)),
    ("continue the limit cycles", ("approve", 0)),
    ("no thanks", ("reject", None)),
    ("just the report", ("reject", None)),
    ("2", ("approve", 1)),
    ("option 2 please", ("approve", 1)),
    ("use k2 please", ("approve", 1)),
    ("run it in total_DA please", ("approve", 0)),
    ("select `total_DA`", ("approve", 0)),
])
def test_actual_approval_and_candidate_choices_still_work(reply, expected):
    assert approval.parse(reply, CANDS) == expected


def _pending(kind):
    proposal = {"change": "Continue the old model", "reason": "Hopf found", "steps": []}
    return {"kind": kind, "query": OLD, "created": time.time(),
            "model": "BIOMD0000000035.xml", "tool": "run_bifurcation_analysis",
            "candidates": CANDS, "other_candidates": [{"parameter": "Binding_kf"}],
            "proposal": proposal, "job_id": "old-search",
            "options": {k: proposal for k in ("yes", "no", "continue", "plot")},
            "continuation": {"plan": [], "plan_versions": [], "steps": []}}


@pytest.mark.parametrize("kind", ["cycles", "candidates", "crnt_wait"])
def test_new_request_drops_every_kind_of_pending_task(monkeypatch, kind):
    stopped = []
    monkeypatch.setattr("dynalyst.analysis.crnt_analysis.stop_crnt_job", stopped.append)
    state = SessionState(persist=False)
    state.set(pending_approval=_pending(kind))
    agent = Agent(state=state, llms={}, save_records=False, verbose=False, approval="ask")
    rec = RunRecord(query=NEW, model={})

    query, report = agent._resolve_pending(NEW, rec)

    assert query == rec.query == NEW and report is None
    assert state["pending_approval"] is None
    assert agent._approved is None and agent._resume is None and agent._cycles_choice is None
    assert rec.approval["choice"].startswith("new request")
    assert stopped == (["old-search"] if kind == "crnt_wait" else [])   # a waiting search stops


