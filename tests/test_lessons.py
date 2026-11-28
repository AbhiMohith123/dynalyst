"""Lessons (orchestration/lessons.py): a call that failed, then succeeded with other arguments,
is saved as data with no model, parameter name or value in it; later sub-agents are shown it;
it expires."""
import json
import time

import pytest

from dynalyst import config
from dynalyst.orchestration import lessons


@pytest.fixture(autouse=True)
def lessons_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LESSONS_PATH", tmp_path / "lessons.jsonl")
    monkeypatch.setattr(config, "LESSONS_ENABLED", True)
    return config.LESSONS_PATH


def _rejected(args, reason):
    return {"tool": "scan_tool", "arguments": args, "guard": {"allowed": False, "reason": reason}}


def _accepted(args, tool="scan_tool"):
    return {"tool": tool, "arguments": args, "guard": {"allowed": True}, "status": "ok"}


def test_a_corrected_call_that_was_accepted_leaves_a_lesson(lessons_file):
    saved = lessons.record([
        _rejected({"model_name_or_path": "modelA.xml", "param_name": "k7", "plot_state": "Zed"},
                  "'Zed' is not a state variable of modelA.xml (state 3 of 9)."),
        _accepted({"model_name_or_path": "modelA.xml", "param_name": "k7"})])
    assert len(saved) == 1
    text = lessons_file.read_text()
    for name in ("modelA", "k7", "Zed"):             # nothing of this run is stored
        assert name not in text
    entry = json.loads(text)
    assert entry["removed"] == ["plot_state"] and entry["tool"] == "scan_tool"
    assert "'X' is not a state variable of FILE" in entry["problem"]


def test_a_retry_with_nothing_changed_teaches_nothing(lessons_file):
    args = {"param_name": "a"}
    assert lessons.record([_rejected(args, "no"), _accepted(dict(args))]) == []
    assert not lessons_file.exists()


def test_only_an_accepted_retry_teaches(lessons_file):
    failed = {"tool": "scan_tool", "arguments": {"x": 2}, "guard": {"allowed": True},
              "status": "error", "error": "no points"}
    assert lessons.record([_rejected({"x": 1}, "bad"), failed]) == []


def test_the_same_lesson_is_counted_not_duplicated(lessons_file):
    for _ in range(3):
        lessons.record([_rejected({"p": 1, "q": 2}, "bad q"), _accepted({"p": 1})])
    entries = [json.loads(l) for l in lessons_file.read_text().splitlines()]
    assert len(entries) == 1 and entries[0]["count"] == 3


def test_hints_name_the_tool_and_the_argument_not_the_model(lessons_file):
    lessons.record([_rejected({"model": "m.xml", "plot_state": "Q"}, "'Q' is not a state"),
                    _accepted({"model": "m.xml"})])
    hint = lessons.for_tools(["scan_tool"])
    assert "removing the argument(s) plot_state" in hint and "m.xml" not in hint
    assert lessons.for_tools(["another_tool"]) == ""


def test_lessons_expire_and_can_be_turned_off(lessons_file, monkeypatch):
    lessons.record([_rejected({"p": 1}, "bad"), _accepted({"p": 2})])
    assert lessons.for_tools(["scan_tool"])
    entry = json.loads(lessons_file.read_text())
    entry["updated"] = time.time() - 86400 * (config.LESSONS_MAX_AGE_DAYS + 1)
    lessons_file.write_text(json.dumps(entry) + "\n")
    assert lessons.for_tools(["scan_tool"]) == ""
    monkeypatch.setattr(config, "LESSONS_ENABLED", False)
    assert lessons.record([_rejected({"p": 1}, "bad"), _accepted({"p": 2})]) == []


def test_tool_switches_and_machine_problems_teach_nothing(lessons_file):
    assert lessons.record([
        {"tool": "tool_a", "arguments": {"q": 1}, "guard": {"allowed": True},
         "status": "error", "error": "tool_a failed: No module named 'x'"},
        _accepted({"q": 1}, tool="tool_b")]) == []
    assert lessons.record([
        {"tool": "scan_tool", "arguments": {"q": 1}, "guard": {"allowed": True},
         "status": "error", "error": "No PDF matching 'x' found"},
        _accepted({"q": 2})]) == []
    assert not lessons_file.exists()


def test_a_tool_error_then_a_success_teaches_too(lessons_file):
    failed = {"tool": "scan_tool", "arguments": {"p": 1, "q": 2}, "guard": {"allowed": True},
              "status": "error", "error": "q is out of range"}
    assert len(lessons.record([failed, _accepted({"p": 1})])) == 1
