"""The older single-purpose agent's tool-use loop (research) and the
notebook/CLI entry points, driven by fake LLMs (no API key, no network).
The main agent's architecture is tested in test_orchestrated_agent.py."""
import io
import json
import contextlib
from types import SimpleNamespace

import pytest

from dynalyst import config


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# --- fake OpenAI client (research + unified agents) -------------------------

class _FakeOpenAI:
    """Replays a script of responses; records every request."""
    script = []
    requests = []

    def __init__(self, api_key=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @classmethod
    def _create(cls, **kwargs):
        cls.requests.append(kwargs)
        finish_reason, content, calls = cls.script.pop(0)
        tool_calls = [SimpleNamespace(id=f"call_{i}", type="function",
                                      function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                      for i, (n, a) in enumerate(calls)] or None
        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=msg)])


@pytest.fixture
def fake_openai(monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(config, "LLM_BACKEND", "openai")
    _FakeOpenAI.requests = []
    return _FakeOpenAI


def _tool_messages(request):
    return {m["tool_call_id"]: m["content"] for m in request["messages"] if m.get("role") == "tool"}






def test_research_agent_keeps_its_own_tools(fake_openai):
    from dynalyst.agents.research_agent import ask
    fake_openai.script = [
        ("tool_calls", None, [("list_available_models", {})]),
        ("stop", "Here are the models.", []),
    ]
    assert _quiet(ask, "what models are available?") == "Here are the models."
    sent = {t["function"]["name"] for t in fake_openai.requests[0]["tools"]}
    assert "sbml_models_by_directory" in _tool_messages(fake_openai.requests[1])["call_0"]




def test_cli_help():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "dynalyst.cli", "--help"], capture_output=True,
                         text=True, cwd=str(config.PROJECT_ROOT))
    assert out.returncode == 0 and "--agent" in out.stdout


def test_notebook_ask_uses_the_agent(monkeypatch):
    from dynalyst import notebook
    from dynalyst.agents import unified_agent
    from dynalyst.orchestration.agent import Agent
    from tests.fakes import call, say, scripted, tools, via_teams
    fake = Agent(llms=via_teams({"orchestrator": scripted(say("No analysis needed.")),
                                 "subagent": scripted()}), save_records=False)
    # a person works in the notebook: it uses the agent that asks before AUTO
    monkeypatch.setitem(unified_agent._AGENTS, (None, True, "ask"), fake)
    assert _quiet(notebook.ask, "hello") == "No analysis needed."
    with pytest.raises(ValueError):
        notebook.ask("hello", agent="nope")
