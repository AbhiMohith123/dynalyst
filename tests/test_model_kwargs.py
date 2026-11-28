"""DYNALYST_MODEL_KWARGS: extra chat-model options (vLLM's thinking switch, Ollama's
context length, ...) reach init_chat_model unchanged, for every role. No model is called."""
import langchain.chat_models

from dynalyst import config
from dynalyst.orchestration import llm


def _captured(monkeypatch, kwargs):
    seen = {}

    def fake_init(spec, **options):
        seen.update(spec=spec, **options)
        return object()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init)
    monkeypatch.setattr(config, "MODEL_KWARGS", kwargs)
    return seen


def test_model_kwargs_reach_the_chat_model(monkeypatch):
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    seen = _captured(monkeypatch, {"extra_body": body})
    llm.get_chat_model("subagent", override="openai:Qwen/Qwen3-32B")
    assert seen["spec"] == "openai:Qwen/Qwen3-32B"
    assert seen["extra_body"] == body
    assert seen["temperature"] == config.AGENT_TEMPERATURE


def test_model_kwargs_can_override_the_temperature(monkeypatch):
    seen = _captured(monkeypatch, {"temperature": 0.6, "num_ctx": 32768})
    llm.get_chat_model("orchestrator", override="ollama:qwen3:4b-instruct")
    assert seen["temperature"] == 0.6 and seen["num_ctx"] == 32768


def test_no_model_kwargs_changes_nothing(monkeypatch):
    seen = _captured(monkeypatch, {})
    llm.get_chat_model("orchestrator", override="openai:gpt-4o-mini")
    assert set(seen) <= {"spec", "callbacks", "temperature", "use_responses_api", "reasoning"}
