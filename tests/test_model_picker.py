"""The notebook's model picker: each choice points every role at its model, endpoint and key
for this process only, and switching between choices leaves nothing stale behind (an
endpoint, a key, or OpenAI's Responses API). No model is called."""
import pytest

from dynalyst import config, model_picker as mp

CONFIG_NAMES = ("AGENT_MODEL", "ORCHESTRATOR_MODEL", "TEAM_MODEL",
                "SUBAGENT_MODEL", "REASONING_EFFORT", "AGENT_TEMPERATURE",
                "RESPONSES_API")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in CONFIG_NAMES:                       # restored after each test
        monkeypatch.setattr(config, name, getattr(config, name))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "NVIDIA_API_KEY", "ANTHROPIC_API_KEY",
                 "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mp, "_ORIGINAL_ENV", {"OPENAI_API_KEY": "sk-own", "OPENAI_BASE_URL": None})


def choice(label_start):
    return next(c for c in mp.CHOICES if c.label.startswith(label_start))


def test_nvidia_choice_uses_its_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    line = mp.apply_choice(choice("Nemotron 3.5 Super VL"))
    assert config.AGENT_MODEL == "openai:nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert mp.os.environ["OPENAI_BASE_URL"] == mp.NVIDIA_URL
    assert mp.os.environ["OPENAI_API_KEY"] == "nvapi-x"
    assert config.RESPONSES_API is False and "nvapi" not in line


def test_switching_restores_openai_and_clears_the_reasoning_settings(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    mp.apply_choice(choice("Nemotron 3.5 Super VL"))
    mp.apply_choice(choice("GPT-6 Sol"))            # back to OpenAI itself, a reasoning model
    assert "OPENAI_BASE_URL" not in mp.os.environ
    assert mp.os.environ["OPENAI_API_KEY"] == "sk-own"   # the user's key, not NVIDIA's
    assert config.RESPONSES_API is True and config.REASONING_EFFORT == "medium"
    mp.apply_choice(choice("Nemotron 3.5 Super VL"))     # and to NVIDIA again
    assert config.RESPONSES_API is False and config.REASONING_EFFORT == ""
    assert config.AGENT_TEMPERATURE is not None


def test_a_typed_key_wins_and_a_missing_key_is_refused(monkeypatch):
    mp.apply_choice(choice("Claude Sonnet"), key=" sk-ant-typed ")
    assert mp.os.environ["ANTHROPIC_API_KEY"] == "sk-ant-typed"
    with pytest.raises(ValueError, match="needs an API key"):
        mp.apply_choice(choice("Nemotron 3.5 Bolt"))   # NVIDIA_API_KEY not set, none typed


def test_google_choice_sets_gemini_and_its_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-own")
    mp.apply_choice(choice("Gemini Pro"))
    assert config.AGENT_MODEL == "google_genai:gemini-pro-latest"
    assert mp.os.environ["GOOGLE_API_KEY"] == "AIza-own"
    with pytest.raises(ValueError, match="start with 'AIza'"):
        mp.apply_choice(choice("Gemini Flash"), key="sk-not-a-google-key")


def test_a_pasted_key_is_cleaned_and_a_wrong_one_refused():
    mp.apply_choice(choice("Nemotron 3.5 Super VL"), key=' "nvapi-abc" ')
    assert mp.os.environ["OPENAI_API_KEY"] == "nvapi-abc"
    with pytest.raises(ValueError, match="start with 'nvapi-'"):
        mp.apply_choice(choice("Nemotron 3.5 Super VL"), key="sk-not-an-nvidia-key")


def test_the_test_request_explains_a_rejected_key(monkeypatch):
    import dynalyst.orchestration.llm as llm

    class Rejecting:
        def invoke(self, _prompt):
            raise RuntimeError("Error code: 401 - Unauthorized")

    monkeypatch.setattr(llm, "get_chat_model", lambda role, *a, **k: Rejecting())
    with pytest.raises(ValueError, match=r"key was rejected \(401\)"):
        mp.apply_choice(choice("Nemotron 3.5 Super VL"), key="nvapi-wrong", verify=True)

    class Answering:
        def invoke(self, _prompt):
            return "OK"

    monkeypatch.setattr(llm, "get_chat_model", lambda role, *a, **k: Answering())
    assert "checked" in mp.apply_choice(choice("Nemotron 3.5 Super VL"), key="nvapi-ok", verify=True)


def test_local_choices_need_no_key():
    line = mp.apply_choice(choice("Qwen3 4B instruct"))
    assert config.AGENT_MODEL == "ollama:qwen3:4b-instruct" and "none needed" in line
    mp.apply_choice(choice("Own vLLM server"), model="Qwen/Qwen3-32B",
                    base_url="http://gpu-server:8000/v1")
    assert config.AGENT_MODEL == "openai:Qwen/Qwen3-32B"
    assert mp.os.environ["OPENAI_BASE_URL"] == "http://gpu-server:8000/v1"
    assert mp.os.environ["OPENAI_API_KEY"] == "local"


def test_a_typed_choice_needs_a_model_name():
    with pytest.raises(ValueError, match="model name"):
        mp.apply_choice(choice("Other LangChain model"))


def test_the_key_box_appears_only_after_a_choice(monkeypatch):
    pytest.importorskip("ipywidgets")
    monkeypatch.setattr("IPython.display.display", lambda *a, **k: None)
    box = mp.pick_model()
    dropdown, _note, model_box, url_box, key_box, use_button, _status = box.children
    assert dropdown.value is None and use_button.disabled
    assert key_box.layout.display == "none" and model_box.layout.display == "none"
    dropdown.value = 0                               # Nemotron Super VL: needs a key
    assert key_box.layout.display is None and not use_button.disabled
    assert model_box.layout.display == "none" and url_box.layout.display == "none"
    dropdown.value = next(i for i, c in enumerate(mp.CHOICES) if c.label.startswith("Own vLLM"))
    assert model_box.layout.display is None and url_box.layout.display is None
