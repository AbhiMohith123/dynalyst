"""
Model picker for the notebook: choose the agent's language model from a list, then give its
API key. The key box appears only after a model is chosen, and only when that model needs a
key; a key already set in the environment is used when the box is left blank.

Usage, in a notebook cell:

    from dynalyst.model_picker import pick_model
    pick_model();

Everything it sets (model, endpoint, key) applies to this Python process only: other programs
on the machine keep their own settings, and nothing is written to a file. Choosing another
model later applies to the next request; the chat box keeps working.
"""
import os
from dataclasses import dataclass

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
# where each LangChain provider reads its key
KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "google_genai": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY", "mistralai": "MISTRAL_API_KEY"}
# the user's own OpenAI settings, before a choice of an OpenAI-compatible endpoint replaces them
_ORIGINAL_ENV = {name: os.environ.get(name) for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL")}


@dataclass(frozen=True)
class Choice:
    label: str
    model: str                    # LangChain "provider:model"; "" = typed by the user
    effort: str | None = None     # reasoning effort (OpenAI reasoning models only)
    base_url: str | None = None   # an OpenAI-compatible endpoint; None = the provider's own
    key_from: str | None = None   # environment variable that may already hold the key
    needs_key: bool = True
    key_prefix: str | None = None # what this provider's keys start with
    note: str = ""


CHOICES = [
    Choice("Nemotron 3.5 Super VL 120B - NVIDIA API (free)",
           "openai:nvidia/nemotron-3.5-super-vl-120b-a12b", base_url=NVIDIA_URL,
           key_from="NVIDIA_API_KEY", key_prefix="nvapi-",
           note="Free key from build.nvidia.com. The most reliable free model in our comparison."),
    Choice("Nemotron 3.5 Bolt E4B - NVIDIA API (free, small)",
           "openai:nvidia/nemotron-3.5-bolt-e4b", base_url=NVIDIA_URL, key_from="NVIDIA_API_KEY",
           key_prefix="nvapi-", note="Fast, but a small model: it skips analyses more often. Check its answers."),
    Choice("GPT-6 Sol - OpenAI", "openai:gpt-6-sol", effort="medium", key_from="OPENAI_API_KEY",
           key_prefix="sk-",
           note="Paid. A reasoning model, used at medium effort."),
    Choice("GPT-4o mini - OpenAI", "openai:gpt-4o-mini", key_from="OPENAI_API_KEY", key_prefix="sk-",
           note="Paid, inexpensive."),
    Choice("Claude Sonnet 5.5 - Anthropic", "anthropic:claude-sonnet-5-5",
           key_from="ANTHROPIC_API_KEY", key_prefix="sk-ant-", note="Paid. Needs an Anthropic API key."),
    # Google's "-latest" aliases follow its newest Pro / Flash model, so the IDs do not go stale
    Choice("Gemini Pro (latest) - Google", "google_genai:gemini-pro-latest",
           key_from="GOOGLE_API_KEY", key_prefix="AIza",
           note="Key from aistudio.google.com (free quota, then paid). Google's newest Pro model."),
    Choice("Gemini Flash (latest) - Google", "google_genai:gemini-flash-latest",
           key_from="GOOGLE_API_KEY", key_prefix="AIza",
           note="Key from aistudio.google.com (free quota). Faster and cheaper than Pro; "
                "check its answers."),
    Choice("Qwen3 4B instruct - local Ollama (free)", "ollama:qwen3:4b-instruct", needs_key=False,
           note="Runs on this machine: install Ollama, then `ollama pull qwen3:4b-instruct`. "
                "Slow without a large GPU, and a small model."),
    Choice("Own vLLM server (OpenAI-compatible)", "", base_url="http://localhost:8000/v1",
           needs_key=False, note="Type the model name vLLM serves and the server address. "
                                 "A key is needed only if the server checks one."),
    Choice("Other LangChain model", "", needs_key=False,
           note="Type any 'provider:model' string (install langchain-<provider> first) and its key "
                "if the provider needs one."),
]


def apply_choice(choice: Choice, key: str = "", model: str = "", base_url: str = "",
                 verify: bool = False) -> str:
    """Point every role of the agent at the chosen model, for this process only. key: the key
    typed in (blank: the one already set, if any). model / base_url: for the choices typed by
    the user. verify: send one tiny request, so a wrong key or an unavailable model shows now
    rather than at the first question. Returns a one-line status; raises ValueError when
    something is missing or the test request fails."""
    from dynalyst.agents.unified_agent import reset_agents
    from dynalyst.orchestration.llm import use_model

    model = (model or choice.model).strip()
    if not model:
        raise ValueError("Type the model name.")
    url = (base_url or choice.base_url or "").strip() or None
    if url and ":" not in model.split("/")[0]:
        model = f"openai:{model}"               # a model served at an OpenAI-compatible URL
    provider = model.split(":", 1)[0] if ":" in model else "openai"
    typed = key.strip().strip("\"'").strip()     # a key pasted with quotes or spaces
    if typed:
        key = typed
    elif choice.key_from == "OPENAI_API_KEY":   # not the endpoint key a previous choice put there
        key = _ORIGINAL_ENV["OPENAI_API_KEY"]
    elif choice.key_from:
        key = os.environ.get(choice.key_from)
    else:
        key = None
    if choice.needs_key and not key:
        where = f" (or set {choice.key_from})" if choice.key_from else ""
        raise ValueError(f"This model needs an API key: paste it in the key box{where}.")
    if key and choice.key_prefix and not key.startswith(choice.key_prefix):
        raise ValueError(f"This does not look like the right key: keys for this model start with "
                         f"'{choice.key_prefix}'. Paste only the key itself.")

    if provider == "openai":
        if url:
            os.environ["OPENAI_BASE_URL"] = url
        elif _ORIGINAL_ENV["OPENAI_BASE_URL"]:
            os.environ["OPENAI_BASE_URL"] = _ORIGINAL_ENV["OPENAI_BASE_URL"]
        else:
            os.environ.pop("OPENAI_BASE_URL", None)
        key = key or ("local" if url else None)  # a local vLLM server takes any non-empty key
        if key:
            os.environ["OPENAI_API_KEY"] = key
    elif key and provider in KEY_VARS:
        os.environ[KEY_VARS[provider]] = key
    status = use_model(model, choice.effort)
    reset_agents()                              # the next request is made with the new model
    if verify:
        _check_model()
    endpoint = os.environ.get("OPENAI_BASE_URL") if provider == "openai" else None
    return (f"{status} | endpoint: {endpoint or 'provider default'} | "
            f"key: {'set (hidden)' if key else 'none needed'}" + (" | checked" if verify else ""))


def _check_model():
    """One tiny request to the chosen model; a clear message when it fails."""
    from dynalyst.orchestration.llm import get_chat_model
    try:
        get_chat_model("orchestrator").invoke("Reply with OK")
    except Exception as exc:
        text = str(exc)
        code = next((c for c in ("401", "403", "404", "429") if c in text), None)
        reason = {"401": "the key was rejected (401). Copy the whole key again, without quotes",
                  "403": "the key was refused (403). Copy the whole key again; if it is right, "
                         "your account may not use this model, so choose another",
                  "404": "this model is not available to your account (404). Choose another",
                  "429": "rate limit reached (429). Wait a minute and try again",
                  }.get(code, f"{type(exc).__name__}: {text[:200]}")
        raise ValueError(f"The test request failed: {reason}.") from exc


def pick_model():
    """Show the model dropdown; the key box and the fields a choice needs appear once a model
    is chosen. Returns the widget."""
    import ipywidgets as widgets
    from IPython.display import display

    def wide():                              # one Layout per widget: display is set per widget
        return widgets.Layout(width="620px")

    style = {"description_width": "110px"}
    choice_box = widgets.Dropdown(options=[("- choose a model -", None)]
                                  + [(c.label, i) for i, c in enumerate(CHOICES)],
                                  value=None, description="Model:", layout=wide(), style=style)
    note = widgets.HTML()
    model_box = widgets.Text(description="Model name:", layout=wide(), style=style)
    url_box = widgets.Text(description="Server URL:", layout=wide(), style=style)
    key_box = widgets.Password(description="API key:", layout=wide(), style=style)
    use_button = widgets.Button(description="Use this model", button_style="primary",
                                icon="check", disabled=True)
    status = widgets.HTML()

    def show(widget, visible):
        widget.layout.display = None if visible else "none"

    def refresh(_change=None):
        index = choice_box.value
        use_button.disabled = index is None
        status.value = ""
        if index is None:
            note.value = ""
            for w in (model_box, url_box, key_box):
                show(w, False)
            return
        c = CHOICES[index]
        note.value = f"<i>{c.note}</i>"
        show(model_box, not c.model)
        model_box.placeholder = ("e.g. Qwen/Qwen3-32B" if c.base_url
                                 else "e.g. google_genai:gemini-2.5-pro")
        show(url_box, bool(c.base_url) and not c.model)
        url_box.value = c.base_url or ""
        existing = (_ORIGINAL_ENV["OPENAI_API_KEY"] if c.key_from == "OPENAI_API_KEY"
                    else os.environ.get(c.key_from) if c.key_from else None)
        show(key_box, c.needs_key or not c.model)
        key_box.value = ""
        key_box.placeholder = (f"found in {c.key_from}: leave blank to use it" if existing
                               else "paste your key" if c.needs_key else "only if needed")

    def on_use(_button):
        c = CHOICES[choice_box.value]
        try:
            status.value = "<i>Checking the model with a short test request...</i>"
            line = apply_choice(c, key=key_box.value, model=model_box.value,
                                base_url=url_box.value if not c.model else "", verify=True)
            status.value = f"<b>Ready.</b> {line}"
            key_box.value = ""
        except Exception as exc:
            status.value = f"<b style='color:#b00020'>{exc}</b>"

    choice_box.observe(refresh, names="value")
    use_button.on_click(on_use)
    refresh()
    box = widgets.VBox([choice_box, note, model_box, url_box, key_box, use_button, status])
    display(box)
    return box
