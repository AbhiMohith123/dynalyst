"""Chat models for the agent's roles, from LangChain "provider:model" strings
(config.AGENT_MODEL, default "openai:gpt-4o-mini"). Any provider LangChain
supports works once its langchain-<provider> package is installed."""
from dynalyst import config

ROLES = ("orchestrator", "team", "subagent")


def model_spec(role: str, override: str | None = None) -> str:
    per_role = {"orchestrator": config.ORCHESTRATOR_MODEL, "team": config.TEAM_MODEL,
                "subagent": config.SUBAGENT_MODEL}[role]
    spec = override or per_role or config.AGENT_MODEL
    return spec if ":" in spec else f"openai:{spec}"


def reasoning_effort(role: str) -> str | None:
    """config.REASONING_EFFORT for this role: 'low' for all, or 'role=low,...' per role."""
    text = (config.REASONING_EFFORT or "").strip()
    if not text:
        return None
    if "=" not in text:
        return text
    pairs = dict(part.split("=", 1) for part in text.split(",") if "=" in part)
    return (pairs.get(role) or "").strip() or None


EFFORTS = ("low", "medium", "high")
# the configured temperature and API, restored when use_model switches back to a model
# without reasoning settings
_DEFAULT_TEMPERATURE, _DEFAULT_RESPONSES_API = config.AGENT_TEMPERATURE, config.RESPONSES_API


def use_model(model: str, effort: str | None = None) -> str:
    """Choose the chat model of every role, and one reasoning effort for all of them.
    model: a LangChain "provider:model" string. effort: "low", "medium" or "high" for a
    reasoning model (its own default temperature, and OpenAI's Responses API for an
    "openai:" model); None for a model without reasoning settings. Applies to agents
    created after the call (a chat started before keeps its models)."""
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS} or None, not {effort!r}")
    config.AGENT_MODEL = model
    for name in ("ORCHESTRATOR_MODEL", "TEAM_MODEL", "SUBAGENT_MODEL"):
        setattr(config, name, None)        # the same model for every role
    config.REASONING_EFFORT = effort or ""
    if effort:
        config.AGENT_TEMPERATURE = None    # reasoning models take only their own default
        config.RESPONSES_API = model_spec("orchestrator").startswith("openai:")
    else:                                  # e.g. after a reasoning model: back to the defaults
        config.AGENT_TEMPERATURE = _DEFAULT_TEMPERATURE
        config.RESPONSES_API = _DEFAULT_RESPONSES_API
    return f"model: {model_spec('orchestrator')} | reasoning effort: {effort or 'not set'}"


def get_chat_model(role: str, override: str | None = None, on_usage=None):
    """The chat model for a role. on_usage(role, usage_metadata) is called after every
    call with the provider's token counts (input, output, and input read from the
    provider's prompt cache), for the run record."""
    from langchain.chat_models import init_chat_model
    callbacks = [_UsageCounter(role, on_usage)] if on_usage else None
    spec = model_spec(role, override)
    options = {}
    if config.AGENT_TEMPERATURE is not None:     # None: the model's own default
        options["temperature"] = config.AGENT_TEMPERATURE
    if config.RESPONSES_API and spec.startswith("openai:"):
        options["use_responses_api"] = True
        effort = reasoning_effort(role)
        if effort:
            options["reasoning"] = {"effort": effort}
    options.update(config.MODEL_KWARGS)          # DYNALYST_MODEL_KWARGS, passed unchanged
    return init_chat_model(spec, callbacks=callbacks, **options)


def _usage_counter_base():
    from langchain_core.callbacks import BaseCallbackHandler
    return BaseCallbackHandler


class _UsageCounter(_usage_counter_base()):
    def __init__(self, role, sink):
        super().__init__()
        self.role, self.sink = role, sink

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for g in generations:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if usage:
                    self.sink(self.role, usage)
