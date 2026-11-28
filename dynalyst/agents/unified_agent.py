"""The agent: one entry point over every capability, built on the
architecture in dynalyst/orchestration/ --

  orchestrator (plans, delegates, answers) -> team supervisors -> sub-agents
  -> execution guard -> analysis tool -> final check.

The chat model comes from config.AGENT_MODEL ("openai:gpt-4o-mini" by
default; any LangChain "provider:model" string, or DYNALYST_MODEL).

    from dynalyst.agents.unified_agent import ask, run
    print(ask("Run a bifurcation analysis of model.xml"))
    record = run("...")        # full record: plan versions, every tool call

approval: "auto" (default, called from code) or "ask" (a person is chatting: the
run pauses before AUTO-07p when its call names no continuation parameter). The chat
entry points (cli.py, notebook.ask, chat_ui.py) pass "ask".
"""
from dynalyst.orchestration.agent import Agent, RunRecord

_AGENTS = {}


def reset_agents() -> None:
    """Discard cached conversation state, including pending requests and counters."""
    _AGENTS.clear()


def get_agent(model: str | None = None, verbose: bool = True, approval: str = "auto") -> Agent:
    key = (model, verbose, approval)
    if key not in _AGENTS:
        _AGENTS[key] = Agent(model=model, verbose=verbose, approval=approval)
    return _AGENTS[key]


def run(user_message: str, conversation_history: list | None = None, verbose: bool = True,
        model: str | None = None, approval: str = "auto") -> RunRecord:
    return get_agent(model, verbose, approval).run(user_message, conversation_history)


def ask(user_message: str, conversation_history: list | None = None, verbose: bool = True,
        model: str | None = None, approval: str = "auto") -> str:
    """Same call shape as the older agents' ask(): returns the answer text."""
    return run(user_message, conversation_history, verbose, model, approval).final_answer


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python -m dynalyst.agents.unified_agent <request>")
    print(ask(" ".join(sys.argv[1:])))
