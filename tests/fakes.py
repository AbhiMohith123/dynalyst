"""Scriptable fake chat models for testing the agent without an API key."""
import itertools
import re

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ids = itertools.count()


def call(name, **args):
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def tools(*calls, text=""):
    return AIMessage(content=text, tool_calls=list(calls))


def say(text):
    return AIMessage(content=text)


def _result(message):
    return ChatResult(generations=[ChatGeneration(message=message)])


class ScriptedChat(GenericFakeChatModel):
    """Replays AIMessages in order, then answers "done." without a tool call (a sub-agent's
    closing report); records every prompt it was given."""
    seen: list = []

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except StopIteration:
            return _result(AIMessage(content="done."))


def scripted(*messages):
    return ScriptedChat(messages=iter(messages), seen=[])


class RelayTeam(ScriptedChat):
    """A team supervisor that passes its task on unchanged: a task written "[sub_agent] text"
    goes to that sub-agent, and the sub-agents' reports come back as the team's report."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, HumanMessage):
            m = re.match(r"Task: \[(\w+)\] (.*)", str(last.content), re.S)
            if m:
                return _result(tools(call(f"delegate_to_{m.group(1)}", task=m.group(2))))
        reports = [str(x.content) for x in messages if isinstance(x, ToolMessage)]
        return _result(AIMessage(content="\n".join(reports) or "done."))


def via_teams(llms: dict) -> dict:
    """The roles for a test written as orchestrator -> sub-agent: each delegate_to_<sub_agent>
    the orchestrator's script makes goes to that sub-agent's team, which relays it."""
    from dynalyst.orchestration.teams import TEAMS, team_of
    orch = llms["orchestrator"]
    messages = []
    for m in orch.messages:
        calls = []
        for tc in getattr(m, "tool_calls", None) or []:
            sub = tc["name"][len("delegate_to_"):] if tc["name"].startswith("delegate_to_") else None
            if sub and sub not in TEAMS and team_of(sub):
                tc = call(f"delegate_to_{team_of(sub)}", task=f"[{sub}] {tc['args'].get('task', '')}")
            calls.append(tc)
        messages.append(tools(*calls, text=m.content) if calls else m)
    orch.messages = iter(messages)          # the same object, so a test can read orch.seen
    return {**llms, "team": RelayTeam(messages=iter(()), seen=[])}
