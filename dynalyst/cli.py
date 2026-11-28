"""
Interactive command-line interface (formerly query_cli.py).

Usage:
    export OPENAI_API_KEY=sk-...
    python -m dynalyst.cli                     # the agent (planner -> sub-agents -> reviewer)
    python -m dynalyst.cli --agent research    # the older RAG + analysis agent (was query_cli.py)
"""
import argparse
import os
import sys

from dynalyst import config


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--agent", choices=["unified", "research"], default="unified",
                    help="unified: the agent (default); research: the older RAG + analysis agent")
    ap.add_argument("--model", default=None,
                    help="LangChain 'provider:model' for the unified agent, e.g. "
                         "openai:gpt-4o-mini (default: config.AGENT_MODEL)")
    args = ap.parse_args()
    if args.agent == "unified":
        from functools import partial
        from dynalyst.agents.unified_agent import ask
        ask = partial(ask, model=args.model, approval="ask")    # a person is chatting
    else:
        from dynalyst.agents.research_agent import ask

    if args.agent == "unified":
        from dynalyst.orchestration.llm import model_spec
        provider = model_spec("orchestrator", args.model).split(":", 1)[0]
    else:
        provider = config.LLM_BACKEND
    key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key_var and not os.environ.get(key_var):
        print(f"ERROR: set {key_var} before running (export {key_var}=...)")
        sys.exit(1)

    print(f"dynalyst ({args.agent} agent)")
    print("Examples:")
    print("  - Run a bifurcation analysis of <model file>")
    print("  - What is the dimension of <model file> and what bifurcations can it have?")
    print("  - Simulate <model file> and summarize the dynamics")
    print("  - Fit two parameters of <model file> to <data.csv>")
    print("Type 'exit' to quit.\n")

    history = []
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        answer = ask(user_input, conversation_history=history)
        print("\n" + answer)

        # keep a simple running history for follow-up questions
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        # trim history to avoid unbounded context growth
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
