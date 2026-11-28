"""The agent's architecture (README.md, "How it works", has the full design):

  orchestrator  a language model in a loop (<= 14 calls) with only the teams and
                write_plan as tools: breaks every question into a plan (and revises it),
                delegates each step to a team, reads the results -- errors included --
                and answers from them (agent.py)
  teams         a team supervisor (<= 6 calls) hands its task's parts to its own
                sub-agents; parts that are independent run in parallel (teams.py)
  sub-agents    (<= 6 calls per task): read-only context tools, then
                their own analysis tools; they see only their task (subagents.py)
  guard         code: the sub-agent owns the tool, valid arguments, retry limit,
                duplicate call reused (guard.py); it never sees the user's request
  tool          the analysis itself: AUTO-07p, jaxkineticmodel, ... (tools.py).
                Before it runs: file safety (safety.py) and the tool's own input
                checks (inputs.py); a refusal comes back as the tool's error
  reviewer      code: re-checks every result from the model's own equations; accept,
                warn, or send back with a correction (<= 3 times) (reviewer.py)
  final check   code: every number, model ID and paper title in the answer appears
                in a tool result; one rewrite, then removal (final_check.py)
"""
