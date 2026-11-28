"""Execution guard (deterministic, no language model): checks an analysis-tool call a
sub-agent selected, before it runs. It never sees the user's request: whether a call
answers the request is the agents' judgement, and a call that cannot work fails in the
tool itself (tools.input_problem), which reports why.

  permission  the sub-agent must own the tool, and the arguments must fit its schema
  retry limit a tool that failed config.MAX_TOOL_RETRIES + 1 times in this request is not
              run again; the same failed call is never repeated
  duplicate   the same call already ran in this request -> its result is reused; the same model
              and parameter drawn config.MAX_DIAGRAMS_PER_PARAMETER times -> no further diagram
"""
import json
from dataclasses import dataclass, field

from dynalyst import config
from dynalyst.orchestration import tools as registry
from dynalyst.orchestration.subagents import SUBAGENTS


@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""
    duplicate_of: dict | None = None     # the earlier step, when a call is a repeat

    def as_dict(self):
        return {"allowed": self.allowed, "reason": self.reason,
                "duplicate": self.duplicate_of is not None}


def _model_of(args: dict) -> str:
    """The model a call names: Markevich2004 and Markevich2004.xml are the same model."""
    return str(args.get("model_name_or_path") or args.get("model") or "").removesuffix(".xml")


def _diagram_key(args: dict) -> tuple:
    """Diagrams count per start state: from another state a diagram is another branch, not a repeat."""
    return _model_of(args), args["param_name"], json.dumps(args.get("initial_conditions"), sort_keys=True,
                                                           default=str)


def _omega_key(args: dict, omega: float) -> tuple:
    """Runs count per start state and per parameter values too: from another state, or at another
    parameter value, a run tests something else, not a repeat."""
    return (_model_of(args), omega, json.dumps(args.get("initial_conditions"), sort_keys=True, default=str),
            json.dumps(args.get("parameters"), sort_keys=True, default=str))


def _size(omega) -> float:
    return float(f"{float(omega):.6g}")


def _omegas_named(args: dict) -> list:
    """The system sizes a stochastic call names."""
    omega = args.get("omega")
    try:
        return [_size(o) for o in (omega if isinstance(omega, (list, tuple)) else [omega])]
    except (TypeError, ValueError):      # none named, or "auto"
        return []


def _omegas_run(step: dict) -> list:
    """The system sizes a stochastic call ran, from its result."""
    try:
        return [_size(p["omega"]) for p in json.loads(step.get("output") or "{}").get("per_omega") or []]
    except (TypeError, ValueError, KeyError, AttributeError):
        return []


def _work(args: dict) -> float:
    """How much a stochastic call asks for: runs times model time (the tool's defaults)."""
    try:
        return float(args.get("n_runs", 20)) * float(args.get("t_end", 100.0))
    except (TypeError, ValueError):
        return float("inf")


def _projections(step: dict) -> dict:
    """{omega: projection} of the runs that stopped early."""
    try:
        return {_size(p["omega"]): p["method_results"]["ssa"]["details"]["projected_full_run"]
                for p in json.loads(step.get("output") or "{}").get("per_omega") or []
                if "projected_full_run" in ((p.get("method_results") or {}).get("ssa") or {}).get("details", {})}
    except (TypeError, ValueError, KeyError, AttributeError):
        return {}


def call_key(tool_name: str, args: dict) -> str:
    named = {k: v for k, v in args.items() if k not in ("model_name_or_path", "model")}
    return tool_name + _model_of(args) + json.dumps(named, sort_keys=True, default=str)


OMEGA_TOOLS = ("stochastic_simulation", "compare_simulations")


@dataclass
class ExecutionGuard:
    executed: dict = field(default_factory=dict)    # call key -> step (succeeded)
    failed: dict = field(default_factory=dict)      # call key -> error (this request)
    failures: dict = field(default_factory=dict)    # tool -> failed calls (this request)
    diagrams: dict = field(default_factory=dict)    # (model, parameter, start) -> diagrams drawn
    omega_runs: dict = field(default_factory=dict)  # (model, omega, start) -> stochastic runs at that size
    omega_futile: dict = field(default_factory=dict)  # (model, omega) -> what a projection ruled out

    def check(self, sub_agent: str, tool_name: str, args: dict) -> GuardDecision:
        spec = SUBAGENTS.get(sub_agent)
        if spec is None or tool_name not in spec.objective_tools:
            owned = list(spec.objective_tools) if spec else []
            return GuardDecision(False, f"'{sub_agent}' does not own '{tool_name}'. "
                                        f"Its analysis tools are: {owned}.")
        if not isinstance(args, dict):
            return GuardDecision(False, "Arguments must be a JSON object.")
        schema = registry.schema_of(registry.all_tools()[tool_name])
        props = schema.get("properties", {})
        missing = [a for a in schema.get("required", []) if a not in args]
        unknown = [a for a in args if props and a not in props]
        if missing or unknown:
            return GuardDecision(False, "Invalid arguments: "
                                 + (f"missing required {missing}; " if missing else "")
                                 + (f"unknown {unknown}; " if unknown else "")
                                 + f"valid arguments are {sorted(props)}.")
        why = self.failed.get(call_key(tool_name, args))
        if why is not None:
            return GuardDecision(False, f"This exact call already failed in this request ({why}). "
                                        "Change the arguments, or stop and report the failure.")
        if self.failures.get(tool_name, 0) > config.MAX_TOOL_RETRIES:
            return GuardDecision(False, f"{tool_name} has failed {self.failures[tool_name]} times "
                                        "in this request. Do not call it again: report the error.")
        earlier = self.executed.get(call_key(tool_name, args))
        if earlier is not None:
            return GuardDecision(True, "duplicate of an earlier call in this request; "
                                       "its result is reused", duplicate_of=earlier)
        if tool_name == "run_bifurcation_analysis" and args.get("param_name") and self.diagrams.get(
                _diagram_key(args), 0) >= config.MAX_DIAGRAMS_PER_PARAMETER:
            return GuardDecision(False, f"{config.MAX_DIAGRAMS_PER_PARAMETER} diagrams in "
                                        f"{args['param_name']} from this start already ran in this "
                                        "request. Report from those results, with what stays "
                                        "unresolved, or start from another state.")
        if tool_name in OMEGA_TOOLS:
            spent = [om for om in _omegas_named(args)
                     if self.omega_runs.get(_omega_key(args, om), 0) >= config.MAX_RUNS_PER_OMEGA]
            if spent:
                return GuardDecision(False, f"omega {spent} from this start already had one run and one "
                                            "retry in this request. Report what "
                                            "stands, with what stays unresolved.")
            model, method = _model_of(args), args.get("ssa_method", "auto")
            futile = [(om, self.omega_futile[(model, om, method)]) for om in _omegas_named(args)
                      if _work(args) >= self.omega_futile.get((model, om, method), (float("inf"), 0))[0]]
            if futile:
                om, (_, minutes) = futile[0]
                return GuardDecision(False, f"omega {om:g} needs about {minutes:g} min (approximate) with "
                                            "these settings, so a retry cannot finish. Use fewer runs, "
                                            "or report what stands.")
        return GuardDecision(True, "ok")

    def record(self, tool_name: str, args: dict, step: dict, error: str | None = None):
        """After a call ran: a success can be reused, a failure counts toward the limit."""
        if tool_name == "run_bifurcation_analysis" and args.get("param_name") and not error:
            key = _diagram_key(args)
            self.diagrams[key] = self.diagrams.get(key, 0) + 1
        if tool_name in OMEGA_TOOLS and not error:
            for om in _omegas_run(step):
                key = _omega_key(args, om)
                self.omega_runs[key] = self.omega_runs.get(key, 0) + 1
            for om, full in _projections(step).items():
                if "cannot finish" in full.get("retry", ""):
                    self.omega_futile[(_model_of(args), om, args.get("ssa_method", "auto"))] = (_work(args), full["minutes"])
        key = call_key(tool_name, args)
        if error:
            self.failed[key] = error[:300]
            self.failures[tool_name] = self.failures.get(tool_name, 0) + 1
        else:
            self.executed[key] = step
