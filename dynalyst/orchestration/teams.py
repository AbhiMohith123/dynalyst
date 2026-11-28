"""Team supervisors: the middle layer. The orchestrator delegates a task to a team; the
team supervisor (a language model, <= config.TEAM_MAX_CALLS calls) hands its parts to its
own sub-agents and returns their results. It has no tools of its own, and it gets only
the orchestrator's task. Every sub-agent belongs to exactly one team (checked at import).
"""
from dataclasses import dataclass

from dynalyst.orchestration.subagents import SUBAGENTS


@dataclass(frozen=True)
class Team:
    name: str
    summary: str               # shown to the orchestrator: when to delegate here
    members: tuple

    @property
    def prompt(self) -> str:
        members = "\n".join(f"- delegate_to_{m}: {SUBAGENTS[m].summary}" for m in self.members)
        title = self.name.replace("_", " ")
        return f"""WHO I AM
I am the {title} supervisor of a systems-biology modelling assistant. I receive one task from
the orchestrator and hand its parts to my sub-agents.

SCOPE
{self.summary}
A task outside this scope: delegate nothing and answer only
"Not my task: <one-line reason>."

TOOLS (when to call each)
{members}

INPUT
The orchestrator's task: the model, every value to use and what to produce.

RULES
1. Each delegation names the model and every value it needs: a sub-agent sees only its task.
   It adds no value the task does not give: an omega only when the task names one, as the tool
   computes it. Comparing deterministic and stochastic behaviour is one task for stochastic.
2. A task one sub-agent's description covers goes to that sub-agent whole, in the
   orchestrator's words, with every value. Split a task only into analyses different
   sub-agents do. Add no step the task does not ask for: a model the task names is
   already available, so no search or download unless the task asks for one.
3. Parts that do not need each other's results go out in the same turn (they run in
   parallel); a part that needs another's result waits for it. The simulation sub-agent gets
   a task only when the task I was given asks for a simulation or a time course.
4. When a sub-agent reports an error or an unresolved reviewer verdict, decide: give it a
   corrected task, or report it. Reviewer warnings are reported as they are: I start no
   other analysis for them.

OUTPUT
No tool call: for each sub-agent, what it ran and its result (numbers, IDs, file and plot
paths, errors and reviewer warnings) exactly as reported. Lists (papers, models, steady states) item by item.

EXAMPLES
1. Task: Steady states of model.xml, then its Jacobian network at the stable one.
Turn 1: delegate_to_steady_state("Find the steady states of model.xml and their stability,
        then the Jacobian network at the stable steady state.")
Output: steady_state ran find_steady_states and jacobian_network: one stable steady state
        X = 1.2; negative feedback loop X -| Y -> X; plot outputs/jacobian/model_network.png.

2. Task: Independent dimension of Smith2004, and can it be reduced?
Turn 1: delegate_to_model_classifier("Independent dimension of Smith2004."),
        delegate_to_model_reduction("Can Smith2004 be reduced?")   (in parallel)
(Smith2004 is a model the task names: it is used as named, not searched or downloaded; if
there is no such file, the tool says so.)

3. Task: Compare 5 Gillespie SSA runs of model.xml at omega 100 with the deterministic ODE.
Turn 1: delegate_to_stochastic("Compare 5 Gillespie SSA runs of model.xml at omega 100 with
        the deterministic ODE.")
Output: stochastic ran compare_simulations: SSA mean S = 0.224 vs ODE S = 0.222; overlay
        plot outputs/stochastic/model/compare.png.
"""


TEAMS = {t.name: t for t in [
    Team("model_team",
         "Finding and characterising models: BioModels / UniProt search and download, a "
         "model's ODEs, typed equations or XPP files turned into a model file, saving changed "
         "copies, SBML annotations, classification (dimension, conservation laws, "
         "timescales), model reduction (QSSA), and reaction network theory (CRNT: candidate "
         "bistability parameter sets; drawing the reaction network).",
         ("model_search", "annotations", "model_classifier", "model_reduction", "network_theory")),
    Team("dynamics_team",
         "What a model does: steady states and stability, the Jacobian interaction network, "
         "time courses (deterministic, and stochastic: Gillespie SSA / CLE), stochastic vs "
         "deterministic comparisons (one task), bifurcation diagrams with AUTO-07p, LHS scans "
         "(one task samples the parameter sets, runs AUTO-07p on each and classifies them), "
         "and custom plots.",
         ("steady_state", "simulation", "stochastic", "bifurcation", "plotting")),
    Team("data_team",
         "Data and literature: fitting parameters to measured data, papers (search, abstracts, "
         "uploaded PDFs and their list, the knowledge base), mapping a paper's values to a model, and checking "
         "a model's network against OmniPath.",
         ("parameter_estimation", "research")),
]}


def team_of(sub_agent: str) -> str | None:
    return next((t.name for t in TEAMS.values() if sub_agent in t.members), None)


def _check_partition():
    seen = [m for t in TEAMS.values() for m in t.members]
    unknown = sorted(set(seen) - set(SUBAGENTS))
    twice = sorted({m for m in seen if seen.count(m) > 1})
    missing = sorted(set(SUBAGENTS) - set(seen))
    if unknown or twice or missing:
        raise RuntimeError(f"teams.py: every sub-agent must be in exactly one team "
                           f"(unknown {unknown}, in two teams {twice}, in none {missing})")


_check_partition()
