"""Reviewer (deterministic, no language model): checks every analysis-tool
result for consistency and scientific correctness, independently of the
tool that produced it.

Verdicts:
  accept                 every check passed
  accept_with_warnings   usable, but something deserves a caveat in the answer
  replan                 a check failed that a different call can fix; the issue
                         and a suggested correction go back to the sub-agent
                         (at most config.REVIEWER_MAX_REPLANS times)
  unresolved             still failing after the last replan

Independent checks re-derive results from the model's own equations rather
than trusting the tool: a reported bifurcation point must be a steady state
with the right eigenvalue signature, steady states are re-solved and their
stability recomputed, simulations must stay finite, keep conserved totals
and actually move when the equations say they should.
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dynalyst import config

REPLAN, WARNING, INFO = "replan", "warning", "info"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = WARNING          # what a failure means
    suggested_args: dict | None = None

    def as_dict(self):
        return {"check": self.name, "passed": self.passed, "severity": self.severity,
                "detail": self.detail}


@dataclass
class Verdict:
    verdict: str
    checks: list = field(default_factory=list)
    suggestion: str | None = None
    suggested_args: dict | None = None

    @property
    def issues(self):
        return [c.detail for c in self.checks if not c.passed and c.severity != INFO]

    def as_dict(self):
        return {"verdict": self.verdict, "issues": self.issues,
                "suggestion": self.suggestion, "suggested_args": self.suggested_args,
                "checks": [c.as_dict() for c in self.checks]}


def _parse(output: str):
    try:
        data = json.loads(output)
        return data if isinstance(data, dict) else None
    except (TypeError, ValueError):
        return None


# each label is followed by the path, alone on the rest of the line (paths may contain spaces)
_TEXT_ARTIFACTS = re.compile(r"(?:Plot saved to|Subplot figure|Combined plot|Category-counts "
                             r"bar chart|Counts chart|Summary CSV|Generated SBML):[ 	]*(.+?)[ 	]*$",
                             re.MULTILINE)


ANSWER_REVIEW_PROMPT = """WHO I AM
I am the reviewer of a systems-biology modelling assistant. I evaluate whether the final
answer is supported by the available analysis and whether the investigation was sufficient
for the user's scientific question.

TASK
Read the user's question first and identify its distinct scientific objectives. Then assess
the answer and the tool results against those objectives. Do not judge an answer only by
whether its wording sounds scientifically plausible.

INPUT
The user's question, the tool results of this request, the earlier answers in this chat (when
the question refers back to them), and the answer to review.

CHECKS
1. MODEL-SPECIFIC SUPPORT
   Model-specific findings must be supported by analysis of the relevant model. General domain
   knowledge may be used as context, but it does not establish that a particular model has a
   particular property, behavior, or mechanism.

2. INVESTIGATION COMPLETENESS
   Check whether the analyses actually performed address all material scientific objectives
   in the question. If the answer skips an objective or a necessary investigation, report the
   missing evidence even when the conclusion might be plausible. Do not impose a fixed number
   of steps: judge sufficiency from the question and the evidence required.

3. COMPARATIVE VALIDITY
   A ranking or comparison must be supported by results for all relevant candidates, obtained
   under sufficiently comparable conditions and evaluated with an appropriate, clearly
   understood metric. A result for one candidate cannot establish which candidate is strongest,
   and candidates the question implies but no result covers leave the ranking unresolved.
   Do not treat distinct metrics (such as response sensitivity, distance to a threshold, and
   width of a regime) as interchangeable.

4. MECHANISTIC AND CAUSAL VALIDITY
   When the answer explains why or how a particular model behaves, require relevant
   model-specific evidence from appropriate structural, mathematical, numerical, or dynamical
   analysis. Depending on the question, this may come from equations, reaction/network
   structure, Jacobian interactions, stability changes, perturbations, simulations, or other
   suitable results. Do not require one particular analysis method when another is appropriate.
   General textbook mechanisms cannot substitute for checking the actual model.

5. NUMERICAL AND SCIENTIFIC CONSISTENCY
   Check reported values, units, parameter ranges, stability classifications, and conclusions
   against the tool results. Identify contradictions, overinterpretation, and unsupported
   extrapolation.

6. UNCERTAINTY AND FAILURE
   Check that failed, unavailable, incomplete, or conflicting analyses are represented
   accurately. Missing evidence must be described as unresolved, not as evidence that a
   behavior is absent or present.

OUTPUT
Only a JSON object: {"issues": [{"claim": "<unsupported claim or missing objective/analysis>",
"why": "<which evidence is missing, what the results show, or why the inference is invalid>"}]}.
If the answer is supported and complete, return {"issues": []}. Do not exempt a model-specific
claim merely because it is labelled as general background or "not from a tool result".
"""


class Reviewer:
    def __init__(self):
        self._models = {}

    # -- entry point ---------------------------------------------------------
    def review(self, tool: str, args: dict, output: str, attempt: int = 0) -> Verdict:
        from dynalyst.auto_env import AUTO_UNAVAILABLE
        data = _parse(output)
        if AUTO_UNAVAILABLE in (output or ""):
            # AUTO cannot build models here: no other call can fix that, so no replan
            text = str(data.get("error") or data.get("conclusion")) if data else output
            detail = text[text.find(AUTO_UNAVAILABLE):][:700] if AUTO_UNAVAILABLE in text else text[:700]
            check = Check("auto_unavailable", False, detail, REPLAN)
            return Verdict("unresolved", [check], suggestion=detail)
        checks = []
        if data is not None and data.get("error"):
            fix, hint = None, ""
            if "Nothing to sample" in str(data["error"]) and (args.get("fixed_params")
                                                              or args.get("vary_only")):
                # usually fixed_params / vary_only listed every parameter although the task
                # only wanted the others at their default (varying) -- drop them
                fix = {k: None for k in ("fixed_params", "vary_only") if args.get(k)}
                hint = (" Unless the task holds ALL of them, drop fixed_params / vary_only "
                        "(the others vary by default).")
            checks.append(Check("tool_error", False, f"The tool reported an error: "
                                f"{str(data['error'])[:600]}{hint}", REPLAN, suggested_args=fix))
        else:
            checks += self._artifacts(data, output)
            handler = getattr(self, f"_review_{tool}", None) or self._family_handler(tool)
            if handler is not None:
                try:
                    checks += handler(args, data, output)
                except Exception as e:  # a reviewer bug must never block a result
                    checks.append(Check("reviewer_internal", False,
                                        f"reviewer could not complete its checks: {e}", INFO))
        return self._verdict(checks, attempt)

    def _verdict(self, checks, attempt) -> Verdict:
        failed = [c for c in checks if not c.passed]
        hard = [c for c in failed if c.severity == REPLAN]
        if hard:
            verdict = "replan" if attempt < config.REVIEWER_MAX_REPLANS else "unresolved"
            fix = next((c for c in hard if c.suggested_args), None)
            return Verdict(verdict, checks, suggestion="; ".join(c.detail for c in hard),
                           suggested_args=fix.suggested_args if fix else None)
        if any(c.severity == WARNING for c in failed):
            return Verdict("accept_with_warnings", checks)
        return Verdict("accept", checks)

    def _family_handler(self, tool):
        if tool in ("lhs_bifurcation_scan", "categorized_lhs_scan"):
            return self._review_lhs
        return None

    # -- shared helpers --------------------------------------------------------
    def _artifacts(self, data, output):
        paths = []
        if data is not None:
            for key in ("plot_path", "counts_plot_path"):
                if data.get(key):
                    paths.append(data[key])
            for p in (data.get("excel_paths") or {}).values():
                paths.append(p)
        else:
            paths += _TEXT_ARTIFACTS.findall(output or "")
        missing = [p for p in paths if not _resolve(p).exists()]
        if not paths:
            return []
        return [Check("output_files_exist", not missing,
                      f"missing output file(s): {missing}" if missing
                      else f"{len(paths)} output file(s) present", REPLAN)]

    def _model(self, model_name_or_path=None, species=None, odes=None, parameters=None,
               initial_conditions=None):
        from dynalyst.analysis.model_classifier import load_any_model
        key = json.dumps([model_name_or_path, species, odes, parameters, initial_conditions],
                         sort_keys=True, default=str)
        if key not in self._models:
            model, _ = load_any_model(model_name_or_path, species, odes, parameters,
                                      initial_conditions)
            self._models[key] = model
        return self._models[key]

    def _kept_nonnegative(self, args, names, params=None) -> set:
        """The variables the model itself keeps >= 0: only for those is a negative value wrong
        (a voltage may have either sign). All of them if the model cannot tell."""
        from dynalyst.analysis.steady_states import kept_nonnegative
        try:
            model = self._model_from_args(args)
            return kept_nonnegative(model, {**model["param_defaults"], **(params or {})})
        except Exception:
            return set(names)

    def _model_from_args(self, args):
        path = args.get("model_name_or_path")
        if path and Path(path).suffix.lower() in (".txt", ".ode"):
            # XPP input: rebuild from the same parser the tool uses
            from dynalyst.analysis.bifurcation_auto import _find_exact_input_path
            from dynalyst.analysis.xpp_to_sbml import parse_xpp
            species, odes, params, ics = parse_xpp(
                _find_exact_input_path(path).read_text(errors="replace"))
            return self._model(None, species, odes, params, ics)
        return self._model(path, args.get("species"), args.get("odes"),
                           args.get("parameters"), args.get("initial_conditions"))

    @staticmethod
    def _numeric(model, params):
        from dynalyst.analysis.steady_states import _lambdify_model
        rhs_fn, jac_fn = _lambdify_model(model)
        vec = [params[p] for p in model["param_names"]]
        f = lambda y: np.array(rhs_fn(*y, *vec), dtype=float)
        j = lambda y: np.array(jac_fn(*y, *vec), dtype=float)
        return f, j

    def _verify_point(self, model, bif_param, value, state, ty, overrides=None):
        """Is (value, state) a steady state with the eigenvalue signature of ty?"""
        from dynalyst.analysis.steady_states import conservation_reduce
        from dynalyst.analysis.conservation_parameters import expose_totals
        model = expose_totals(model)
        params = {**model["param_defaults"], **(overrides or {}), bif_param: value}
        names = model["state_names"]
        if any(n not in state for n in names):
            return Check(f"verify_{ty}", True, f"{ty} at {bif_param}={value:.6g}: state "
                         f"incomplete, not re-checked", INFO)
        y = np.array([state[n] for n in names], dtype=float)
        f, jac = self._numeric(model, params)
        residual = float(np.max(np.abs(f(y))))
        jnorm = float(np.max(np.abs(jac(y)))) if len(y) else 0.0
        tol = 1e-5 * (1.0 + jnorm * (1.0 + float(np.max(np.abs(y)))))
        if residual > tol:
            return Check(f"verify_{ty}", False, f"{ty} at {bif_param}={value:.6g} is not a steady "
                         f"state of the model (max |dy/dt| = {residual:.3g})", WARNING)
        reduced = conservation_reduce({**model, "param_defaults": params})
        yr = np.array([state[n] for n in reduced["state_names"]], dtype=float)
        _, jac_r = self._numeric(reduced, params)
        eig = np.linalg.eigvals(jac_r(yr)) if len(yr) else np.array([])
        scale = 1.0 + (float(np.max(np.abs(eig))) if eig.size else 0.0)
        if ty == "LP" and eig.size and bif_param in reduced["param_names"]:
            # solve for the fold (f = 0, zero eigenvalue) from the reported point: a ratio of
            # eigenvalues cannot tell a fold from a nearby slow mode in a stiff model
            from dynalyst.analysis.fold_refine import refine_fold
            from dynalyst.analysis.steady_states import _lambdify_model
            rhs_fn, jac_fn = _lambdify_model(reduced)
            names_p = reduced["param_names"]
            k = names_p.index(bif_param)
            base = [params[q] for q in names_p]
            def _vec(p):
                v = list(base); v[k] = p; return v
            res = refine_fold(lambda y, p: np.array(rhs_fn(*y, *_vec(p)), dtype=float),
                              lambda y, p: np.array(jac_fn(*y, *_vec(p)), dtype=float),
                              yr, float(value))
            ok = bool(res["accepted"])
            detail = (f"LP at {bif_param}={value:.6g}: steady state (residual {residual:.2g}); "
                      f"fold equations {res['detail']}")
            return Check("verify_LP", ok, detail, WARNING)
        if ty in ("LP", "BP"):
            ratio = float(np.min(np.abs(eig))) / scale if eig.size else 0.0
            ok = ratio < 1e-2
            detail = (f"{ty} at {bif_param}={value:.6g}: steady state (residual {residual:.2g}), "
                      f"smallest |eigenvalue|/scale = {ratio:.2g}"
                      + ("" if ok else " -- no zero eigenvalue, not a fold/branch point"))
        else:  # HB
            cplx = eig[np.abs(eig.imag) > 1e-9 * scale]
            ratio = float(np.min(np.abs(cplx.real) / np.abs(cplx))) if cplx.size else math.inf
            ok = ratio < 1e-2
            detail = (f"HB at {bif_param}={value:.6g}: " + (
                f"complex pair with Re/|lambda| = {ratio:.2g}" if cplx.size else
                "no complex eigenvalue pair") + ("" if ok else " -- not a Hopf point"))
        return Check(f"verify_{ty}", ok, detail, WARNING)

    def _dimension(self, model):
        from dynalyst.analysis.model_classifier import conservation_laws
        return len(model["state_names"]) - len(conservation_laws(model))

    # -- bifurcation -------------------------------------------------------------
    def _review_run_bifurcation_analysis(self, args, data, output):
        checks = []
        param = data.get("swept_parameter")
        points = data.get("bifurcation_points") or []
        cov = data.get("range_coverage") or {}
        used = data.get("auto_constants_used") or {}
        if cov.get("abnormal_terminations"):
            # ONE retry with finer steps (capped); if the call already asked for finer steps
            # or more points, a second identical stop will not be fixed by more of the same:
            # report the diagram as it is, with the warning
            retried = (args.get("nmx") or args.get("dsmin")
                       or (args.get("n_pts") or 0) > config.AUTO_DEFAULT_POINTS)
            msg = (f"{cov['abnormal_terminations']} branch(es) stopped abnormally (AUTO 'MX'); "
                   "structure past that point would be missed")
            if retried or points:
                checks.append(Check("continuation_complete", False, msg + (
                    "; already retried with finer steps, so reported as it is" if retried else ""),
                    WARNING))
            else:
                checks.append(Check(
                    "continuation_complete", False, msg, REPLAN, suggested_args={
                        "nmx": min(int(2 * (used.get("nmx") or 20000)), config.AUTO_MAX_NMX),
                        "dsmin": max(float(used.get("dsmin") or 1e-6) / 10, config.AUTO_MIN_DSMIN)}))
        else:
            checks.append(Check("continuation_complete", True, "no abnormal terminations", INFO))
        try:
            model = self._model_from_args(args)
        except Exception as e:
            return checks + [Check("independent_verification", True,
                                   f"model could not be rebuilt for re-checking: {e}", INFO)]
        overrides = {k: v for k, v in (data.get("fixed_parameters") or {}).items()
                     if k in model["param_defaults"]}
        for pt in points:
            if param in pt and isinstance(pt.get("full_state"), dict):
                checks.append(self._verify_point(model, param, float(pt[param]),
                                                 pt["full_state"], pt["type"], overrides))
        if any(p.get("type") == "HB" for p in points):
            checks.append(self._hopf_dimension(model))
        return checks

    def _review_generic_bifurcation_model(self, args, data, output):
        """The same checks as one diagram, on the reference model the tool built."""
        return self._review_run_bifurcation_analysis(
            {"model_name_or_path": data.get("model_file")}, data, output)

    def _hopf_dimension(self, model):
        n = self._dimension(model)
        return Check("hopf_needs_2d", n >= 2, f"Hopf point in a {n}D model"
                     + ("" if n >= 2 else " -- impossible (needs >= 2 dimensions); numerical artifact"),
                     WARNING)

    def _review_lhs(self, args, data, output):
        checks = []
        n = args.get("n_samples") or 20
        sig = re.search(r"Bifurcation signatures found: (\{.*\})", output)
        if sig:
            counts = json.loads(sig.group(1).replace("'", '"'))
            failed = counts.get("FAILED", 0)
            offset = args.get("additive_offset")
            checks.append(Check("sample_failure_rate", failed <= 0.5 * n,
                                f"{failed}/{n} samples failed to converge", REPLAN,
                                suggested_args=({"additive_offset": float(offset) / 4}
                                                if offset else None)))
        cats = re.search(r"(?:Final category counts|Category counts): (\{.*\})", output)
        if cats:
            counts = json.loads(cats.group(1).replace("'", '"'))
            classified = sum(counts.values())
            checks.append(Check("samples_classified", classified >= 0.5 * n,
                                f"{classified}/{n} samples fell into a category", WARNING))
            if {"ultrasensitive", "graded"} & set(counts):
                checks.append(Check("ultrasensitivity_labels", False,
                                    "ultrasensitive vs graded is unreliable (finite window): read both as "
                                    "'no fold or branch point'", WARNING))
        return checks

    # -- BioModels ----------------------------------------------------------------
    def _review_search_biomodels(self, args, data, output):
        n = len(data.get("results") or [])
        return [Check("search_results", n > 0, f"{n} model(s) found", WARNING)]

    def _review_find_biomodels_by_paper(self, args, data, output):
        n = data.get("n_matching") or 0
        return [Check("paper_found", n > 0,
                      f"{n} BioModels entr{'y' if n == 1 else 'ies'} from this paper"
                      + (" (variants; the default and the others must be named)" if n > 1 else "")
                      if n else data.get("note", "no entry from this paper"), WARNING)]

    def _review_find_models_for_article(self, args, data, output):
        n = len(data.get("models") or [])
        return [Check("article_models", True, f"{n} model(s) whose article title is printed in "
                      "the PDF" if n else data.get("note", "none found"), INFO)]

    def _review_fetch_biomodels_model(self, args, data, output):
        import libsbml
        path = _resolve(data.get("saved_as", ""))
        ok = path.exists() and libsbml.readSBMLFromFile(str(path)).getModel() is not None
        return [Check("model_downloaded", ok, f"{path.name} saved and readable as SBML" if ok
                      else f"{path} missing or not SBML", REPLAN)]

    # -- reaction network theory --------------------------------------------------
    def _review_crnt_bistability(self, args, data, output):
        if data.get("mass_action") is False:
            return [Check("crnt_applicable", True, "not a mass-action network: CRNT does not "
                          "apply (reported as such)", INFO)]
        checks = []
        sets = data.get("multistationary_sets") or []
        if data.get("bistable") is True:      # CRNT alone never checks stability
            checks.append(Check("no_stability_claim", False, "bistable claimed, but CRNT does "
                                "not check stability", REPLAN))
        if sets:
            missing = [s["model_file"] for s in sets if not _resolve(s["model_file"]).exists()]
            checks.append(Check("parameter_sets_saved", not missing,
                                "model files saved" if not missing else f"missing: {missing}",
                                WARNING))
        elif data.get("bistable") is False:
            proof = data.get("deficiency_zero_theorem") or data.get("deficiency_one_theorem")
            checks.append(Check("no_bistability_basis", True,
                                "ruled out by a deficiency theorem" if proof else
                                "no bistable set found: evidence, not proof", INFO))
        else:
            checks.append(Check("crnt_undecided", True, "CRNT could not decide; reported", INFO))
        return checks

    # -- steady states -----------------------------------------------------------
    def _review_find_steady_states(self, args, data, output):
        if "states_across" in data:      # each value of the range as its own result
            base = data.get("parameters_used") or {}
            return [c for row in data["states_across"] if row["steady_states"]
                    for c in self._review_find_steady_states(args, {
                        "steady_states": row["steady_states"],
                        "parameters_used": {**base, data["scan_parameter"]: row["value"]}}, output)]
        states = [s for s in data.get("steady_states", []) if "state" in s]
        if not states:
            if data.get("non_physical_solutions"):
                # the equations' solutions all have a negative value: reported, with the note
                # saying when they are the model's steady states -- another search finds the same
                return [Check("steady_states_found", True,
                              "no non-negative steady state; the solutions with a negative value "
                              "are reported (non_physical_note says when they are the model's "
                              "steady states)", WARNING)]
            return [Check("steady_states_found", False, "no steady state found", REPLAN,
                          suggested_args={"n_random_starts": 150})]
        checks = []
        model = self._model_from_args(args)
        params = {**model["param_defaults"], **(data.get("parameters_used") or {})}
        f, jac = self._numeric(model, params)
        bad, mislabeled, negative = [], [], []
        from dynalyst.analysis.steady_states import classify_stability, exact_conservation_laws
        conserved_count = len(exact_conservation_laws({**model, "param_defaults": params}))
        kept = self._kept_nonnegative(args, model["state_names"], params)
        for s in states:
            y = np.array([s["state"][n] for n in model["state_names"]], dtype=float)
            if float(np.max(np.abs(f(y)))) > 1e-5 * (1 + float(np.max(np.abs(y)))):
                bad.append(s["state"])
            if classify_stability(np.linalg.eigvals(jac(y)), conserved_count).split(" ")[0] != s["stability"].split(" ")[0]:
                mislabeled.append(s["state"])
            if any(v < -1e-9 * (1 + float(np.max(np.abs(y))))
                   for v, name in zip(y, model["state_names"]) if name in kept):
                negative.append(s["state"])
        checks.append(Check("states_are_steady", not bad,
                            f"{len(states) - len(bad)}/{len(states)} reported states re-verified "
                            "as steady states" + (f"; not steady: {bad}" if bad else ""), REPLAN))
        checks.append(Check("stability_labels", not mislabeled,
                            "stability labels match recomputed eigenvalues" if not mislabeled
                            else f"stability label disagrees with eigenvalues for {mislabeled}",
                            WARNING))
        checks.append(Check("non_negative", not negative,
                            "no state the model keeps non-negative is negative" if not negative else
                            f"negative concentrations (non-physical) in {negative}", WARNING))
        return checks

    # -- simulation --------------------------------------------------------------
    def _review_simulate_model(self, args, data, output):
        rows = data.get("sampled_time_series") or []
        species = [s for s in (data.get("species") or []) if rows and s in rows[0]]
        if not rows or not species:
            return []
        values = np.array([[r[s] for s in species] for r in rows], dtype=float)
        checks = [Check("finite", bool(np.all(np.isfinite(values))),
                        "trajectory is finite" if np.all(np.isfinite(values)) else
                        "trajectory contains NaN/inf", REPLAN,
                        suggested_args={"t_end": float(args.get("t_end", 100.0)) / 10})]
        if not np.all(np.isfinite(values)):
            return checks
        scale = 1.0 + float(np.max(np.abs(values)))
        from dynalyst.analysis.typed_equations import is_typed_file
        if not is_typed_file(args.get("model_name_or_path") or ""):   # typed variables: any sign
            kept = self._kept_nonnegative(args, species)
            ok = all(np.all(values[:, j] > -1e-6 * scale) for j, s in enumerate(species) if s in kept)
            checks.append(Check("non_negative", bool(ok), "concentrations stay non-negative" if ok
                                else "negative concentrations appear", WARNING))
        try:
            model = self._model(args["model_name_or_path"])
        except Exception:
            return checks
        if sorted(model["state_names"]) != sorted(species):
            return checks
        idx = [species.index(n) for n in model["state_names"]]
        traj = values[:, idx]
        f, _ = self._numeric(model, model["param_defaults"])
        moving = float(np.max(np.ptp(traj, axis=0))) > 1e-9 * scale
        if not args.get("initial_conditions"):
            d0 = float(np.max(np.abs(f(traj[0]))))
            checks.append(Check("dynamics_consistent", moving or d0 < 1e-9 * scale,
                                "trajectory moves as the equations predict" if moving or d0 < 1e-9 * scale
                                else f"trajectory is flat although |dy/dt| = {d0:.3g} at t=0 -- the "
                                     "simulator did not integrate this model's equations", WARNING))
        from dynalyst.analysis.model_classifier import conservation_laws
        for c in conservation_laws(model):
            totals = traj @ c
            drift = float(np.ptp(totals)) / (1e-12 + float(np.max(np.abs(totals))))
            checks.append(Check("conservation", drift < 1e-3,
                                f"conserved total drifts by {drift:.2g} (relative)", WARNING))
        return checks

    # -- stochastic simulation ----------------------------------------------------
    def _review_stochastic_simulation(self, args, data, output):
        rows = data.get("sampled_time_series") or []
        vals = [v for r in rows for k, v in r.items() if k not in ("omega", "time")]
        finite = bool(vals) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
        checks = [Check("finite", finite,
                        "trajectories are finite" if finite else
                        "a trajectory contains NaN/inf (an SSA run stopped at the event limit, "
                        "or the ODE blew up)", WARNING)]
        if vals and finite:
            low = min(vals)
            checks.append(Check("non_negative", low > -1e-9,
                                "all mean trajectories stay non-negative" if low > -1e-9 else
                                f"a mean trajectory goes negative ({low:.3g})", WARNING))
        for w in data.get("warnings") or []:
            checks.append(Check("tool_warning", False, w, WARNING))
        # a stochastic run in which no molecule of any species ever appeared simulated
        # nothing: its "results" must not be read as the model's stochastic behaviour
        for p in data.get("per_omega") or []:
            sp_stats = p.get("species") or {}
            means = [s.get("ssa_mean") for s in sp_stats.values() if s.get("ssa_mean") is not None]
            moving = any(abs(s.get("ode_final") or 0) > 0 for s in sp_stats.values())
            if means and moving and all(m == 0 for m in means):
                checks.append(Check(
                    "ssa_not_empty", False,
                    f"at omega={p.get('omega'):g} the SSA died out (initial "
                    f"{p.get('initial_molecules')} molecules) but the ODE did not: a real "
                    "low-copy outcome; a larger omega gives the deterministic limit",
                    WARNING))
        # deterministic limit: at the largest omega with an SSA, a species whose ODE does not
        # oscillate should have an SSA mean near the ODE (a large gap means bistability,
        # very low copy numbers, or a propensity error)
        with_ssa = [p for p in data.get("per_omega") or [] if isinstance(p.get("ssa"), dict)]
        if with_ssa and any(
                s.get("ssa_vs_ode_mean_rel_diff") is not None
                for p in with_ssa for s in (p.get("species") or {}).values()):
            top = max(with_ssa, key=lambda p: p["omega"])
            far = {n: s["ssa_vs_ode_mean_rel_diff"] for n, s in (top.get("species") or {}).items()
                   if s.get("ssa_vs_ode_mean_rel_diff") is not None and not s.get("ode_oscillation")
                   and s["ssa_vs_ode_mean_rel_diff"] > 0.2 and abs(s.get("ode_final", 0)) > 1e-9}
            checks.append(Check(
                "deterministic_limit", not far,
                f"at omega={top['omega']:g} the SSA means agree with the ODE" if not far else
                f"at omega={top['omega']:g} the SSA mean differs from the ODE for "
                f"{ {k: f'{100 * v:.0f}%' for k, v in far.items()} }: expected for bistable or "
                "very low-copy systems; otherwise check the propensity table", WARNING))
        if not data.get("reactions"):
            checks.append(Check("reactions", False, "no reaction channels were reported", REPLAN))
        for v in data.get("stability_verdicts") or []:
            if "deterministic only" in str(v.get("verdict")):
                checks.append(Check(
                    "distribution_vs_ode", False,
                    f"omega={v['omega']:g}, {v['method']}: the ODE has several stable states but "
                    "the stochastic distribution has one peak: the runs may be too short to "
                    "switch", WARNING))
        return checks

    def _review_compare_simulations(self, args, data, output):
        checks = self._review_stochastic_simulation(args, data, output)
        completed = data.get("status") == "completed"
        checks.append(Check("comparison_complete", completed,
                            "All requested comparison methods completed." if completed else
                            "some requested methods failed, were skipped or stopped early: the comparison "
                            "is partial", WARNING))
        return checks

    def _review_jacobian_network(self, args, data, output):
        states = data.get("steady_states") or []
        checks = [Check("steady_states", bool(states), f"{len(states)} steady state(s) analysed"
                        if states else "no steady state", REPLAN)]
        for s in states:
            if s.get("warning"):
                checks.append(Check("is_steady_state", False, s["warning"], WARNING))
            J = s.get("jacobian")
            if isinstance(J, list):
                ok = all(np.isfinite(v) for row in J for v in row)
                checks.append(Check("jacobian_finite", ok, f"state {s['index']}: Jacobian "
                                    + ("finite" if ok else "has NaN/inf"), REPLAN))
        return checks

    def _review_propensity_table(self, args, data, output):
        rows = data.get("reactions") or []
        checks = [Check("reactions", bool(rows),
                        f"{len(rows)} reaction channel(s)" if rows else "no reaction channels",
                        REPLAN)]
        if rows and data.get("integer_stoichiometry") is False:
            checks.append(Check("integer_stoichiometry", False,
                                "some stoichiometry is not whole numbers: the SSA cannot run "
                                "on this model as written (the CLE can)", WARNING))
        return checks

    def _review_custom_plot(self, args, data, output):
        conds = data.get("conditions") or []
        ok = all(c.get("finite", True) for c in conds)
        return [Check("finite", ok, "all plotted trajectories finite" if ok else
                      "a plotted trajectory contains NaN/inf", REPLAN,
                      suggested_args={"t_end": float(args.get("t_end", 100.0)) / 10})]

    def _review_compare_initial_conditions(self, args, data, output):
        finals = [v for r in data.get("results", []) for k, v in r.get("final_state", {}).items()
                  if k != "time"]
        ok = all(math.isfinite(float(v)) for v in finals)
        return [Check("finite", ok, "all runs finite" if ok else "a run produced NaN/inf", REPLAN)]

    # -- parameter estimation ----------------------------------------------------
    def _review_estimate_parameters(self, args, data, output):
        checks = []
        first, last = data.get("initial_loss"), data.get("final_loss")
        lr, n_iter = data.get("learning_rate", 0.05), data.get("n_iter", 600)
        improved = first is not None and last is not None and last < first
        checks.append(Check("loss_decreased", improved,
                            f"loss {first:.3g} -> {last:.3g}" if improved else
                            f"loss did not decrease ({first} -> {last})", REPLAN,
                            suggested_args={"learning_rate": lr / 5, "n_iter": int(n_iter * 2)}))
        if improved and data.get("loss_reduction_factor") and data["loss_reduction_factor"] < 10:
            checks.append(Check("loss_reduction", False,
                                f"loss fell only {data['loss_reduction_factor']:.3g}x", WARNING))
        at_bound = [p for p, v in (data.get("fitted_parameters") or {}).items() if v.get("at_bound")]
        lo, hi = data.get("bounds_factor", [0.1, 10.0])
        checks.append(Check("within_bounds", not at_bound,
                            "no fitted parameter at its search bound" if not at_bound else
                            f"{at_bound} ended at the search bound -- the optimum may lie outside "
                            f"[{lo}x, {hi}x] nominal", REPLAN,
                            suggested_args={"lower_bound": lo / 10, "upper_bound": hi * 10}))
        ms = data.get("multistart") or {}
        bad = [p for p, v in (ms.get("parameters") or {}).items()
               if v.get("status", "").startswith("not identifiable")]
        if bad:
            pairs = [c["parameters"] for c in ms.get("correlated_pairs", [])]
            checks.append(Check("identifiability", False,
                                f"{bad} are not identifiable from these data: equally good fits "
                                "spread widely" + (f"; correlated pairs {pairs}" if pairs else ""),
                                WARNING))
        elif ms and ms.get("n_good_fits", 0) < 2:
            checks.append(Check("identifiability", False,
                                "only one start point reached a good fit, so identifiability "
                                "cannot be assessed; use more n_initializations", WARNING))
        runs = data.get("runs") or []
        failed = sum(1 for r in runs if r.get("failed"))
        if runs and failed:
            checks.append(Check("start_points", failed <= len(runs) / 2,
                                f"{failed}/{len(runs)} start points failed", WARNING))
        return checks

    # -- classifier / annotations ------------------------------------------------
    def _review_classify_model(self, args, data, output):
        n, laws, dim = (data.get("n_state_variables"), data.get("n_conservation_laws"),
                        data.get("independent_dimension"))
        ok = n is not None and dim == n - laws and dim >= 1
        return [Check("dimension_consistent", ok,
                      f"{n} state variables - {laws} conservation law(s) = {dim}D", WARNING)]

    def _review_reduce_model(self, args, data, output):
        checks = []
        v = data.get("validation")
        if v and "max_relative_error" in v:
            checks.append(Check("reduction_accurate", bool(v.get("accurate")),
                                f"reduced model deviates up to {v['max_relative_error']:.2%} from "
                                f"the full model (tolerance {v.get('tolerance', 0.05):.0%})"
                                + ("" if v.get("accurate") else " -- not a faithful replacement"),
                                WARNING))
        ts = data.get("timescale_analysis") or {}
        if data.get("qssa") and ts.get("qssa_recommended") is False:
            checks.append(Check("qssa_justified", False, "QSSA was forced although the timescale "
                                f"analysis does not support it ({ts.get('reason')})", WARNING))
        exp = data.get("sbml_export")
        if exp and exp.get("error"):
            checks.append(Check("export", False, f"SBML export failed: {exp['error']}", WARNING))
        return checks

    def _review_omnipath_network_check(self, args, data, output):
        mapping = data.get("species_mapping") or {}
        unmapped = data.get("unmapped_species") or []
        edges = data.get("edges") or []
        checks = [Check("species_mapped", not unmapped,
                        f"{len(mapping) - len(unmapped)} of {len(mapping)} species mapped to proteins"
                        + (f"; unmapped: {unmapped} (the check ignores them; give them in "
                           "species_map to include them)" if unmapped else ""), WARNING),
                  Check("model_edges_read", bool(edges),
                        f"{len(edges)} protein-level edge(s) read from the reactions" if edges else
                        "no enzyme -> substrate or binding edge could be read from the reactions: "
                        "the model's network was not checked", WARNING)]
        genes = set(data.get("model_genes") or [])
        bad = [e for e in edges if not {e.get("source"), e.get("target")} <= genes]
        checks.append(Check("edges_use_mapped_proteins", not bad,
                            "every edge joins mapped proteins" if not bad else
                            f"edges with unknown proteins: {bad[:3]}", REPLAN))
        orth = [s for s, m in mapping.items() if "orthologue" in str(m.get("note", ""))]
        if orth:
            checks.append(Check("orthologue_mapping", True, f"{len(orth)} species matched to "
                                "orthologues by gene symbol", INFO))
        return checks

    def _review_get_annotations(self, args, data, output):
        has = bool(data.get("has_any_annotation"))
        return [Check("has_annotations", has, "annotations present" if has else
                      "this SBML file carries no annotations", WARNING)]


def _resolve(path: str) -> Path:
    p = Path(str(path).strip().strip("\"'`"))
    return p if p.is_absolute() else config.PROJECT_ROOT / p
