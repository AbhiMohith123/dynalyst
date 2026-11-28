"""Follow-up questions: what to do next, derived deterministically from what
the run actually found (special points, steady states, fit quality,
identifiability, reviewer warnings, timescale separation). Each suggestion
is a question for the user plus the exact query that would run it.

This is also where the agent "knows when" to reduce a model: every model a
run touched with >= 3 state variables gets a timescale check, and a
validated QSSA reduction is offered when the gap justifies it."""
import json
import re

MAX_FOLLOW_UPS = 7


def _data(step):
    try:
        d = json.loads(step.get("output") or "")
        return d if isinstance(d, dict) else {}
    except (TypeError, ValueError):
        return {}


def _model_of(step):
    args = step.get("arguments") or {}
    return args.get("model_name_or_path") or args.get("sbml_path")


def _suggest_for(step, seen_tools):
    tool, args, out, data = step.get("tool"), step.get("arguments") or {}, step.get("output") or "", _data(step)
    model = _model_of(step)
    status = step.get("status")
    s = []
    from dynalyst.auto_env import AUTO_UNAVAILABLE
    if AUTO_UNAVAILABLE in out:   # a setup problem: other settings cannot help
        s.append(("AUTO-07p cannot build models on this machine. Rebuild it with "
                  "`python scripts/build_auto.py` (or `bash scripts/build_auto.sh`), then run "
                  "this again?", f"Retry: {step.get('task')}"))
        return s
    if status in ("error", "unresolved", "sent_back"):
        s.append((f"The {step['sub_agent'].replace('_', ' ')} step did not finish cleanly. "
                  "Retry it with different settings?", f"Retry: {step.get('task')}"))
        if step.get("sub_agent") == "bifurcation" and tool:
            # offered, never done silently: the sweep is not a bifurcation diagram
            model = (args.get("model") or args.get("model_name_or_path") or "").strip()
            s.append(("Show a plain steady-state sweep instead? (It follows one branch and cannot "
                      "identify folds, branch points or Hopf points.)",
                      f"Do a plain parameter sweep (no continuation) of {model}".strip()))
        return s
    if tool == "classify_model":
        from dynalyst import config
        ts = data.get("timescales") or {}
        if ts.get("qssa_recommended") and config.QSSA_ENABLED:
            s.append((f"{data.get('model')} has a timescale gap: reduce it with a validated "
                      f"quasi-steady-state approximation on {ts.get('fast_species')}?",
                      f"Reduce {model} with the quasi-steady-state approximation and validate it"))
        dim = data.get("independent_dimension") or 0
        if dim >= 2:
            s.append((f"It is {dim}D, so Hopf bifurcations are possible. Run a bifurcation analysis?",
                      f"Run a bifurcation analysis of {model}"))
        s.append(("Find its steady states and their stability?", f"Find the steady states of {model}"))
    elif tool == "find_steady_states":
        # points of a continuum of steady states are not separate stable states: no bistability
        stable = [x for x in data.get("steady_states", []) if str(x.get("stability", "")).startswith("stable")
                  and x.get("isolated") is not False]
        if len(stable) >= 2 and not data.get("continuum_note"):
            kind = {2: "bistability", 3: "tristability"}.get(len(stable), "multistability")
            s.append((f"{len(stable)} stable steady states coexist ({kind}). Trace them with a "
                      "bifurcation diagram?", f"Run a bifurcation analysis of {model}"))
        s.append(("Simulate from different initial conditions to see which state is reached?",
                  f"Simulate {model} and plot the dynamics"))
    elif tool == "run_bifurcation_analysis":
        types = [p.get("type") for p in data.get("bifurcation_points", [])]
        param = data.get("swept_parameter")
        if types.count("LP") >= 2:
            s.append((f"{types.count('LP')} folds were found in {param}. Check for coexisting stable "
                      "steady states between them?", f"Find the steady states of {model}"))
        if not types and param:      # a call the guard refused has no result to read
            s.append((f"No bifurcation in {param}. Try another continuation parameter?",
                      f"Run a bifurcation analysis of {model} without fixing the parameter"))
        if "HB" in types:
            s.append(("A Hopf point was found. Examine nearby trajectories for oscillations?",
                      f"Simulate {model} and plot the dynamics"))
        # other parameters the eigenvalue screen found close to a crossing
        screen = (data.get("automatic_parameter_search") or {}).get("parameter_screen") or []
        for r in screen:
            if r["parameter"] != param and r.get("crossing_near") is not None:
                s.append((f"{r['parameter']} also has a {r['nearest_crossing'].split(' (')[0]} "
                          f"near {r['crossing_near']:.4g} (it is {r['value']:.4g} now). "
                          f"Continue in {r['parameter']} instead?",
                          f"Run a bifurcation analysis of {model} with param_name {r['parameter']}"))
                break
        if types and not s:
            s.append((f"Try another continuation parameter than {param}?",
                      f"Run a bifurcation analysis of {model} without fixing the parameter"))
    elif tool in ("lhs_bifurcation_scan", "categorized_lhs_scan"):
        scanned = args.get("model")
        varied = re.search(r"Sampled parameter sets \(\d+; varied parameters only\):\s*id\s+outcome\s+(\S+)",
                           out)
        if varied and scanned:   # one parameter that THIS scan varied, taken from its own table
            s.append(("Vary only one parameter to see which one controls the behaviour?",
                      f"LHS sample 20 points of {scanned} varying only {varied.group(1)} and "
                      "split them into categories"))
        elif scanned:
            s.append((f"Repeat the scan of {scanned} within a tighter window around its baseline?",
                      f"LHS sample 20 points of {scanned} within +-10% of each parameter's baseline"))
    elif tool == "estimate_parameters":
        ms = data.get("multistart") or {}
        bad = [p for p, v in (ms.get("parameters") or {}).items()
               if str(v.get("status", "")).startswith("not identifiable")]
        pairs = ms.get("correlated_pairs") or []
        if bad and pairs:
            keep = pairs[0]["parameters"][0]
            s.append((f"{bad} are not identifiable ({pairs[0]['parameters']} are correlated). Fix "
                      f"{keep} at its model value and refit the rest?",
                      f"Fit {[p for p in bad if p != keep]} of {model} to {args.get('data_path')}"))
        elif bad:
            s.append(("Some parameters are not identifiable. Refit with more start points?",
                      f"Fit {bad} of {model} to {args.get('data_path')} with 20 initializations"))
        if any(v.get("at_bound") for v in (data.get("fitted_parameters") or {}).values()):
            s.append(("A fitted value hit its bound. Refit with wider bounds?",
                      f"Fit {list(data.get('fitted_parameters'))} of {model} to "
                      f"{args.get('data_path')} with lower_bound 0.01 and upper_bound 100"))
        if not bad:
            s.append(("Use the fitted parameters: rerun the bifurcation analysis with them?",
                      f"Run a bifurcation analysis of {model} with fixed_params "
                      f"{ {p: round(v['fitted'], 6) for p, v in (data.get('fitted_parameters') or {}).items()} }"))
    elif tool == "reduce_model":
        exp = (data.get("sbml_export") or {}).get("model_name")
        acc = (data.get("validation") or {}).get("accurate")
        if exp and acc is not False:
            s.append((f"Analyse the reduced model {exp}: run its bifurcation analysis?",
                      f"Run a bifurcation analysis of {exp}"))
        if acc is False:
            s.append(("The QSSA is not accurate enough. Keep the full model, or accept a larger "
                      "tolerance?", f"Reduce {model} with the quasi-steady-state approximation, "
                      "tolerance 0.1"))
    elif tool in ("search_biomodels", "find_models_for_article"):
        hits = data.get("results") or data.get("models") or []
        if hits:
            h = hits[0]
            s.append((f"Download {h['id']} ({h['name']}) and use it?",
                      f"Download BioModels model {h['id']}"))
            s.append((f"Show {h['id']}'s description and article?",
                      f"Get the BioModels information for {h['id']}"))
    elif tool == "get_biomodels_info":
        s.append((f"Download {data.get('id')} and analyse it?",
                  f"Download BioModels model {data.get('id')}"))
    elif tool == "fetch_biomodels_model" and data.get("use_as"):
        name = data["use_as"]
        s.append((f"Classify {name} (dimension, conservation laws)?", f"Classify {name}"))
        s.append((f"Can {name} be bistable (reaction network theory)?",
                  f"Check with CRNT whether {name} can be bistable"))
        s.append((f"Run a bifurcation analysis of {name}?", f"Run a bifurcation analysis of {name}"))
    elif tool == "crnt_bistability":
        # CRNT answers one question; each further analysis is the user's choice
        sets = data.get("multistationary_sets") or []
        # each candidate bistability set first: AUTO-07p confirms (or not) its bistability; the
        # set sits on a fold, so the tool's own screen finds the total that crosses it
        for i, v in enumerate(sets, start=1):
            totals = ", ".join(f"{t} {x:.4g}" for t, x in v["conservation_totals"].items())
            s.append((f"Check candidate set {i} ({totals}) for bistability with AUTO-07p?",
                      f"Run a bifurcation analysis of {v['model_file']} and check for bistability"))
        if model and data.get("mass_action") is not False:
            s.append((f"Check whether {model} itself, with its own values, is bistable: "
                      "bifurcation analysis with AUTO-07p?",
                      f"Run a bifurcation analysis of {model} and check for bistability"))
        if not sets and model and data.get("mass_action") is False:
            s.append(("CRNT does not apply here. Check bistability with a bifurcation analysis?",
                      f"Run a bifurcation analysis of {model} and check for bistability"))
        if model:
            s.append(("Look for branch points or Hopf points with an LHS scan?",
                      f"Run an LHS bifurcation survey of {model}"))
            if data.get("mass_action") is not False:
                s.append(("Draw the reaction network?", f"Draw the reaction network of {model}"))
            s.append(("Show the interaction network from the Jacobian (who activates / inhibits "
                      "whom)?", f"Draw the Jacobian interaction network of {model}"))
    elif tool in ("simulate_model", "custom_plot", "compare_initial_conditions"):
        behaviour = data.get("behaviour") or {}
        if behaviour.get("verdict") == "still changing at t_end" and model:
            longer = 5 * float(args.get("t_end") or data.get("t_end") or 100)
            s.append((f"The run ended before the dynamics settled. Simulate {model} for {longer:g} "
                      "time units to see whether it settles or oscillates?",
                      f"Simulate {model} for {longer:g} time units"))
        s.append(("Look at the steady states behind these dynamics?", f"Find the steady states of {model}"))
        s.append(("Compare the deterministic trajectory with Gillespie SSA at low copy numbers?",
                  f"Compare deterministic ODE and Gillespie SSA for {model} at several system sizes"))
    elif tool in ("stochastic_simulation", "compare_simulations"):
        omegas = data.get("omega") or []
        name = model or data.get("model")
        methods = data.get("requested_methods") or data.get("methods") or ["ssa"]
        labels = {"ode": "deterministic ODE", "ssa": "tau-leaping" if
                  data.get("ssa_method") == "tau_leap" else "Gillespie SSA", "cle": "CLE"}
        rerun = ("Compare " if "ode" in methods else "Run ") + " and ".join(labels[m] for m in methods)
        if len(omegas) == 1:
            s.append(("Compare the noise at smaller and larger system sizes?",
                      f"{rerun} for {name} with omega "
                      f"[{omegas[0] / 10:g}, {omegas[0]:g}, {omegas[0] * 10:g}]"))
        if not data.get("matlab_script"):
            s.append(("Export this as a MATLAB script (SSA, CLE and ode45)?",
                      f"Write the propensity table of {name} with a MATLAB script"))
        # only a run skipped for time can be forced; one skipped because the stoichiometry is
        # not whole numbers cannot run at all
        if any("SSA skipped" in w and "budget" in w for w in data.get("warnings") or []):
            s.append(("The exact SSA was skipped for time. Run it anyway (it will take a while)?",
                      f"{rerun} for {name} with t_end {data.get('t_end')}, "
                      f"n_runs {data.get('n_runs')} and ssa_method 'exact'"))
        verdicts = " ".join(str(v.get("verdict")) for v in data.get("stability_verdicts") or [])
        frozen = any("No run switched" in " ".join(m.get("explanation", []))
                     for p in data.get("per_omega") or []
                     for m in ((p.get("distribution") or {}).get("by_method") or {}).values())
        if "deterministic only" in verdicts or frozen:
            s.append(("Runs stayed in one state. Run 5x longer to see switching between states?",
                      f"{rerun} for {name} with t_end "
                      f"{5 * float(data.get('t_end') or 100):g}"))
        if "bistable" in verdicts or "multistable" in verdicts:
            s.append(("Where does the bistability start? Draw the bifurcation diagram?",
                      f"Run a bifurcation analysis of {name}"))
    elif tool == "propensity_table":
        name = model or data.get("model")
        s.append(("Simulate it with Gillespie SSA?",
                  f"Run Gillespie SSA for {name} at several system sizes"))
    elif tool == "omnipath_network_check":
        miss = data.get("missing_internal_edges") or []
        loops = [l for l in data.get("loop_candidates") or [] if l.get("kind") == "feedback"]
        if data.get("unmapped_species"):
            s.append((f"{data['unmapped_species']} could not be matched to proteins. Tell me which "
                      "proteins they are and rerun the check?",
                      f"Run the OmniPath network check of {model} with species_map for "
                      f"{data['unmapped_species']}"))
        if miss:
            m = miss[0]
            s.append((f"OmniPath lists {m['source']} -> {m['target']}, which {model} does not "
                      f"contain{' (it would close a feedback loop)' if m.get('closes_loop') else ''}. "
                      "Look up the papers behind it?",
                      f"Search the literature for {m['source']} {m['target']} "
                      f"{'phosphorylation' if m.get('kind') == 'enzyme-substrate' else 'regulation'}"))
        if loops:
            s.append((f"Candidate feedback {loops[0]['path']}. Check whether {model} can already be "
                      "bistable without it (reaction network theory)?",
                      f"Check with CRNT whether {model} can be bistable"))
    elif tool == "get_annotations":
        s.append(("Simulate this model?", f"Simulate {model} and plot the dynamics"))
    return s


def _timescale_offer(model):
    """Offer a QSSA reduction for a model that is >= 3D after conservation
    laws and has a timescale gap."""
    from dynalyst.analysis.model_classifier import classify_model
    data = json.loads(classify_model(model))
    ts = data.get("timescales") or {}
    if (data.get("independent_dimension") or 0) >= 3 and ts.get("qssa_recommended"):
        gap = (ts.get("spectral_gap") or {}).get("median")
        return (f"{data['model']} is {data['independent_dimension']}D and its species {ts['fast_species']} "
                # a rate of ~0 among the slow ones (a conserved direction) gives a gap of 1e299
                f"relax {'over 10^6' if not gap or gap > 1e6 else f'~{gap:.0f}'}x faster than the "
                "rest. Reduce it with a validated QSSA before "
                "further analysis?",
                f"Reduce {model} with the quasi-steady-state approximation and validate it")
    return None


# a follow-up that only repeats a tool this run already ran on the same model
_DONE_BY = (("Classify ", "classify_model"), ("Download BioModels model ", "fetch_biomodels_model"),
            ("Get the BioModels information for ", "get_biomodels_info"),
            ("Check with CRNT whether ", "crnt_bistability"),
            ("Draw the reaction network of ", "draw_reaction_network"),
            ("Find the steady states of ", "find_steady_states"),
            ("Run a bifurcation analysis of ", "run_bifurcation_analysis"))


def _key(query: str) -> str:
    """A follow-up query with every model written the same way (a path or a file name
    'data/sbml_models/M.xml' as 'M'), so two spellings of one action are one action."""
    query = re.sub(r"(?:[\w.\-]+[/\\])+([\w\-]+?)(?:\.xml)?(?=[\s,?]|$)", r"\1", query)
    return re.sub(r"\b([\w\-]+)\.xml\b", r"\1", query)


def _already_done(query: str, done: set) -> bool:
    for prefix, tool in _DONE_BY:
        if not query.startswith(prefix):
            continue
        # '... of M and check for bistability' asks for the same run as '... of M'
        model = re.sub(r"\s+and check for bistability$", "", query[len(prefix):].strip())
        if (tool, model) in done:
            return True
    return False


def suggest(record) -> list[dict]:
    steps = [s for s in record.steps if s.get("kind") == "delegation"]
    # a call that failed, then succeeded with the same tool later in the run, needs no retry
    steps = [s for i, s in enumerate(steps)
             if s.get("status") == "ok" or not any(
                 t.get("tool") == s.get("tool") and t.get("status") == "ok" for t in steps[i + 1:])]
    seen_tools = {s.get("tool") for s in steps}
    # (tool, model) pairs this run completed; the model as _key writes it
    done = set()
    for s in steps:
        args = s.get("arguments") or {}
        model = _model_of(s) or args.get("model_id") or args.get("model")
        if model and s.get("status") == "ok":
            done.add((s.get("tool"), _key(str(model))))
    # parameters some run of this request found points for: "No bifurcation in p" (from a run
    # on another branch, or a narrower window) would contradict them
    found = set()
    for s in steps:
        if s.get("tool") == "run_bifurcation_analysis":
            try:
                d = json.loads(s.get("output") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(d, dict) and d.get("bifurcation_points"):
                found.add(d.get("swept_parameter"))
    out, keys = [], set()
    # one offer per kind: "simulate M for 5000" and "for 10000" (from two short runs) are one
    # (a number inside a model name, BIOMD0000000010, stays: two models are two offers)
    shape = lambda q: re.sub(r"(?<![\w.])\d+(?:\.\d+)?(?:e[+-]?\d+)?(?!\w)", "#", q)
    for step in reversed(steps):  # most recent results first
        for question, query in _suggest_for(step, seen_tools):
            if any(question.startswith(f"No bifurcation in {p}.") for p in found):
                continue
            key = _key(query)
            if (key in keys or shape(question) in [shape(o["question"]) for o in out]
                    or _already_done(key, done)):
                continue
            keys.add(key)
            out.append({"question": question, "query": query})
    # one model however it was written ("M" and "M.xml" are one model, one offer)
    models = {_key(str(m)) for m in (_model_of(s) for s in steps) if m}
    # a reduction changes the reaction network itself: not offered after a network analysis
    from dynalyst import config
    if config.QSSA_ENABLED and not seen_tools & {"reduce_model", "classify_model", "crnt_bistability",
                         "draw_reaction_network"}:
        for m in sorted(models):
            try:
                offer = _timescale_offer(m)
            except Exception:
                offer = None
            if offer:  # after what this run's own results suggest
                out.insert(min(1, len(out)), {"question": offer[0], "query": offer[1]})
    return out[:MAX_FOLLOW_UPS]


# a heading line that opens the answer's own list of suggestions
_OWN_NEXT = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*|\*\*|__)?[ \t]*(?:what(?:'s| is)? next|next steps?|suggested "
    r"(?:next )?(?:steps|analyses)|further (?:steps|analyses)|possible follow[- ]?ups?|"
    r"follow[- ]?up (?:questions|analyses|suggestions)|what would you like (?:to do )?next|"
    r"how would you like to proceed|would you like (?:me )?to)\b[^\n]*$", re.I | re.M)
# a closing offer ("If you need anything else, let me know!")
_CLOSING_OFFER = re.compile(
    r"\n\s*(?:if you (?:have|need|would like|want)|let me know|feel free|please let me know)"
    r"[^\n]*$", re.I)


def strip_own_next_steps(answer: str) -> str:
    """The answer without a closing section of its own suggestions ('Next steps',
    'What next?'), which the generated follow-ups replace. Only a section that runs
    to the end of the answer is removed."""
    answer = _CLOSING_OFFER.sub("", (answer or "").rstrip()).rstrip()
    hits = list(_OWN_NEXT.finditer(answer))
    if not hits:
        return answer
    head = hits[-1]
    rest = answer[head.end():]
    if re.search(r"^[ \t]*(?:#{1,6}[ \t]|\*\*[^*\n]+\*\*[ \t]*:?[ \t]*$)", rest, re.M):
        return answer       # another section follows: not a closing list
    return answer[:head.start()].rstrip()


def format_follow_ups(items) -> str:
    if not items:
        return ""
    return "\n\n**What next?**\n" + "\n".join(f"{i + 1}. {x['question']}" for i, x in enumerate(items))
