"""Synchronous per-start diagnostics around the installed JaxKineticModel trainer."""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import numpy as np


def dependency_info(trainer):
    import jax
    import diffrax
    path = Path(inspect.getfile(type(trainer))).resolve()
    try:
        version = importlib.metadata.version("jaxkineticmodel")
    except importlib.metadata.PackageNotFoundError:
        version = "unpackaged source"
    return {"jaxkineticmodel_version": version, "training_source": str(path),
            "training_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "jax": jax.__version__, "diffrax": diffrax.__version__,
            "solver": type(trainer.kin_model.solver).__name__,
            "initial_step": trainer.kin_model.dt0,
            "max_steps": trainer.kin_model.max_steps}


def train_diagnosed(trainer, trace_path=None):
    """Synchronize loss, gradients and updates inside each start's exception boundary.

    Keep the best evaluated parameters (loss and parameters always refer to the same
    point). An invalid proposed step is backtracked in optimization coordinates;
    starts are independent, so one solver failure cannot erase completed fits.
    """
    import jax
    import jax.numpy as jnp
    import json
    def record(item):
        if trace_path is not None:
            with Path(trace_path).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item, default=str) + '\n')
    ys, ts = jnp.asarray(trainer.dataset), jnp.asarray(trainer.ts)
    log = trainer.optim_space == "log"
    def objective(params):
        opt = jax.tree.map(jnp.log2, params) if log else params
        return trainer.loss_func(opt, ts, ys)
    evaluate = jax.jit(objective)
    runs, optimized = [], {}
    for index, row in trainer.parameter_sets.iterrows():
        params = {k: float(v) for k, v in row.items()}
        run = {"init": int(index), "initial_parameters": dict(params), "failed": False,
               "initial_loss": None, "final_loss": None, "iterations": 0, "trace": [],
               "termination": "iteration limit", "last_finite_parameters": None}
        best_loss, best_params = np.inf, None
        opt_state = trainer.optimizer.init(params)
        record({'init':int(index), 'event':'initial_evaluation', 'parameters':params})
        try:
            current = float(evaluate(params).block_until_ready())
            if not np.isfinite(current):
                raise FloatingPointError("Initial loss is not finite.")
            run["initial_loss"] = current
            best_loss, best_params = current, dict(params)
            for iteration in range(trainer.n_iter):
                run["last_finite_parameters"] = dict(params)
                if current <= trainer.loss_threshold:
                    run["termination"] = "loss threshold"
                    break
                candidate_state, candidate, loss, grads = trainer.make_step(
                    opt_state=opt_state, params=params, ts=ts, ys=ys)
                # Device errors must be raised here, not during later JSON conversion.
                jax.block_until_ready((candidate_state, candidate, loss, grads))
                candidate = {k: float(v) for k,v in candidate.items()}
                gradients = np.array([float(v) for v in grads.values()])
                if not np.all(np.isfinite(gradients)) or not np.isfinite(float(loss)):
                    raise FloatingPointError("Non-finite loss or gradient.")
                item = {"iteration": iteration, "parameters": dict(params), "loss": current,
                        "gradient_norm": float(np.linalg.norm(gradients)),
                        "proposed_parameters": candidate, "solver_status": "completed"}
                run["trace"].append(item)
                record({'init':int(index), **item})
                accepted = False
                for shrink in range(13):
                    fraction = 0.5 ** shrink
                    if log and all(np.isfinite(v) and v > 0 for v in candidate.values()):
                        trial = {k: float(np.exp2(np.log2(params[k]) + fraction *
                                  (np.log2(candidate[k]) - np.log2(params[k])))) for k in params}
                    else:
                        trial = {k: params[k] + fraction*(candidate[k]-params[k]) for k in params}
                    if not all(np.isfinite(v) and v > 0 for v in trial.values()):
                        item["rejected_reason"] = "non-positive or non-finite proposed parameter"
                        continue
                    try:
                        proposed_loss = float(evaluate(trial).block_until_ready())
                        if not np.isfinite(proposed_loss):
                            raise FloatingPointError("Non-finite proposed loss")
                    except Exception as exc:
                        item["rejected_reason"] = f"{type(exc).__name__}: {exc}"
                        continue
                    # Backtracking protects against large uphill optimizer excursions.
                    if proposed_loss <= current * 1.001 + 1e-15:
                        params, current, opt_state = trial, proposed_loss, candidate_state
                        item.update(step_fraction=fraction, evaluated_loss=current)
                        accepted = True
                        break
                run["iterations"] = iteration + 1
                if not accepted:
                    run["termination"] = "no finite improving step"
                    item["solver_status"] = "step rejected"
                    break
                if current < best_loss:
                    best_loss, best_params = current, dict(params)
            run["last_finite_parameters"] = dict(params)
        except Exception as exc:
            run.update(termination="solver/gradient failure", error=f"{type(exc).__name__}: {exc}",
                       failure_parameters=dict(params))
        run["failed"] = best_params is None
        if best_params is not None:
            optimized[int(index)] = best_params
            run.update(final_loss=float(best_loss), best_parameters=best_params,
                       converged=best_loss <= trainer.loss_threshold)
        runs.append(run)
        record({'init':int(index), 'event':'finished', **{k:v for k,v in run.items() if k != 'trace'}})
    return optimized, runs
