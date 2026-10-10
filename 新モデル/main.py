from __future__ import annotations

"""Solve the symmetry-reduced master problem from an RSD initial point.

The outer algorithm is a majorization-minimization method for the concave,
piecewise-linear ordinal-efficiency violation. At each iteration:

1. solve the 762 small inner LPs;
2. obtain one supergradient for each representative;
3. minimize the resulting affine upper bound over the exact feasible set
   defined by bistochasticity, anonymity, neutrality, and strategy-proofness.

The master LP uses constraint generation: only violated strategy-proofness
constraints are added, but the returned point is checked against the full
constraint matrix before it is accepted.

The method is monotone but, because the original outer problem is nonconvex,
it does not certify a global optimum.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from matrix import ModelData, build_or_load_model, solve_oe_violation


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ObjectiveEvaluation:
    value: float
    supergradient: FloatArray
    max_inner_violation: float


@dataclass(frozen=True)
class MasterSolution:
    vector: FloatArray
    active_constraints: IntArray
    cut_rounds: int
    max_sp_violation: float


def evaluate_objective_and_supergradient(
    model: ModelData,
    mechanism_vector: FloatArray,
) -> ObjectiveEvaluation:
    """Evaluate the exact average OE violation and one supergradient."""

    if mechanism_vector.shape != (model.number_of_variables,):
        raise ValueError(
            "Unexpected mechanism-vector shape: "
            f"{mechanism_vector.shape}."
        )

    objective = 0.0
    global_supergradient = np.zeros_like(mechanism_vector)
    max_inner_violation = 0.0
    weights = model.normalized_weights

    for representative_id, cumulative in enumerate(
        model.cumulative_matrices
    ):
        representative_slice = model.representative_slice(representative_id)
        allocation_vector = mechanism_vector[representative_slice]
        result = solve_oe_violation(
            cumulative=cumulative,
            score=model.score_vectors[representative_id],
            inner_equalities=model.inner_equalities,
            inner_rhs=model.inner_rhs,
            allocation_vector=allocation_vector,
        )
        weight = weights[representative_id]
        objective += weight * result.value
        global_supergradient[representative_slice] = (
            weight * result.supergradient
        )
        max_inner_violation = max(
            max_inner_violation,
            result.max_primal_violation,
        )

    return ObjectiveEvaluation(
        value=float(objective),
        supergradient=global_supergradient,
        max_inner_violation=max_inner_violation,
    )


def solve_master_lp(
    model: ModelData,
    linear_objective: FloatArray,
    active_constraints: IntArray,
    *,
    method: str = "highs-ipm",
    sp_tolerance: float = 1e-9,
    cut_batch_size: int = 5_000,
    max_cut_rounds: int = 50,
) -> MasterSolution:
    """Solve the exact master LP by strategy-proofness cut generation."""

    active_set = {int(index) for index in active_constraints.tolist()}

    for cut_round in range(max_cut_rounds + 1):
        active = np.asarray(sorted(active_set), dtype=np.int64)
        if active.size:
            active_sp_matrix = model.sp_matrix[active]
            a_ub = -active_sp_matrix
            b_ub = np.zeros(active.size, dtype=np.float64)
        else:
            a_ub = None
            b_ub = None

        result = linprog(
            c=linear_objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=model.equality_matrix,
            b_eq=model.equality_rhs,
            bounds=(0.0, 1.0),
            method=method,
        )

        if not result.success:
            raise RuntimeError(
                "The master LP failed: "
                f"status={result.status}, message={result.message}"
            )

        vector = np.asarray(result.x, dtype=np.float64)
        sp_values = np.asarray(model.sp_matrix @ vector).reshape(-1)
        violated = np.flatnonzero(sp_values < -sp_tolerance)
        max_sp_violation = float(
            max(0.0, -sp_values.min(initial=0.0))
        )

        if violated.size == 0:
            return MasterSolution(
                vector=vector,
                active_constraints=active,
                cut_rounds=cut_round,
                max_sp_violation=max_sp_violation,
            )

        # Add the most violated constraints. Existing active rows are omitted.
        ordered = violated[np.argsort(sp_values[violated])]
        added = 0
        for index in ordered:
            integer_index = int(index)
            if integer_index in active_set:
                continue
            active_set.add(integer_index)
            added += 1
            if added >= cut_batch_size:
                break

        if added == 0:
            raise RuntimeError(
                "SP violations remain, but no new cut could be generated."
            )

    raise RuntimeError(
        f"SP cut generation did not converge in {max_cut_rounds} rounds."
    )


def feasible_line_search(
    model: ModelData,
    current: FloatArray,
    candidate: FloatArray,
    current_evaluation: ObjectiveEvaluation,
    *,
    minimum_step: float = 2.0 ** -12,
) -> tuple[FloatArray, ObjectiveEvaluation, float]:
    """Numerical safeguard; convex combinations preserve all constraints."""

    step = 1.0
    while step >= minimum_step:
        trial = (1.0 - step) * current + step * candidate
        trial_evaluation = evaluate_objective_and_supergradient(model, trial)
        if trial_evaluation.value <= current_evaluation.value + 1e-9:
            return trial, trial_evaluation, step
        step *= 0.5

    return current.copy(), current_evaluation, 0.0


def save_solution(
    path: Path,
    model: ModelData,
    mechanism_vector: FloatArray,
    history: list[float],
    active_constraints: IntArray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        representatives=np.asarray(model.representatives, dtype=np.int8),
        orbit_weights=model.orbit_weights,
        mechanism_vector=mechanism_vector,
        rsd_initial=model.rsd_initial,
        objective_history=np.asarray(history, dtype=np.float64),
        active_sp_constraints=active_constraints,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimize ordinal-efficiency violation from an RSD initial "
            "mechanism under exact SP, anonymity, and neutrality."
        )
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("model_cache.pkl"),
        help="Precomputed symmetry/constraint cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and rebuild the cache.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--master-method",
        choices=("highs", "highs-ipm", "highs-ds"),
        default="highs-ipm",
    )
    parser.add_argument(
        "--sp-tolerance",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--cut-batch-size",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--max-cut-rounds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("solution.npz"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    build_start = time.perf_counter()
    model = build_or_load_model(
        arguments.cache,
        rebuild=arguments.rebuild_cache,
    )
    build_seconds = time.perf_counter() - build_start

    print(f"representatives        : {len(model.representatives):,}")
    print(f"original profiles      : {model.number_of_profiles:,}")
    print(f"decision variables     : {model.number_of_variables:,}")
    print(f"equality constraints   : {model.equality_matrix.shape[0]:,}")
    print(f"SP constraint pool     : {model.sp_matrix.shape[0]:,}")
    print(f"cache/build seconds    : {build_seconds:.2f}")

    current = model.rsd_initial.copy()
    print(f"RSD equality violation : {model.max_equality_violation(current):.3e}")
    print(f"RSD SP violation       : {model.max_sp_violation(current):.3e}")

    current_evaluation = evaluate_objective_and_supergradient(
        model,
        current,
    )
    history = [current_evaluation.value]
    active_constraints = np.empty(0, dtype=np.int64)

    print(f"RSD OE objective       : {current_evaluation.value:.12g}")
    print(
        "RSD inner residual      : "
        f"{current_evaluation.max_inner_violation:.3e}"
    )

    for iteration in range(1, arguments.max_iterations + 1):
        iteration_start = time.perf_counter()

        master = solve_master_lp(
            model,
            current_evaluation.supergradient,
            active_constraints,
            method=arguments.master_method,
            sp_tolerance=arguments.sp_tolerance,
            cut_batch_size=arguments.cut_batch_size,
            max_cut_rounds=arguments.max_cut_rounds,
        )
        active_constraints = master.active_constraints

        candidate_evaluation = evaluate_objective_and_supergradient(
            model,
            master.vector,
        )
        step = 1.0
        candidate = master.vector

        if candidate_evaluation.value > current_evaluation.value + 1e-8:
            candidate, candidate_evaluation, step = feasible_line_search(
                model,
                current,
                master.vector,
                current_evaluation,
            )

        improvement = current_evaluation.value - candidate_evaluation.value
        relative_improvement = improvement / max(
            1.0,
            abs(current_evaluation.value),
        )

        equality_violation = model.max_equality_violation(candidate)
        sp_violation = model.max_sp_violation(candidate)
        elapsed = time.perf_counter() - iteration_start

        print(
            f"iter={iteration:03d}  "
            f"objective={candidate_evaluation.value:.12g}  "
            f"improvement={improvement:.3e}  "
            f"step={step:.3g}  "
            f"active_sp={active_constraints.size:,}  "
            f"cut_rounds={master.cut_rounds}  "
            f"eq={equality_violation:.2e}  "
            f"sp={sp_violation:.2e}  "
            f"seconds={elapsed:.2f}"
        )

        if equality_violation > 1e-7 or sp_violation > 1e-7:
            raise RuntimeError("The master solution is numerically infeasible.")

        if candidate_evaluation.value > current_evaluation.value + 1e-7:
            raise RuntimeError("The MM objective increased unexpectedly.")

        current = candidate
        current_evaluation = candidate_evaluation
        history.append(current_evaluation.value)

        if step == 0.0 or relative_improvement <= arguments.relative_tolerance:
            break

    save_solution(
        arguments.output,
        model,
        current,
        history,
        active_constraints,
    )

    print(f"saved solution         : {arguments.output}")
    print(f"final objective        : {current_evaluation.value:.12g}")
    print(f"RSD improvement        : {history[0] - history[-1]:.12g}")


if __name__ == "__main__":
    main()