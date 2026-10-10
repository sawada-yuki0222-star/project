from __future__ import annotations

"""Symmetry-reduced random-assignment model for n=4.

This module builds the probability-matrix side of the model:

* strict preference profiles;
* anonymity/neutrality orbit representatives;
* representative stabilizer constraints;
* bistochasticity constraints;
* strategy-proofness constraints;
* the RSD initial mechanism;
* the inner LP used to measure ordinal-efficiency violation.

The convention is that rows index agents and columns index objects.
Objects and agents are represented by integers 0, ..., n-1.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations
from pathlib import Path
import pickle
from typing import Iterable, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import linprog


Ranking: TypeAlias = tuple[int, ...]
Profile: TypeAlias = tuple[Ranking, ...]
Permutation: TypeAlias = tuple[int, ...]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

MODEL_VERSION = 1


@dataclass(frozen=True)
class CanonicalForm:
    """Canonical representative and the map back to the original labels.

    The returned permutations satisfy

        original_profile = (rep_to_agent, rep_to_object) representative.

    Therefore, for a representative allocation R,

        P_original[rep_to_agent[i], rep_to_object[a]] = R[i, a].
    """

    representative: Profile
    rep_to_agent: Permutation
    rep_to_object: Permutation


@dataclass(frozen=True)
class OEResult:
    value: float
    dominating_allocation: FloatArray
    supergradient: FloatArray
    max_primal_violation: float


@dataclass
class ModelData:
    version: int
    n: int
    rankings: tuple[Ranking, ...]
    permutations: tuple[Permutation, ...]
    representatives: tuple[Profile, ...]
    orbit_weights: IntArray
    representative_to_id: dict[Profile, int]
    cumulative_matrices: tuple[sparse.csr_matrix, ...]
    score_vectors: FloatArray
    inner_equalities: sparse.csr_matrix
    inner_rhs: FloatArray
    equality_matrix: sparse.csr_matrix
    equality_rhs: FloatArray
    sp_matrix: sparse.csr_matrix
    rsd_initial: FloatArray

    @property
    def number_of_variables(self) -> int:
        return len(self.representatives) * self.n * self.n

    @property
    def number_of_profiles(self) -> int:
        return int(self.orbit_weights.sum())

    @property
    def normalized_weights(self) -> FloatArray:
        return self.orbit_weights.astype(np.float64) / self.number_of_profiles

    def representative_slice(self, representative_id: int) -> slice:
        width = self.n * self.n
        start = representative_id * width
        return slice(start, start + width)

    def representative_matrix(
        self,
        vector: FloatArray,
        representative_id: int,
    ) -> FloatArray:
        return vector[self.representative_slice(representative_id)].reshape(
            self.n,
            self.n,
        )

    def allocation_for_profile(
        self,
        vector: FloatArray,
        profile: Profile,
    ) -> FloatArray:
        canonical = canonicalize(profile, self.permutations)
        representative_id = self.representative_to_id[
            canonical.representative
        ]
        representative_matrix = self.representative_matrix(
            vector,
            representative_id,
        )

        allocation = np.zeros((self.n, self.n), dtype=np.float64)
        for representative_agent, original_agent in enumerate(
            canonical.rep_to_agent
        ):
            for representative_object, original_object in enumerate(
                canonical.rep_to_object
            ):
                allocation[original_agent, original_object] = (
                    representative_matrix[
                        representative_agent,
                        representative_object,
                    ]
                )
        return allocation

    def max_sp_violation(self, vector: FloatArray) -> float:
        if self.sp_matrix.shape[0] == 0:
            return 0.0
        values = np.asarray(self.sp_matrix @ vector).reshape(-1)
        return float(max(0.0, -values.min(initial=0.0)))

    def max_equality_violation(self, vector: FloatArray) -> float:
        residual = np.asarray(
            self.equality_matrix @ vector - self.equality_rhs
        ).reshape(-1)
        return float(np.max(np.abs(residual), initial=0.0))

    def min_probability(self, vector: FloatArray) -> float:
        return float(np.min(vector))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def inverse_permutation(permutation: Permutation) -> Permutation:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[target] = source
    return tuple(inverse)


def apply_group(
    profile: Profile,
    agent_permutation: Permutation,
    object_permutation: Permutation,
) -> Profile:
    """Apply agent and object relabeling to a profile."""

    n = len(profile)
    transformed: list[Ranking | None] = [None] * n
    for old_agent, ranking in enumerate(profile):
        new_agent = agent_permutation[old_agent]
        transformed[new_agent] = tuple(
            object_permutation[object_id] for object_id in ranking
        )

    if any(ranking is None for ranking in transformed):
        raise ValueError("The supplied agent map is not a permutation.")
    return tuple(ranking for ranking in transformed if ranking is not None)


def canonical_representative(
    profile: Profile,
    object_permutations: Iterable[Permutation],
) -> Profile:
    """Return the lexicographically smallest anonymous/neutral relabeling."""

    best: Profile | None = None
    for object_to_canonical in object_permutations:
        candidate = tuple(
            sorted(
                tuple(object_to_canonical[object_id] for object_id in ranking)
                for ranking in profile
            )
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise ValueError("No object permutations were supplied.")
    return best


def canonicalize(
    profile: Profile,
    object_permutations: Iterable[Permutation],
) -> CanonicalForm:
    """Canonicalize a profile and retain a deterministic inverse relabeling."""

    n = len(profile)
    best_key: tuple[Profile, Permutation, Permutation] | None = None

    for object_to_canonical in object_permutations:
        transformed = [
            tuple(object_to_canonical[object_id] for object_id in ranking)
            for ranking in profile
        ]
        ordered_old_agents = tuple(
            sorted(range(n), key=lambda agent: (transformed[agent], agent))
        )
        representative = tuple(
            transformed[old_agent] for old_agent in ordered_old_agents
        )
        representative_to_agent = ordered_old_agents
        representative_to_object = inverse_permutation(object_to_canonical)

        key = (
            representative,
            representative_to_agent,
            representative_to_object,
        )
        if best_key is None or key < best_key:
            best_key = key

    if best_key is None:
        raise ValueError("No object permutations were supplied.")

    result = CanonicalForm(
        representative=best_key[0],
        rep_to_agent=best_key[1],
        rep_to_object=best_key[2],
    )

    reconstructed: list[Ranking | None] = [None] * n
    for representative_agent, original_agent in enumerate(result.rep_to_agent):
        reconstructed[original_agent] = tuple(
            result.rep_to_object[object_id]
            for object_id in result.representative[representative_agent]
        )
    if tuple(reconstructed) != profile:
        raise AssertionError("Canonicalization map failed reconstruction.")

    return result


def enumerate_representatives(
    rankings: tuple[Ranking, ...],
    object_permutations: tuple[Permutation, ...],
    n: int,
) -> tuple[Profile, ...]:
    """Enumerate anonymity classes first, then quotient by neutrality."""

    representatives: set[Profile] = set()
    for anonymous_profile in combinations_with_replacement(rankings, n):
        representatives.add(
            canonical_representative(
                anonymous_profile,
                object_permutations,
            )
        )
    return tuple(sorted(representatives))


def stabilizers(
    profile: Profile,
    permutations_: tuple[Permutation, ...],
) -> tuple[tuple[Permutation, Permutation], ...]:
    output: list[tuple[Permutation, Permutation]] = []
    for agent_permutation in permutations_:
        for object_permutation in permutations_:
            if (
                apply_group(
                    profile,
                    agent_permutation,
                    object_permutation,
                )
                == profile
            ):
                output.append((agent_permutation, object_permutation))
    return tuple(output)


def cell_orbit_roots(
    profile: Profile,
    permutations_: tuple[Permutation, ...],
) -> tuple[int, ...]:
    n = len(profile)
    union_find = UnionFind(n * n)

    for agent_permutation, object_permutation in stabilizers(
        profile,
        permutations_,
    ):
        for agent in range(n):
            for object_id in range(n):
                source = agent * n + object_id
                target = (
                    agent_permutation[agent] * n
                    + object_permutation[object_id]
                )
                union_find.union(source, target)

    roots = tuple(union_find.find(cell) for cell in range(n * n))
    canonical_root: dict[int, int] = {}
    for cell, root in enumerate(roots):
        canonical_root[root] = min(canonical_root.get(root, cell), cell)
    return tuple(canonical_root[root] for root in roots)


def orbit_weights(
    representatives: tuple[Profile, ...],
    permutations_: tuple[Permutation, ...],
) -> IntArray:
    group_size = len(permutations_) ** 2
    weights = []
    for profile in representatives:
        stabilizer_size = len(stabilizers(profile, permutations_))
        if group_size % stabilizer_size != 0:
            raise AssertionError("Orbit-stabilizer divisibility failed.")
        weights.append(group_size // stabilizer_size)
    return np.asarray(weights, dtype=np.int64)


def cumulative_matrix(profile: Profile) -> sparse.csr_matrix:
    """A_theta with the redundant final rank t=n omitted."""

    n = len(profile)
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []

    row = 0
    for agent, ranking in enumerate(profile):
        for cutoff in range(1, n):
            for object_id in ranking[:cutoff]:
                row_indices.append(row)
                column_indices.append(agent * n + object_id)
                coefficients.append(1.0)
            row += 1

    return sparse.csr_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(n * (n - 1), n * n),
        dtype=np.float64,
    )


def bistochastic_equalities(
    n: int,
) -> tuple[sparse.csr_matrix, FloatArray]:
    """Independent row/column equations for an n by n bistochastic matrix."""

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []

    row = 0
    for agent in range(n):
        for object_id in range(n):
            row_indices.append(row)
            column_indices.append(agent * n + object_id)
            coefficients.append(1.0)
        row += 1

    # The last column equation is implied by all row equations and the
    # first n-1 column equations.
    for object_id in range(n - 1):
        for agent in range(n):
            row_indices.append(row)
            column_indices.append(agent * n + object_id)
            coefficients.append(1.0)
        row += 1

    matrix = sparse.csr_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(2 * n - 1, n * n),
        dtype=np.float64,
    )
    rhs = np.ones(2 * n - 1, dtype=np.float64)
    return matrix, rhs


def rsd_allocation(
    profile: Profile,
    priority_orders: tuple[Permutation, ...],
) -> FloatArray:
    """Exact RSD marginal assignment matrix, averaged over all priorities."""

    n = len(profile)
    allocation = np.zeros((n, n), dtype=np.float64)

    for priority_order in priority_orders:
        available = set(range(n))
        for agent in priority_order:
            selected_object = next(
                object_id
                for object_id in profile[agent]
                if object_id in available
            )
            allocation[agent, selected_object] += 1.0
            available.remove(selected_object)

    allocation /= len(priority_orders)
    return allocation


def build_global_equalities(
    representatives: tuple[Profile, ...],
    permutations_: tuple[Permutation, ...],
    n: int,
) -> tuple[sparse.csr_matrix, FloatArray]:
    """Bistochasticity plus stabilizer invariance for every representative."""

    number_of_variables = len(representatives) * n * n
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    rhs: list[float] = []
    row = 0

    for representative_id, profile in enumerate(representatives):
        offset = representative_id * n * n

        for agent in range(n):
            for object_id in range(n):
                row_indices.append(row)
                column_indices.append(offset + agent * n + object_id)
                coefficients.append(1.0)
            rhs.append(1.0)
            row += 1

        for object_id in range(n - 1):
            for agent in range(n):
                row_indices.append(row)
                column_indices.append(offset + agent * n + object_id)
                coefficients.append(1.0)
            rhs.append(1.0)
            row += 1

        roots = cell_orbit_roots(profile, permutations_)
        for cell, root in enumerate(roots):
            if cell == root:
                continue
            row_indices.extend((row, row))
            column_indices.extend((offset + cell, offset + root))
            coefficients.extend((1.0, -1.0))
            rhs.append(0.0)
            row += 1

    matrix = sparse.csr_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(row, number_of_variables),
        dtype=np.float64,
    )
    return matrix, np.asarray(rhs, dtype=np.float64)


def build_sp_constraints(
    representatives: tuple[Profile, ...],
    representative_to_id: dict[Profile, int],
    rankings: tuple[Ranking, ...],
    permutations_: tuple[Permutation, ...],
    n: int,
) -> sparse.csr_matrix:
    """Build all nontrivial SP inequalities C x >= 0 on representatives."""

    number_of_variables = len(representatives) * n * n
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    row = 0

    # Deduplicate because symmetric deviations can produce the same inequality.
    seen_rows: set[tuple[tuple[int, int], ...]] = set()

    for truthful_id, truthful_profile in enumerate(representatives):
        truthful_offset = truthful_id * n * n

        for agent in range(n):
            true_ranking = truthful_profile[agent]
            for reported_ranking in rankings:
                if reported_ranking == true_ranking:
                    continue

                deviating_profile_list = list(truthful_profile)
                deviating_profile_list[agent] = reported_ranking
                deviating_profile = tuple(deviating_profile_list)

                canonical = canonicalize(deviating_profile, permutations_)
                deviating_id = representative_to_id[
                    canonical.representative
                ]
                deviating_offset = deviating_id * n * n

                original_agent_to_rep = inverse_permutation(
                    canonical.rep_to_agent
                )
                original_object_to_rep = inverse_permutation(
                    canonical.rep_to_object
                )
                deviating_rep_agent = original_agent_to_rep[agent]

                cumulative_objects: list[int] = []
                for cutoff in range(1, n):
                    cumulative_objects.append(true_ranking[cutoff - 1])
                    row_coefficients: dict[int, int] = {}

                    for original_object in cumulative_objects:
                        truthful_column = (
                            truthful_offset
                            + agent * n
                            + original_object
                        )
                        row_coefficients[truthful_column] = (
                            row_coefficients.get(truthful_column, 0) + 1
                        )

                        deviating_rep_object = original_object_to_rep[
                            original_object
                        ]
                        deviating_column = (
                            deviating_offset
                            + deviating_rep_agent * n
                            + deviating_rep_object
                        )
                        row_coefficients[deviating_column] = (
                            row_coefficients.get(deviating_column, 0) - 1
                        )

                    signature = tuple(
                        sorted(
                            (column, coefficient)
                            for column, coefficient in row_coefficients.items()
                            if coefficient != 0
                        )
                    )
                    if not signature or signature in seen_rows:
                        continue
                    seen_rows.add(signature)

                    for column, coefficient in signature:
                        row_indices.append(row)
                        column_indices.append(column)
                        coefficients.append(float(coefficient))
                    row += 1

    return sparse.csr_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(row, number_of_variables),
        dtype=np.float64,
    )


def solve_oe_violation(
    cumulative: sparse.csr_matrix,
    score: FloatArray,
    inner_equalities: sparse.csr_matrix,
    inner_rhs: FloatArray,
    allocation_vector: FloatArray,
    *,
    tolerance: float = 1e-9,
) -> OEResult:
    """Solve the exact inner LP and return one supergradient of v_theta."""

    result = linprog(
        c=-score,
        A_ub=-cumulative,
        b_ub=-(cumulative @ allocation_vector),
        A_eq=inner_equalities,
        b_eq=inner_rhs,
        bounds=(0.0, 1.0),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": tolerance,
            "dual_feasibility_tolerance": tolerance,
        },
    )

    if not result.success:
        raise RuntimeError(
            "The ordinal-efficiency LP failed: "
            f"status={result.status}, message={result.message}"
        )

    dominating_allocation = np.asarray(result.x, dtype=np.float64)
    value = float(-result.fun - score @ allocation_vector)

    # For SciPy's minimization form, ineqlin.marginals is the derivative
    # with respect to b_ub. Since b_ub = -A p, a supergradient of
    # v(p) = -opt_value(p) - s^T p is A^T marginals - s.
    marginals = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    supergradient = (
        np.asarray(cumulative.T @ marginals).reshape(-1) - score
    )

    dominance_violation = float(
        np.max(
            np.maximum(
                cumulative @ allocation_vector
                - cumulative @ dominating_allocation,
                0.0,
            ),
            initial=0.0,
        )
    )
    equality_violation = float(
        np.max(
            np.abs(
                inner_equalities @ dominating_allocation - inner_rhs
            ),
            initial=0.0,
        )
    )
    lower_bound_violation = max(
        0.0,
        float(-np.min(dominating_allocation)),
    )
    upper_bound_violation = max(
        0.0,
        float(np.max(dominating_allocation) - 1.0),
    )

    max_primal_violation = max(
        dominance_violation,
        equality_violation,
        lower_bound_violation,
        upper_bound_violation,
    )

    if value < -100.0 * tolerance:
        raise RuntimeError(
            f"The OE violation should be nonnegative, got {value}."
        )

    return OEResult(
        value=max(0.0, value),
        dominating_allocation=dominating_allocation,
        supergradient=supergradient,
        max_primal_violation=max_primal_violation,
    )


def build_model(n: int = 4) -> ModelData:
    if n != 4:
        raise NotImplementedError(
            "This implementation is deliberately specialized to n=4."
        )

    rankings = tuple(permutations(range(n)))
    permutations_ = rankings
    representatives = enumerate_representatives(
        rankings,
        permutations_,
        n,
    )
    representative_to_id = {
        profile: profile_id
        for profile_id, profile in enumerate(representatives)
    }

    weights = orbit_weights(representatives, permutations_)
    if len(representatives) != 762:
        raise AssertionError(
            f"Expected 762 representatives, found {len(representatives)}."
        )
    if int(weights.sum()) != 24**4:
        raise AssertionError(
            f"Orbit weights sum to {weights.sum()}, not {24**4}."
        )

    cumulative_matrices = tuple(
        cumulative_matrix(profile) for profile in representatives
    )
    score_vectors = np.vstack(
        [
            np.asarray(
                cumulative.T @ np.ones(cumulative.shape[0])
            ).reshape(-1)
            for cumulative in cumulative_matrices
        ]
    )
    inner_equalities, inner_rhs = bistochastic_equalities(n)

    equality_matrix, equality_rhs = build_global_equalities(
        representatives,
        permutations_,
        n,
    )
    sp_matrix = build_sp_constraints(
        representatives,
        representative_to_id,
        rankings,
        permutations_,
        n,
    )

    rsd_initial = np.concatenate(
        [
            rsd_allocation(profile, permutations_).reshape(-1)
            for profile in representatives
        ]
    )

    model = ModelData(
        version=MODEL_VERSION,
        n=n,
        rankings=rankings,
        permutations=permutations_,
        representatives=representatives,
        orbit_weights=weights,
        representative_to_id=representative_to_id,
        cumulative_matrices=cumulative_matrices,
        score_vectors=score_vectors,
        inner_equalities=inner_equalities,
        inner_rhs=inner_rhs,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        sp_matrix=sp_matrix,
        rsd_initial=rsd_initial,
    )

    if model.max_equality_violation(rsd_initial) > 1e-10:
        raise AssertionError("RSD violates an equality constraint.")
    if model.max_sp_violation(rsd_initial) > 1e-10:
        raise AssertionError("RSD violates strategy-proofness constraints.")

    return model


def save_model(model: ModelData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> ModelData:
    path = Path(path)
    with path.open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, ModelData):
        raise TypeError("The cache does not contain ModelData.")
    if model.version != MODEL_VERSION:
        raise ValueError(
            f"Cache version {model.version} does not match "
            f"code version {MODEL_VERSION}."
        )
    return model


def build_or_load_model(
    cache_path: str | Path,
    *,
    rebuild: bool = False,
) -> ModelData:
    path = Path(cache_path)
    if path.exists() and not rebuild:
        return load_model(path)
    model = build_model(n=4)
    save_model(model, path)
    return model