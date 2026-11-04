from __future__ import annotations

from pathlib import Path

import numpy as np

from matrix import (
    build_or_load_model,
    cumulative_matrix,
    solve_oe_violation,
)


OBJECT_LABELS = ("a", "b", "c", "d")
OBJECT_TO_ID = {
    label: object_id
    for object_id, label in enumerate(OBJECT_LABELS)
}


def parse_ranking(text: str) -> tuple[int, ...]:
    """Convert 'abcd' into (0, 1, 2, 3)."""
    normalized = text.replace(">", "").replace(" ", "").lower()

    if len(normalized) != 4:
        raise ValueError(
            "Each ranking must contain exactly four objects."
        )

    try:
        ranking = tuple(
            OBJECT_TO_ID[label]
            for label in normalized
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown object label: {error.args[0]}"
        ) from error

    if len(set(ranking)) != 4:
        raise ValueError(
            "Each of a, b, c, d must appear exactly once."
        )

    return ranking


def profile_to_text(
    profile: tuple[tuple[int, ...], ...],
) -> str:
    rows = []

    for agent, ranking in enumerate(profile, start=1):
        ranking_text = " > ".join(
            OBJECT_LABELS[object_id]
            for object_id in ranking
        )
        rows.append(f"agent {agent}: {ranking_text}")

    return "\n".join(rows)


def main() -> None:
    cache_path = Path("model_cache.pkl")
    solution_path = Path("solution.npz")

    if not solution_path.exists():
        raise FileNotFoundError(
            "solution.npz was not found. Run main.py first."
        )

    model = build_or_load_model(cache_path)
    solution = np.load(solution_path)

    mechanism_vector = np.asarray(
        solution["mechanism_vector"],
        dtype=np.float64,
    )
    rsd_vector = np.asarray(
        solution["rsd_initial"],
        dtype=np.float64,
    )

    if mechanism_vector.shape != (
        model.number_of_variables,
    ):
        raise ValueError(
            "The solution shape does not match the model cache."
        )

    # ここを調べたい選好プロファイルに変更してください。
    profile = (
        parse_ranking("abcd"),
        parse_ranking("acbd"),
        parse_ranking("dcab"),
        parse_ranking("cdab"),
    )

    optimized_matrix = model.allocation_for_profile(
        mechanism_vector,
        profile,
    )
    rsd_matrix = model.allocation_for_profile(
        rsd_vector,
        profile,
    )

    cumulative = cumulative_matrix(profile)
    score = np.asarray(
        cumulative.T @ np.ones(
            cumulative.shape[0],
            dtype=np.float64,
        )
    ).reshape(-1)

    oe_result = solve_oe_violation(
        cumulative=cumulative,
        score=score,
        inner_equalities=model.inner_equalities,
        inner_rhs=model.inner_rhs,
        allocation_vector=optimized_matrix.reshape(-1),
    )

    dominating_matrix = (
        oe_result.dominating_allocation.reshape(4, 4)
    )

    np.set_printoptions(
        precision=10,
        suppress=True,
    )

    print("Preference profile")
    print("------------------")
    print(profile_to_text(profile))

    print("\nOptimized mechanism matrix P*(theta)")
    print("-------------------------------------")
    print(optimized_matrix)

    print("\nRow sums")
    print(optimized_matrix.sum(axis=1))

    print("\nColumn sums")
    print(optimized_matrix.sum(axis=0))

    print("\nRSD matrix")
    print("----------")
    print(rsd_matrix)

    print("\nDifference from RSD")
    print("-------------------")
    print(optimized_matrix - rsd_matrix)

    print("\nOrdinal-efficiency violation")
    print("----------------------------")
    print(f"{oe_result.value:.12g}")

    print("\nBest dominating matrix Q*")
    print("-------------------------")
    print(dominating_matrix)

    np.savetxt(
        "optimized_matrix.csv",
        optimized_matrix,
        delimiter=",",
        header="a,b,c,d",
        comments="",
    )

    print(
        "\nSaved the selected matrix to "
        "optimized_matrix.csv"
    )


if __name__ == "__main__":
    main()