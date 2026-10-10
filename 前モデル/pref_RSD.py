from __future__ import annotations

from fractions import Fraction
from itertools import islice, permutations, product
from math import factorial
from typing import Iterable, List, Sequence


Preference = List[int]
Profile = List[Preference]


def _check_profile(preferences: Profile) -> None:
    n = len(preferences)
    expected = list(range(n))

    if any(len(pref) != n for pref in preferences):
        raise ValueError("Each agent must rank exactly n objects.")
    if any(sorted(pref) != expected for pref in preferences):
        raise ValueError("Each preference must be a permutation of 0..n-1.")


def serial_dictatorship(preferences: Profile, order: Sequence[int]) -> List[int]:
    """
    preferences: preference profile
    order: dictator order
    return: assignment[i] = object assigned to agent i
    """
    n = len(preferences)
    remaining = [True] * n
    assignment = [-1] * n

    for i in order:
        for o in preferences[i]:
            if remaining[o]:
                assignment[i] = o
                remaining[o] = False
                break

        if assignment[i] == -1:
            raise ValueError("No remaining object found; check input.")

    return assignment


def rsd_assignment_matrix(preferences: Profile, exact: bool = True):
    """
    Compute the random assignment matrix induced by RSD.
    P[i][o] is the probability that agent i receives object o.
    """
    _check_profile(preferences)
    n = len(preferences)
    m = factorial(n)

    if exact:
        P = [[Fraction(0, 1) for _ in range(n)] for _ in range(n)]
        per_order_weight = Fraction(1, m)
    else:
        P = [[0.0 for _ in range(n)] for _ in range(n)]
        per_order_weight = 1.0 / m

    # Do not materialize list(permutations(...)); this keeps memory O(n^2).
    for order in permutations(range(n)):
        assignment = serial_dictatorship(preferences, order)
        for i, o in enumerate(assignment):
            P[i][o] += per_order_weight

    return P


def all_preferences(n: int) -> List[Preference]:
    """Enumerate all strict preference orders for one agent."""
    return [list(p) for p in permutations(range(n))]


def all_preference_profiles(n: int) -> Iterable[Profile]:
    """
    Enumerate all preference profiles.
    There are (n!)^n profiles.
    """
    prefs = all_preferences(n)
    for profile in product(prefs, repeat=n):
        yield [list(pref) for pref in profile]


def count_preference_profiles(n: int) -> int:
    """Return the number of all preference profiles: (n!)^n."""
    return factorial(n) ** n


def pretty_print_matrix(P) -> None:
    for row in P:
        print("  ".join(str(x) for x in row))


def check_doubly_stochastic(P):
    n = len(P)
    row_sums = [sum(P[i][j] for j in range(n)) for i in range(n)]
    col_sums = [sum(P[i][j] for i in range(n)) for j in range(n)]
    return row_sums, col_sums


def pretty_print_profile(profile: Profile) -> None:
    print("preferences:")
    for i, pref in enumerate(profile):
        print(f"  agent {i}: {pref}")


def enumerate_all_rsd(n: int, exact: bool = True) -> Iterable[tuple[Profile, list]]:
    """Yield (profile, RSD matrix) for every preference profile."""
    for profile in all_preference_profiles(n):
        yield profile, rsd_assignment_matrix(profile, exact=exact)


if __name__ == "__main__":
    n = 4

    total = count_preference_profiles(n)
    print(f"n = {n}")
    print(f"number of preference profiles = {total}")
    print()

    for idx, (profile, P) in enumerate(islice(enumerate_all_rsd(n, exact=True), 3), start=1):
        print(f"===== profile #{idx} =====")
        pretty_print_profile(profile)
        print("RSD assignment matrix P (exact Fractions):")
        pretty_print_matrix(P)
        rs, cs = check_doubly_stochastic(P)
        print("Row sums:", rs)
        print("Col sums:", cs)
        print()