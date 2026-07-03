"""
MAIN.py
-------
Entry point: faithful Lichtenberg Algorithm (M=0, loads the author's real
LFND.mat) + Random Keys decoding + a memetic 2-opt step applied ONLY to the
iteration's global-best agent, applied to the berlin52 TSPLIB instance.

Mirrors the structure/intent of the author's MAIN.m, with hyperparameters
matching the welded-beam example's spirit but sized for this problem:
    pop = 50, n_iter = 200, ref = 0.4

Known optimum for berlin52: 7542
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tsp_problem import TSPProblem
from la_optimization import LichtenbergAlgorithm

output_dir = "../docs/figures"
os.makedirs(output_dir, exist_ok=True)
# ---------------------------------------------------------------------------
# Hyperparameters (as in MAIN.m: pop, n_iter, ref -- M is fixed to 0 here
# since we load the author's official LFND.mat rather than regenerating it)
# ---------------------------------------------------------------------------

POP_SIZE = 50
N_ITER = 200
REF = 0.4
SEED = 42
KNOWN_OPTIMUM = 7542
LF_PATH = "../data/LFND.mat"

TWOOPT_PASSES = 60   # effectively run 2-opt to local convergence (cheap at n=52)


def main():
    problem = TSPProblem()
    dim = problem.n_cities  # 52

    def fitness_fn(keys: np.ndarray) -> float:
        return problem.fitness_from_keys(keys)

    def two_opt_hook(best_keys: np.ndarray):
        """
        Memetic step: decode the CURRENT GLOBAL BEST agent's Random-Keys
        vector into a tour, run 2-opt on it, and re-encode the improved
        tour back into a consistent key vector. Applied ONLY to the
        global best (never to the rest of the population) so LA's own
        stochastic search remains the dominant driver and 2-opt acts as a
        cheap local polish, not a random-restart 2-opt in disguise.
        """
        tour = problem.decode_random_keys(best_keys)
        improved_tour, improved_len = problem.two_opt(
            tour, max_passes=TWOOPT_PASSES, first_improvement=False
        )
        improved_keys = problem.encode_tour_to_keys(improved_tour)
        return improved_keys, improved_len

    def perturb_hook(best_keys: np.ndarray, rng: np.random.Generator):
        """
        Stagnation escape move (Iterated Local Search-style): double-bridge
        kick the incumbent's tour, then 2-opt the kicked tour back to local
        optimality. Double-bridge is a 4-opt move that plain 2-opt cannot
        reverse in a single step, which is precisely why it's the standard
        mechanism for escaping 2-opt local optima on TSP.
        """
        tour = problem.decode_random_keys(best_keys)
        kicked_tour = problem.double_bridge(tour, rng)
        polished_tour, polished_len = problem.two_opt(
            kicked_tour, max_passes=TWOOPT_PASSES, first_improvement=False
        )
        polished_keys = problem.encode_tour_to_keys(polished_tour)
        return polished_keys, polished_len

    la = LichtenbergAlgorithm(
        dim=dim,
        n_pop=POP_SIZE,
        n_iter=N_ITER,
        ref=REF,
        fitness_fn=fitness_fn,
        lb=0.0,
        ub=1.0,
        lf_path=LF_PATH,
        seed=SEED,
    )

    print(f"Running LA (M=0, official LFND.mat) + Random Keys + 2-opt-on-best "
          f"on berlin52 (pop={POP_SIZE}, n_iter={N_ITER}, ref={REF})")
    t0 = time.time()
    best_keys, best_fit, history = la.optimize(
        verbose=True, two_opt_hook=two_opt_hook, perturb_hook=perturb_hook,
        restart_patience=6, restart_fraction=0.2,
    )
    elapsed = time.time() - t0

    best_tour = problem.decode_random_keys(best_keys)
    # final exhaustive 2-opt polish on the overall best, for reporting
    best_tour, best_fit = problem.two_opt(best_tour, max_passes=30, first_improvement=False)

    gap_pct = 100.0 * (best_fit - KNOWN_OPTIMUM) / KNOWN_OPTIMUM

    print("\n--- Results ---------------------------------------------")
    print(f"Best tour length : {best_fit:.2f}")
    print(f"Known optimum    : {KNOWN_OPTIMUM}")
    print(f"Optimality gap   : {gap_pct:.3f} %")
    print(f"Elapsed time     : {elapsed:.2f} s")
    print(f"Best tour        : {best_tour.tolist()}")

    # --- plots -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(history, lw=1.5)
    axes[0].axhline(KNOWN_OPTIMUM, color="red", ls="--", lw=1, label="Optimum (7542)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best tour length")
    axes[0].set_title("LA convergence on berlin52")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    coords = problem.coords
    ordered = coords[np.append(best_tour, best_tour[0])]
    axes[1].plot(ordered[:, 0], ordered[:, 1], "-o", ms=4, lw=1)
    axes[1].set_title(f"Best route found ({best_fit:.1f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence.png", dpi=150)
    print("\nSaved plot to ../docs/figures/convergence.png")


if __name__ == "__main__":
    main()
