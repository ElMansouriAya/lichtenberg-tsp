"""
benchmark.py
------------
Ablation & Comparative Study: isolates the contribution of each component
of the LA-seeded ILS hybrid by running THREE configurations, under the
exact same budget (pop=50, n_iter=200, ref=0.4), on all five TSPLIB
instances:

    1. Pure LA + Random Keys
       No 2-opt, no perturbation, no restarts. This is the raw
       continuous-to-discrete mapping: whatever LA's own update rule
       finds via argsort decoding, nothing else. This isolates whether
       Random Keys decoding alone is a sufficient discrete search
       mechanism for LA to find good tours.

    2. LA + 2-opt (No Kicks)
       2-opt polishes the global-best agent every iteration, but there is
       no double-bridge escape move and no stagnation-triggered restart.
       This isolates how much of the gain is "free" local-search polish
       vs. genuine escape from local optima -- this configuration is
       expected to plateau hard at whatever 2-opt-local-optimum it first
       reaches, by design.

    3. Full Hybrid LA-ILS
       2-opt-on-best + double-bridge kick on stagnation + partial
       population restart. The complete framework from prior sections.

Each configuration is run over multiple seeds per instance (mean/std
reported, NOT a single cherry-picked seed -- see prior discussion on
run-to-run variance). Outputs:

    - A markdown ablation table (mean gap % per config per instance,
      plus the per-instance improvement attributable to each component)
    - benchmark_ablation_convergence.png: a 5-panel figure (one per
      instance), each panel overlaying the three configurations' best-of-
      seeds convergence curves, so the professor can see directly where
      each configuration plateaus.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tsp_problem import get_instance
from la_optimization import LichtenbergAlgorithm

output_dir = "../docs/figures"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed hyperparameters -- IDENTICAL across all configs and instances. The
# point of an ablation is that only the presence/absence of components
# changes, never the budget.
# ---------------------------------------------------------------------------
POP_SIZE = 50
N_ITER = 200
REF = 0.4
SEED = 42
LF_PATH = "../data/LFND.mat"
N_SEEDS = 5

TWOOPT_PASSES = 60
RESTART_PATIENCE_FULL = 6
RESTART_FRACTION_FULL = 0.2

INSTANCES = ["eil51", "berlin52", "st70", "pr76", "rat99"]

# ---------------------------------------------------------------------------
# Configuration registry: each entry says which hooks to enable and what
# restart_patience to use. restart_patience=0 fully disables the stagnation
# mechanism in LichtenbergAlgorithm.optimize (see la_optimization.py).
# ---------------------------------------------------------------------------
CONFIGS = {
    "Pure LA + Random Keys": dict(use_two_opt=False, use_perturb=False, restart_patience=0),
    "LA + 2-opt (No Kicks)": dict(use_two_opt=True,  use_perturb=False, restart_patience=0),
    "Full Hybrid LA-ILS":    dict(use_two_opt=True,  use_perturb=True,  restart_patience=RESTART_PATIENCE_FULL),
}
CONFIG_ORDER = list(CONFIGS.keys())  # preserve display order


def run_single(name: str, seed: int, config_name: str):
    """Runs ONE (instance, seed, configuration) combination."""
    cfg = CONFIGS[config_name]
    problem, optimum = get_instance(name)
    dim = problem.n_cities

    def fitness_fn(keys: np.ndarray) -> float:
        return problem.fitness_from_keys(keys)

    two_opt_hook = None
    if cfg["use_two_opt"]:
        def two_opt_hook(best_keys: np.ndarray):
            tour = problem.decode_random_keys(best_keys)
            improved_tour, improved_len = problem.two_opt(
                tour, max_passes=TWOOPT_PASSES, first_improvement=False
            )
            improved_keys = problem.encode_tour_to_keys(improved_tour)
            return improved_keys, improved_len

    perturb_hook = None
    if cfg["use_perturb"]:
        def perturb_hook(best_keys: np.ndarray, rng: np.random.Generator):
            tour = problem.decode_random_keys(best_keys)
            kicked_tour = problem.double_bridge(tour, rng)
            polished_tour, polished_len = problem.two_opt(
                kicked_tour, max_passes=TWOOPT_PASSES, first_improvement=False
            )
            polished_keys = problem.encode_tour_to_keys(polished_tour)
            return polished_keys, polished_len

    la = LichtenbergAlgorithm(
        dim=dim, n_pop=POP_SIZE, n_iter=N_ITER, ref=REF,
        fitness_fn=fitness_fn, lb=0.0, ub=1.0, lf_path=LF_PATH, seed=seed,
    )

    t0 = time.time()
    best_keys, best_fit, history = la.optimize(
        verbose=False, two_opt_hook=two_opt_hook, perturb_hook=perturb_hook,
        restart_patience=cfg["restart_patience"], restart_fraction=RESTART_FRACTION_FULL,
    )
    elapsed = time.time() - t0

    # Reporting-time polish: ONLY apply if the configuration already uses
    # 2-opt internally (idempotent in that case -- best_fit is already
    # 2-opt-locally-optimal, so this just confirms it). For "Pure LA +
    # Random Keys", we deliberately do NOT apply any 2-opt here, because
    # the entire point of this configuration is to report what LA's own
    # continuous-to-discrete mapping produces on its own, unaided.
    if cfg["use_two_opt"]:
        best_tour = problem.decode_random_keys(best_keys)
        best_tour, best_fit = problem.two_opt(best_tour, max_passes=30, first_improvement=False)
    else:
        best_tour = problem.decode_random_keys(best_keys)

    gap_pct = 100.0 * (best_fit - optimum) / optimum

    return {
        "name": name, "config": config_name, "seed": seed, "n_cities": dim,
        "best": best_fit, "optimum": optimum, "gap_pct": gap_pct,
        "time_s": elapsed, "history": history, "best_tour": best_tour,
    }


def run_ablation():
    """Runs all (instance x config x seed) combinations, returns raw results."""
    all_runs = []
    for name in INSTANCES:
        print(f"\n=== {name} ===")
        for config_name in CONFIG_ORDER:
            seed_runs = []
            for s in range(N_SEEDS):
                r = run_single(name, seed=SEED + s, config_name=config_name)
                seed_runs.append(r)
            gaps = [r["gap_pct"] for r in seed_runs]
            times = [r["time_s"] for r in seed_runs]
            print(f"  {config_name:<26s} mean_gap={np.mean(gaps):6.3f}%  "
                  f"std={np.std(gaps):5.3f}  mean_time={np.mean(times):5.2f}s")
            all_runs.extend(seed_runs)
    return all_runs


def summarize(all_runs):
    """Aggregates raw per-seed runs into per-(instance, config) summaries."""
    summaries = {}
    for name in INSTANCES:
        for config_name in CONFIG_ORDER:
            runs = [r for r in all_runs if r["name"] == name and r["config"] == config_name]
            gaps = np.array([r["gap_pct"] for r in runs])
            times = np.array([r["time_s"] for r in runs])
            best_run = min(runs, key=lambda r: r["best"])
            summaries[(name, config_name)] = {
                "n_cities": runs[0]["n_cities"],
                "optimum": runs[0]["optimum"],
                "mean_gap_pct": gaps.mean(),
                "std_gap_pct": gaps.std(),
                "best_gap_pct": best_run["gap_pct"],
                "mean_time_s": times.mean(),
                "best_history": best_run["history"],
            }
    return summaries


def print_markdown_table_full_hybrid(summaries):
    """
    Reproduces the earlier per-instance summary table (mean/std/best gap,
    mean time) for the Full Hybrid LA-ILS configuration specifically --
    this is the "main result" table from before the ablation study was
    added, now using the SAME runs already collected for the ablation
    (no extra computation needed).
    """
    print(f"\n## Full Hybrid LA-ILS -- Summary (mean over {N_SEEDS} seeds per instance)\n")
    header = "| Instance | Cities | Known Optimum | Mean Gap (%) | Std (%) | Best Gap (%) | Mean Time (s) |"
    sep =    "|----------|--------|----------------|---------------|---------|--------------|----------------|"
    print(header)
    print(sep)
    mean_gaps = []
    for name in INSTANCES:
        s = summaries[(name, "Full Hybrid LA-ILS")]
        print(f"| {name} | {s['n_cities']} | {s['optimum']} | "
              f"{s['mean_gap_pct']:.3f} | {s['std_gap_pct']:.3f} | "
              f"{s['best_gap_pct']:.3f} | {s['mean_time_s']:.2f} |")
        mean_gaps.append(s["mean_gap_pct"])
    print(f"\n**Overall mean gap across instances:** {np.mean(mean_gaps):.3f}%")


def print_markdown_table(summaries):
    print("\n## Ablation Study: Contribution of Each Component")
    print(f"### (mean over {N_SEEDS} seeds, pop={POP_SIZE}, n_iter={N_ITER}, ref={REF}, identical budget across configs)\n")

    header = ("| Instance | Cities | Pure LA+RK Gap (%) | LA+2-opt Gap (%) | Full Hybrid Gap (%) | "
               "2-opt Gain (pp) | Kicks Gain (pp) |")
    sep = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)

    pure_gaps, twoopt_gaps, full_gaps = [], [], []
    for name in INSTANCES:
        s_pure = summaries[(name, "Pure LA + Random Keys")]
        s_2opt = summaries[(name, "LA + 2-opt (No Kicks)")]
        s_full = summaries[(name, "Full Hybrid LA-ILS")]

        twoopt_gain = s_pure["mean_gap_pct"] - s_2opt["mean_gap_pct"]
        kicks_gain = s_2opt["mean_gap_pct"] - s_full["mean_gap_pct"]

        print(f"| {name} | {s_pure['n_cities']} | {s_pure['mean_gap_pct']:.2f} | "
              f"{s_2opt['mean_gap_pct']:.2f} | {s_full['mean_gap_pct']:.2f} | "
              f"{twoopt_gain:+.2f} | {kicks_gain:+.2f} |")

        pure_gaps.append(s_pure["mean_gap_pct"])
        twoopt_gaps.append(s_2opt["mean_gap_pct"])
        full_gaps.append(s_full["mean_gap_pct"])

    print(f"\n**Overall mean gap -- Pure LA+RK:** {np.mean(pure_gaps):.2f}%  |  "
          f"**LA+2-opt:** {np.mean(twoopt_gaps):.2f}%  |  "
          f"**Full Hybrid:** {np.mean(full_gaps):.2f}%")
    print(f"\n**Interpretation:** 'pp' = percentage-point reduction in mean optimality gap "
          f"attributable to adding that component on top of the previous configuration. "
          f"'2-opt Gain' isolates the contribution of local search polish alone; "
          f"'Kicks Gain' isolates the additional contribution of double-bridge escape + "
          f"restarts on top of 2-opt. A near-zero or negative 'Kicks Gain' on an instance "
          f"would indicate 2-opt-on-best already reached that instance's effective floor "
          f"within this iteration budget.")


def plot_ablation_convergence(summaries, out_path=f"{output_dir}/benchmark_ablation_convergence.png"):
    fig, axes = plt.subplots(1, len(INSTANCES), figsize=(5 * len(INSTANCES), 4.2), sharey=False)
    colors = {"Pure LA + Random Keys": "tab:red",
              "LA + 2-opt (No Kicks)": "tab:orange",
              "Full Hybrid LA-ILS": "tab:green"}

    for ax, name in zip(axes, INSTANCES):
        optimum = summaries[(name, CONFIG_ORDER[0])]["optimum"]
        for config_name in CONFIG_ORDER:
            s = summaries[(name, config_name)]
            gap_curve = 100.0 * (np.array(s["best_history"]) - optimum) / optimum
            ax.plot(gap_curve, label=config_name, color=colors[config_name], lw=1.6)
        ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{name} (n={summaries[(name, CONFIG_ORDER[0])]['n_cities']})")
        ax.set_xlabel("Iteration")
        ax.set_yscale("symlog")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Gap to known optimum (%)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Ablation: Pure LA+RandomKeys vs. LA+2-opt vs. Full Hybrid LA-ILS "
                 "(best-of-seeds run per configuration)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved ablation convergence plot to {out_path}")


def main():
    all_runs = run_ablation()
    summaries = summarize(all_runs)
    print_markdown_table_full_hybrid(summaries)
    print_markdown_table(summaries)
    plot_ablation_convergence(summaries)


if __name__ == "__main__":
    main()
