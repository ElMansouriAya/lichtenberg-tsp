"""
la_optimization.py
-------------------
Faithful Python translation of the author's official MATLAB source
(LA_optimization.m) for the Lichtenberg Algorithm (LA), specialized to the
M=0 case (load a precomputed Lichtenberg Figure from LFND.mat instead of
regenerating it via the DLA-style random walk in LA_figure).

This module intentionally mirrors the MATLAB control flow variable-for-
variable so it can be checked line-by-line against LA_optimization.m:

    MATLAB                              Python (this file)
    ----------------------------------- -----------------------------------
    LA_figure(d,Np,Rc,S_c,0)            load_LF()  (just loads LFND.mat)
    LA_points(K,LB,UB,x0,scale,d)       la_points(K, lb, ub, x0, scale, d)
    bound_check(s,LB,UB)                bound_check(s, lb, ub)
    LA_optimization(...)                LichtenbergAlgorithm.optimize(...)

Key fidelity notes (verified against LA_optimization.m and the real
LFND.mat supplied by the author):
  1. LFND.mat stores the cloud under the variable name 'LFND', shape
     (Np, 2) = (7792, 2), as RAW pixel coordinates from the DLA simulation
     (NOT pre-centered at the origin; values range roughly [150, 465]).
     The algorithm's own scale/recenter step in la_points() handles this
     -- we must NOT renormalize the cloud ourselves before passing it in,
     exactly as the MATLAB code doesn't.
  2. For d > 3, MATLAB loops `for i=1:2:d+2`, rotating the SAME 2D cloud K
     by a fresh random angle 'gama' for every consecutive pair of output
     columns, then trims the (d+2)-wide buffer down to d columns. This
     means one rotation/pair is computed in excess and discarded -- kept
     here exactly as in the source, not "cleaned up", for fidelity.
  3. Scale is computed PER DIMENSION from the *unscaled* column range,
     then applied; the centroid Pcc and resulting delta are computed
     AFTER scaling (two separate loops in MATLAB, preserved as two
     separate vectorized passes here).
  4. MATLAB overwrites row `round(Np/2)` of Xi with the exact centroid
     Pcc before computing delta. After the delta shift this guarantees
     one specific row of the final cloud equals x0 exactly -- a built-in
     elitism mechanism. Preserved here.
  5. In the main loop, for `ref != 0`, MATLAB hardcodes the global/local
     split as `pop1 = round(0.4*n)` (local/refined) and `pop2 = n - pop1`
     (global) -- this 0.4 is a LITERAL CONSTANT in the source, NOT tied to
     the user-supplied `ref` value (which only rescales LB/UB for the
     local cloud, via LB*ref / UB*ref). This is preserved exactly, flagged
     here so it isn't mistaken for a transcription bug.
  6. Per-agent candidate selection in MATLAB rebuilds `S = [S_global;
     S_ref]` via fresh `randperm` draws inside the `for i=1:n` loop and
     then reads row `S(i,:)`. Since S_global/S_ref are i.i.d. uniform
     random *subsets* of X_global/X_local re-drawn every i, reading row i
     of the freshly shuffled stack is distributionally identical to:
     "if i <= pop2: draw one uniformly random row from X_global, else
     draw one uniformly random row from X_local." We implement this
     equivalent (and far cheaper) form rather than literally re-running
     randperm(Np, pop) just to discard all but one row -- the per-agent
     candidate distribution is unchanged.
"""

from __future__ import annotations
import os
import numpy as np
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# LA_figure(d, Np, Rc, S, M=0)  ->  just load the precomputed figure
# ---------------------------------------------------------------------------
def load_LF(mat_path: str = "LFND.mat") -> np.ndarray:
    """
    Mirrors:  if M==0: load('LFND'); map = LFND;
    Returns the (Np, 2) raw point cloud, as float64, UNMODIFIED (no
    centering/normalization -- la_points() performs all of that, exactly
    as the MATLAB pipeline does).
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Could not find '{mat_path}'. This implementation requires the "
            f"author's official precomputed Lichtenberg Figure (M=0 mode); "
            f"no synthetic fallback is used."
        )
    data = loadmat(mat_path)
    if "LFND" not in data:
        raise KeyError(
            f"'LFND' variable not found in {mat_path}. Keys present: "
            f"{[k for k in data.keys() if not k.startswith('__')]}"
        )
    return data["LFND"].astype(np.float64)


# ---------------------------------------------------------------------------
# 2D rotation matrix, R = [cos -sin; sin cos]
# ---------------------------------------------------------------------------
def _rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# LA_points(K, LB, UB, x0, scale_factor, d)
# ---------------------------------------------------------------------------
def la_points(K: np.ndarray, lb: np.ndarray, ub: np.ndarray,
              x0: np.ndarray, scale_factor: float, d: int,
              rng: np.random.Generator) -> np.ndarray:
    """
    Direct translation of LA_points for the d > 3 branch (our TSP case is
    d=52), with the d <= 3 branch included for completeness/fidelity.

    K   : (Np, 2) raw Lichtenberg Figure point cloud (2D case)
    lb  : (d,) lower bounds for this call (already ref-scaled if local)
    ub  : (d,) upper bounds for this call
    x0  : (d,) trigger point (current best / search-space center)
    """
    Np = K.shape[0]

    if d <= 3:
        # MATLAB: teta = rand; R = [cos -sin; sin cos]; K = (R*K')';
        theta = rng.random()
        R = _rot2d(theta)
        K_rot = (R @ K.T).T          # (Np, 2)
        Xi = np.zeros((Np, d))
        for j in range(d):
            Xi[:, j] = K_rot[:, j] if j < K_rot.shape[1] else K_rot[:, 0]
    else:
        # MATLAB: for i = 1:2:d+2
        #             gama = rand; R = [cos -sin; sin cos];
        #             Xi(:,i:i+1) = (R*K')';
        #         end
        #         Xi = Xi(:,1:d);
        #
        # MATLAB matrices auto-grow on indexed assignment. The loop visits
        # 1-indexed i = 1, 3, 5, ... up to and including d+2 (if d+2 is
        # odd) or up to d+1 (the largest odd value <= d+2, if d+2 is
        # even). The last pair written is columns i:i+1, so the actual
        # required width is d+2 when d is even, but d+3 when d is odd
        # (since the last 1-indexed i lands on d+2 itself in that case,
        # needing column d+3). Get this wrong and you under-allocate by
        # one column for every odd-dimensional problem (e.g. eil51, d=51).
        width = d + 2 if d % 2 == 0 else d + 3
        Xi = np.zeros((Np, width))
        for i in range(0, width, 2):
            gama = rng.random()
            R = _rot2d(gama)
            Xi[:, i:i + 2] = (R @ K.T).T
        Xi = Xi[:, :d]

    # ---- per-dimension scaling (uses UNSCALED column range) -------------
    col_min = Xi.min(axis=0)
    col_max = Xi.max(axis=0)
    span = col_max - col_min
    span = np.where(span == 0, 1e-12, span)        # guard against deg. col
    scale = scale_factor * (ub - lb) / span         # (d,)
    Xi = Xi * scale                                  # broadcast per column

    # ---- centroid Pcc computed AFTER scaling, elitist row injection -----
    col_min2 = Xi.min(axis=0)
    col_max2 = Xi.max(axis=0)
    Pcc = (col_max2 - col_min2) / 2.0 + col_min2     # (d,)

    mid = int(round(Np / 2.0)) - 1                    # MATLAB round() is 1-indexed
    mid = int(np.clip(mid, 0, Np - 1))
    Xi[mid, :] = Pcc

    delta = Pcc - x0                                  # (d,)
    X = Xi - delta                                    # broadcast subtract per column
    return X


# ---------------------------------------------------------------------------
# bound_check(s, LB, UB)
# ---------------------------------------------------------------------------
def bound_check(s: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.clip(s, lb, ub)


# ---------------------------------------------------------------------------
# LA_optimization(fhandle, d, im, n, LB, UB, ref, n_iter, Np, Rc, S_c, M, fnonlin)
# ---------------------------------------------------------------------------
class LichtenbergAlgorithm:
    def __init__(self, dim: int, n_pop: int, n_iter: int, ref: float,
                 fitness_fn, lb=0.0, ub=1.0,
                 lf_path: str = "LFND.mat", seed: int | None = None):
        self.d = dim
        self.n = n_pop
        self.n_iter = n_iter
        self.ref = ref
        self.fitness_fn = fitness_fn
        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.rng = np.random.default_rng(seed)

        # M = 0 case: load the precomputed figure once, used every iteration
        self.LF = load_LF(lf_path)

    def optimize(self, verbose: bool = True, two_opt_hook=None, perturb_hook=None,
                 restart_patience: int = 12, restart_fraction: float = 0.3):
        """
        two_opt_hook:  callable(best_vector) -> (improved_vector, improved_fitness).
                       Applied ONLY to the global-best agent at the end of
                       each iteration (memetic step).

        perturb_hook:  callable(best_vector, rng) -> (kicked_vector, kicked_fitness).
                       Applied to `best` ONLY on stagnation, as an escape
                       move (e.g. double-bridge + 2-opt). This targets the
                       actual lock-in point: once `best` is 2-opt-locally
                       optimal, no candidate drawn from a cloud centered on
                       `best` can beat it, so randomly restarting OTHER
                       agents (restart_fraction below) cannot, by itself,
                       move `best`. perturb_hook is what actually escapes
                       the basin; restart_fraction just keeps the rest of
                       the population from fully collapsing in the
                       meantime. NOT part of the original MATLAB source.

        restart_patience / restart_fraction: anti-stagnation mechanism,
                      NOT part of the original MATLAB source. The faithful
                      LA loop below always recenters BOTH the global and
                      local candidate clouds on the single current `best`
                      (x_start = best every iteration, exactly as in
                      LA_optimization.m). On a 52-D Random-Keys TSP
                      landscape this collapses population diversity within
                      ~10 iterations. Set restart_patience=0 to disable
                      both mechanisms and reproduce the literal MATLAB
                      stagnation behavior exactly.
        """
        d, n = self.d, self.n
        lb, ub = self.lb, self.ub

        # ---- initial random population & fitness -------------------
        Individuals = self.rng.uniform(lb, ub, size=(n, d))
        Fitness = np.array([self.fitness_fn(Individuals[i]) for i in range(n)])
        I = np.argmin(Fitness)
        fmin = Fitness[I]
        best_ever = Individuals[I].copy()      # incumbent -- never lost, only improved
        anchor = best_ever.copy()              # MATLAB's `best` / x_start -- allowed to wander

        history = [fmin]
        stagnant_iters = 0

        for t in range(self.n_iter):
            x_start = anchor.copy()                      # MATLAB: x_start = best;
            scale_factor = 1.2 * self.rng.random()       # MATLAB: scale_factor = 1.2*rand;

            X_global = la_points(self.LF, lb, ub, x_start, scale_factor, d, self.rng)

            X_local = None
            if self.ref != 0:
                X_local = la_points(self.LF, lb * self.ref, ub * self.ref,
                                     x_start, scale_factor, d, self.rng)
                # NOTE: this 0.4 is a literal constant in the author's MATLAB
                # source (LA_optimization.m: "pop1 = round((0.4)*n);"),
                # independent of the actual `ref` value -- preserved as-is.
                pop1 = int(round(0.4 * n))   # local/refined agent count
                pop2 = n - pop1               # global agent count

            Np_global = X_global.shape[0]
            Np_local = X_local.shape[0] if X_local is not None else 0

            for i in range(n):
                if self.ref != 0:
                    if i < pop2:
                        cand = X_global[self.rng.integers(0, Np_global)]
                    else:
                        cand = X_local[self.rng.integers(0, Np_local)]
                else:
                    cand = X_global[self.rng.integers(0, Np_global)]

                cand = bound_check(cand, lb, ub)
                Fnew = self.fitness_fn(cand)

                if Fnew <= Fitness[i]:
                    Individuals[i] = cand
                    Fitness[i] = Fnew
                if Fnew <= fmin:
                    best_ever = cand.copy()
                    anchor = cand.copy()
                    fmin = Fnew

            # ---- memetic step: 2-opt polish on the GLOBAL BEST only ----
            if two_opt_hook is not None:
                polished, polished_fit = two_opt_hook(best_ever)
                if polished_fit < fmin:
                    best_ever = polished
                    anchor = polished.copy()
                    fmin = polished_fit
                    # resync into the population so it can keep competing
                    worst_idx = np.argmax(Fitness)
                    if fmin < Fitness[worst_idx]:
                        Individuals[worst_idx] = best_ever
                        Fitness[worst_idx] = fmin

            # ---- stagnation tracking & diversity injection -------------
            if fmin < history[-1] - 1e-9:
                stagnant_iters = 0
            else:
                stagnant_iters += 1

            if restart_patience > 0 and stagnant_iters >= restart_patience:
                # 1) escape move: double-bridge kick + 2-opt, applied to
                #    the INCUMBENT, producing a new ANCHOR for the next
                #    iteration's candidate clouds. This is the move that
                #    actually matters -- `anchor` is what every candidate
                #    cloud gets recentered on, so moving it is what lets
                #    the search leave the current 2-opt basin. The
                #    incumbent (`best_ever`/`fmin`) is only overwritten if
                #    the kicked+polished tour is actually better.
                if perturb_hook is not None:
                    kicked, kicked_fit = perturb_hook(best_ever, self.rng)
                    anchor = kicked.copy()                  # anchor always wanders
                    if kicked_fit < fmin:
                        best_ever = kicked.copy()
                        fmin = kicked_fit
                    anchor_pop_idx = np.argmax(Fitness)
                    Individuals[anchor_pop_idx] = kicked
                    Fitness[anchor_pop_idx] = kicked_fit

                # 2) keep the rest of the population from fully collapsing
                n_restart = max(1, int(round(restart_fraction * n)))
                worst_indices = np.argsort(Fitness)[::-1][:n_restart]
                best_idx_protected = np.argmin(Fitness)
                for idx in worst_indices:
                    if idx == best_idx_protected:
                        continue
                    Individuals[idx] = self.rng.uniform(lb, ub, size=d)
                    Fitness[idx] = self.fitness_fn(Individuals[idx])
                stagnant_iters = 0
                if verbose:
                    print(f"[LA] iter {t:4d}  stagnation -> double-bridge kick + "
                          f"restarted {n_restart} agents")

            history.append(fmin)
            if verbose and (t % 10 == 0 or t == self.n_iter - 1):
                print(f"[LA] iter {t:4d}  fmin = {fmin:.4f}")

        return best_ever, fmin, history
