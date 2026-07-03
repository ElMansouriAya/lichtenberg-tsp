"""
tsp_problem.py
--------------
Encapsulates TSPLIB instances (berlin52 plus four additional benchmark
instances for the scaling study: eil51, st70, pr76, rat99), the Random-Keys
decoder used to bridge LA's continuous output to a discrete permutation, and
a 2-opt local search operator used as the memetic refinement step.

All coordinates below were taken verbatim from official TSPLIB mirrors
(NODE_COORD_SECTION), not approximated or regenerated, and cross-checked
against the standard optimum table (Reinelt, TSPLIB95) for each instance:
    eil51    : 426
    berlin52 : 7542
    st70     : 675
    pr76     : 108159
    rat99    : 1211
"""

import numpy as np

# ---------------------------------------------------------------------------
# berlin52 (TSPLIB). 52 locations in Berlin (Groetschel). Optimum = 7542.
# ---------------------------------------------------------------------------
BERLIN52_COORDS = np.array([
    [565, 575], [25, 185], [345, 750], [945, 685], [845, 655],
    [880, 660], [25, 230], [525, 1000], [580, 1175], [650, 1130],
    [1605, 620], [1220, 580], [1465, 200], [1530, 5], [845, 680],
    [725, 370], [145, 665], [415, 635], [510, 875], [560, 365],
    [300, 465], [520, 585], [480, 415], [835, 625], [975, 580],
    [1215, 245], [1320, 315], [1250, 400], [660, 180], [410, 250],
    [420, 555], [575, 665], [1150, 1160], [700, 580], [685, 595],
    [685, 610], [770, 610], [795, 645], [720, 635], [760, 650],
    [475, 960], [95, 260], [875, 920], [700, 500], [555, 815],
    [830, 485], [1170, 65], [830, 610], [605, 625], [595, 360],
    [1340, 725], [1740, 245]
], dtype=float)

# ---------------------------------------------------------------------------
# eil51 (TSPLIB). 51-city problem (Christofides/Eilon). Optimum = 426.
# ---------------------------------------------------------------------------
EIL51_COORDS = np.array([
    [37, 52], [49, 49], [52, 64], [20, 26], [40, 30], [21, 47], [17, 63],
    [31, 62], [52, 33], [51, 21], [42, 41], [31, 32], [5, 25], [12, 42],
    [36, 16], [52, 41], [27, 23], [17, 33], [13, 13], [57, 58], [62, 42],
    [42, 57], [16, 57], [8, 52], [7, 38], [27, 68], [30, 48], [43, 67],
    [58, 48], [58, 27], [37, 69], [38, 46], [46, 10], [61, 33], [62, 63],
    [63, 69], [32, 22], [45, 35], [59, 15], [5, 6], [10, 17], [21, 10],
    [5, 64], [30, 15], [39, 10], [32, 39], [25, 32], [25, 55], [48, 28],
    [56, 37], [30, 40]
], dtype=float)

# ---------------------------------------------------------------------------
# st70 (TSPLIB). 70-city problem (Smith/Thompson). Optimum = 675.
# ---------------------------------------------------------------------------
ST70_COORDS = np.array([
    [64, 96], [80, 39], [69, 23], [72, 42], [48, 67], [58, 43], [81, 34],
    [79, 17], [30, 23], [42, 67], [7, 76], [29, 51], [78, 92], [64, 8],
    [95, 57], [57, 91], [40, 35], [68, 40], [92, 34], [62, 1], [28, 43],
    [76, 73], [67, 88], [93, 54], [6, 8], [87, 18], [30, 9], [77, 13],
    [78, 94], [55, 3], [82, 88], [73, 28], [20, 55], [27, 43], [95, 86],
    [67, 99], [48, 83], [75, 81], [8, 19], [20, 18], [54, 38], [63, 36],
    [44, 33], [52, 18], [12, 13], [25, 5], [58, 85], [5, 67], [90, 9],
    [41, 76], [25, 76], [37, 64], [56, 63], [10, 55], [98, 7], [16, 74],
    [89, 60], [48, 82], [81, 76], [29, 60], [17, 22], [5, 45], [79, 70],
    [9, 100], [17, 82], [74, 67], [10, 68], [48, 19], [83, 86], [84, 94]
], dtype=float)

# ---------------------------------------------------------------------------
# pr76 (TSPLIB). 76-city problem (Padberg/Rinaldi). Optimum = 108159.
# ---------------------------------------------------------------------------
PR76_COORDS = np.array([
    [3600, 2300], [3100, 3300], [4700, 5750], [5400, 5750], [5608, 7103],
    [4493, 7102], [3600, 6950], [3100, 7250], [4700, 8450], [5400, 8450],
    [5610, 10053], [4492, 10052], [3600, 10800], [3100, 10950], [4700, 11650],
    [5400, 11650], [6650, 10800], [7300, 10950], [7300, 7250], [6650, 6950],
    [7300, 3300], [6650, 2300], [5400, 1600], [8350, 2300], [7850, 3300],
    [9450, 5750], [10150, 5750], [10358, 7103], [9243, 7102], [8350, 6950],
    [7850, 7250], [9450, 8450], [10150, 8450], [10360, 10053], [9242, 10052],
    [8350, 10800], [7850, 10950], [9450, 11650], [10150, 11650], [11400, 10800],
    [12050, 10950], [12050, 7250], [11400, 6950], [12050, 3300], [11400, 2300],
    [10150, 1600], [13100, 2300], [12600, 3300], [14200, 5750], [14900, 5750],
    [15108, 7103], [13993, 7102], [13100, 6950], [12600, 7250], [14200, 8450],
    [14900, 8450], [15110, 10053], [13992, 10052], [13100, 10800], [12600, 10950],
    [14200, 11650], [14900, 11650], [16150, 10800], [16800, 10950], [16800, 7250],
    [16150, 6950], [16800, 3300], [16150, 2300], [14900, 1600], [19800, 800],
    [19800, 10000], [19800, 11900], [19800, 12200], [200, 12200], [200, 1100],
    [200, 800]
], dtype=float)

# ---------------------------------------------------------------------------
# rat99 (TSPLIB). Rattled grid (Pulleyblank). Optimum = 1211.
# ---------------------------------------------------------------------------
RAT99_COORDS = np.array([
    [6, 4], [15, 15], [24, 18], [33, 12], [48, 12], [57, 14], [67, 10],
    [77, 10], [86, 15], [6, 21], [17, 26], [23, 25], [32, 35], [43, 23],
    [55, 35], [65, 36], [78, 39], [87, 35], [3, 53], [12, 44], [28, 53],
    [33, 49], [47, 46], [55, 52], [64, 50], [71, 57], [87, 57], [4, 72],
    [15, 78], [22, 70], [34, 71], [42, 79], [54, 77], [66, 79], [78, 67],
    [87, 73], [7, 81], [17, 95], [26, 98], [32, 97], [43, 88], [57, 89],
    [64, 85], [78, 83], [83, 98], [5, 109], [13, 111], [25, 102], [38, 119],
    [46, 107], [58, 110], [67, 110], [74, 113], [88, 110], [2, 124],
    [17, 134], [23, 129], [36, 131], [42, 137], [53, 123], [63, 135],
    [72, 134], [87, 129], [2, 146], [16, 147], [25, 153], [38, 155],
    [42, 158], [57, 154], [66, 151], [73, 151], [86, 149], [5, 177],
    [13, 162], [25, 169], [35, 177], [46, 172], [54, 166], [65, 174],
    [73, 161], [86, 162], [2, 195], [14, 196], [28, 189], [38, 187],
    [46, 195], [57, 194], [63, 188], [77, 193], [85, 194], [8, 211],
    [12, 217], [22, 210], [34, 216], [47, 203], [58, 213], [66, 206],
    [78, 210], [85, 204]
], dtype=float)

# ---------------------------------------------------------------------------
# Instance registry: name -> (coords, known optimum)
# ---------------------------------------------------------------------------
TSPLIB_INSTANCES = {
    "eil51":    (EIL51_COORDS,    426),
    "berlin52": (BERLIN52_COORDS, 7542),
    "st70":     (ST70_COORDS,     675),
    "pr76":     (PR76_COORDS,     108159),
    "rat99":    (RAT99_COORDS,    1211),
}


def get_instance(name: str):
    """Returns (TSPProblem, known_optimum) for a registered TSPLIB instance name."""
    if name not in TSPLIB_INSTANCES:
        raise KeyError(f"Unknown instance '{name}'. Available: {list(TSPLIB_INSTANCES)}")
    coords, optimum = TSPLIB_INSTANCES[name]
    return TSPProblem(coords), optimum


class TSPProblem:
    """Distance/decoding utilities for a TSPLIB-style instance."""

    def __init__(self, coords: np.ndarray = BERLIN52_COORDS):
        self.coords = coords
        self.n_cities = coords.shape[0]
        self.dist_matrix = self._build_distance_matrix(coords)

    @staticmethod
    def _build_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """
        TSPLIB's EUC_2D edge weight type specifies distances as the
        Euclidean distance ROUNDED TO THE NEAREST INTEGER (the `nint`
        convention in Reinelt's TSPLIB95 spec), not raw floating-point
        Euclidean distance. This matters for comparing against published
        optima (e.g. pr76 = 108159 is only reproducible under this exact
        rounding rule; unrounded distances would not match it).
        """
        diff = coords[:, None, :] - coords[None, :, :]
        raw = np.sqrt((diff ** 2).sum(axis=-1))
        return np.round(raw)

    # ------------------------------------------------------------------
    # Random Keys encoding: LA produces a real-valued vector in [0, 1]^d.
    # argsort of that vector gives a valid permutation of city indices.
    # ------------------------------------------------------------------
    @staticmethod
    def decode_random_keys(keys: np.ndarray) -> np.ndarray:
        """Map a continuous key vector to a city permutation via argsort."""
        return np.argsort(keys)

    def tour_length(self, tour: np.ndarray) -> float:
        """Closed-tour length (returns to the starting city)."""
        nxt = np.roll(tour, -1)
        return self.dist_matrix[tour, nxt].sum()

    def fitness_from_keys(self, keys: np.ndarray) -> float:
        """Convenience: decode + evaluate in one call."""
        return self.tour_length(self.decode_random_keys(keys))

    # ------------------------------------------------------------------
    # Double-bridge perturbation (4-opt kick), standard ILS escape move
    # ------------------------------------------------------------------
    def double_bridge(self, tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Classic double-bridge move: cut the tour into 4 segments A-B-C-D
        and reconnect as A-C-B-D. This is a 4-opt move that 2-opt cannot
        undo in a single step, which is exactly why it's the standard
        perturbation used to escape 2-opt local optima in Iterated Local
        Search for TSP.
        """
        n = len(tour)
        if n < 8:
            return tour.copy()
        pts = sorted(rng.choice(range(1, n), size=3, replace=False))
        p1, p2, p3 = pts
        A, B, C, D = tour[:p1], tour[p1:p2], tour[p2:p3], tour[p3:]
        return np.concatenate([A, C, B, D])

    def encode_tour_to_keys(self, tour: np.ndarray) -> np.ndarray:
        """
        Inverse of decode_random_keys: given a permutation (e.g. after
        2-opt polish), produce a consistent key vector in [0,1] such that
        argsort(keys) reproduces that exact permutation. Used to feed
        2-opt-improved tours back into LA's continuous population.
        """
        keys = np.empty(self.n_cities)
        ranks = np.linspace(0.0, 1.0, self.n_cities)
        keys[tour] = ranks
        return keys

    # ------------------------------------------------------------------
    # 2-opt local search (memetic refinement step)
    # ------------------------------------------------------------------
    def two_opt(self, tour: np.ndarray, max_passes: int = 1,
                first_improvement: bool = False) -> tuple[np.ndarray, float]:
        """
        Classic 2-opt: repeatedly reverse a segment [i+1, j] if doing so
        shortens the tour. Returns (improved_tour, improved_length).

        first_improvement=True applies the first improving move found per
        sweep (cheap, good for "polish every agent every iteration").
        first_improvement=False does best-improvement per sweep (stronger,
        recommended only for the iteration-best agent to control cost).
        """
        n = self.n_cities
        best_tour = tour.copy()
        best_len = self.tour_length(best_tour)

        for _ in range(max_passes):
            improved = False
            best_delta = 0.0
            best_move = None

            for i in range(n - 1):
                a, b = best_tour[i], best_tour[i + 1]
                for j in range(i + 2, n):
                    c = best_tour[j]
                    d = best_tour[(j + 1) % n]
                    if a == d or b == c:
                        continue
                    delta = (
                        self.dist_matrix[a, c] + self.dist_matrix[b, d]
                        - self.dist_matrix[a, b] - self.dist_matrix[c, d]
                    )
                    if delta < best_delta - 1e-10:
                        if first_improvement:
                            best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                            best_len += delta
                            improved = True
                            break
                        best_delta = delta
                        best_move = (i, j)
                if first_improvement and improved:
                    break

            if not first_improvement:
                if best_move is None:
                    break
                i, j = best_move
                best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                best_len += best_delta
                improved = True

            if not improved:
                break

        return best_tour, best_len
