# Hybrid Lichtenberg Algorithm for the Traveling Salesman Problem

> A faithful reproduction and hybrid extension of the Lichtenberg Algorithm (LA) for combinatorial optimization, adapted to the Traveling Salesman Problem using Random Keys encoding and Iterated Local Search.

---

## Overview

This project reproduces the **Lichtenberg Algorithm (LA)** proposed by Pereira et al. and adapts it to solve the Traveling Salesman Problem (TSP).

Instead of directly applying the original continuous optimization algorithm, this implementation introduces a hybrid architecture combining:

- Random Keys representation
- 2-opt local search
- Iterated Local Search (ILS)
- TSPLIB benchmark evaluation
- Reproducible experimental protocol
- Ablation studies and critical analysis

The goal is not only to reproduce the original algorithm but also to evaluate its actual contribution when applied to discrete optimization problems.

---

## Key Features

- Faithful implementation of the original Lichtenberg Algorithm
- Random Keys encoding for continuous-to-discrete mapping
- Hybrid LA + ILS optimization pipeline
- Benchmarking on TSPLIB instances
- Statistical evaluation across multiple runs
- Ablation study
- Computational complexity analysis
- Full technical report

---

## Repository Structure

```text
.
├── original_matlab/
├── data/
├── docs/
│   ├── Hybrid_Lichtenberg_Algorithm_for_TSP_Report.pdf
│   └── figures/
├── src/
│   ├── main.py
│   ├── benchmark.py
│   ├── tsp_problem.py
│   └── la_optimization.py
├── README.md
└── requirements.txt
```

---

## Methodology

The optimization pipeline follows:

```
Lichtenberg Algorithm
          │
          ▼
Random Keys Encoding
          │
          ▼
Candidate Tour
          │
          ▼
2-opt Local Search
          │
          ▼
Iterated Local Search
          │
          ▼
Best Solution
```

---

## Experimental Evaluation

Experiments were conducted using multiple TSPLIB benchmark instances.

Evaluation includes:

- Best cost
- Mean cost
- Standard deviation
- Runtime
- Optimality gap
- Ablation analysis

---

## Results

The study shows that:

- Random Keys successfully bridges continuous and discrete optimization.
- Local search contributes the majority of the optimization improvements.
- The Lichtenberg exploration mechanism provides limited but measurable diversification under specific conditions.
- Population diversity remains an important factor for escaping local minima.

A complete discussion is available in the report.

---

## Technical Report

The full research report is available here:

** [Hybrid Lichtenberg Algorithm for TSP Report](docs/Hybrid_Lichtenberg_Algorithm_for_TSP_Report.pdf)**

The report includes:

- Background
- Mathematical formulation
- Algorithm implementation
- Experimental protocol
- Results
- Critical analysis
- Future work

---

## Technologies

- Python
- NumPy
- TSPLIB95
- Matplotlib

---
