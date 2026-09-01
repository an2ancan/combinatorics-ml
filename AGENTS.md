# AGENTS.md

## Repository Overview

This repository explores Machine Learning and Deep Learning approaches for classical Combinatorial Optimization problems, starting with the **Travelling Salesman Problem (TSP)**.

---

## Workspace Rules & Guidelines

### 1. No Emojis Rule (Strict)
- **Do NOT use emojis anywhere in this repository.**
- This applies to:
  - Notebook markdown cells, code cells, and outputs
  - Python scripts, comments, and docstrings
  - Print / logging statements: use text tags like `[OK]`, `[WARNING]`, `[INFO]`, `[ERROR]` instead of checkmarks or icons
  - Markdown files (`README.md`, `AGENTS.md`, etc.)
  - Git commit messages

### 2. Code & Notebook Conventions
- **Google Colab Ready**: All `.ipynb` notebooks must run standalone on Google Colab with clear section numbers, pip installations, and Google Drive mounting.
- **Distance Computations**: Use `scipy.spatial.distance.cdist(coords, coords, metric='euclidean')` for pairwise Euclidean distance matrices.
- **Metric Evaluation**: Always compute final tour lengths using unrounded float32/float64 Euclidean distances (not scaled integer approximations).
- **Progress Tracking**: Use `tqdm.auto` for progress loops in notebooks.
- **Clean JSON formatting**: Notebook JSON files must follow `nbformat: 4`, `nbformat_minor: 0` with clean formatting.

---

## Notebook Descriptions

| Notebook | Purpose | Key Details |
|---|---|---|
| `TSP_Data_Generation.ipynb` | Generates uniform 2D TSP datasets | Generates 128,000 train & 10,000 val instances per size (10, 20, ..., 200). Coordinates sampled i.i.d. from Uniform(0, 1)^2. Saved as `.npy` to Google Drive. |
| `TSP_Baseline_ORTools.ipynb` | Google OR-Tools baseline solver | Solves validation instances using OR-Tools Routing Model with `PATH_CHEAPEST_ARC` and `GUIDED_LOCAL_SEARCH`. Records tour lengths, solve times, and solver statuses. |
| `TSP_Baseline_Concorde.ipynb` | Concorde exact solver baseline | Uses `pyconcorde` (branch-and-cut) to find provably optimal solutions. Primary ground truth for small-to-medium instances (TSP-10 to TSP-100). |
| `TSP_Baseline_LKH3.ipynb` | LKH-3 heuristic baseline | Compiles and executes LKH-3 (Lin-Kernighan heuristic) via TSPLIB format. Near-optimal baseline for medium-to-large problem sizes up to TSP-200. |
| `TSP_Model_Seq2Seq.ipynb` | Seq2Seq Pointer Network solver | Trains a neural Pointer Network on uniform 2D TSP instances using REINFORCE policy gradients with an EMA baseline. Evaluates optimality gap against classical baselines. |

---

## Directory & Data Layout

```
TSP_Data/
├── tsp_10/
│   ├── train.npy                     # (128000, 10, 2) float32
│   └── val.npy                       # (10000, 10, 2) float32
├── tsp_20/ ...
└── baseline_results/
    ├── config.json
    ├── ortools_summary.csv
    ├── ortools_per_instance.csv
    ├── ortools_lengths_tsp_{n}.npy
    ├── concorde_summary.csv
    ├── concorde_lengths_tsp_{n}.npy
    ├── lkh3_summary.csv
    └── lkh3_lengths_tsp_{n}.npy
```
