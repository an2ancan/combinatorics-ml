# Combinatorics ML

Experimenting with machine learning approaches to classical combinatorial optimisation problems.

## Stage 1 — TSP Data Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/an2ancan/combinatorics-ml/blob/main/TSP_Data_Generation.ipynb)

The first stage focuses on preparing training and validation data for the
**Travelling Salesman Problem (TSP)**.

The notebook [`TSP_Data_Generation.ipynb`](TSP_Data_Generation.ipynb) generates
datasets of random TSP instances by sampling 2-D city coordinates from a
uniform distribution over the unit square.

### Parameters

| Parameter | Value |
|---|---|
| Problem sizes | 10, 20, 30, …, 200 |
| Training instances per size | 128 000 |
| Validation instances per size | 10 000 |
| Point distribution | U(0, 1)^2 |
| Storage format | NumPy `.npy` |

Each instance is a tensor of shape `(num_nodes, 2)` containing coordinates
sampled i.i.d. from Uniform(0, 1).

### Output structure

```
TSP_Data/
├── tsp_10/
│   ├── train.npy   (128000, 10, 2)
│   └── val.npy     (10000, 10, 2)
├── tsp_20/
│   ├── train.npy   (128000, 20, 2)
│   └── val.npy     (10000, 20, 2)
├── ...
└── tsp_200/
    ├── train.npy   (128000, 200, 2)
    └── val.npy     (10000, 200, 2)
```

### Quick start

```python
import numpy as np

train = np.load("TSP_Data/tsp_50/train.npy")  # (128000, 50, 2)
val   = np.load("TSP_Data/tsp_50/val.npy")    # (10000, 50, 2)
```

## Stage 2 — OR-Tools Baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/an2ancan/combinatorics-ml/blob/main/TSP_Baseline_ORTools.ipynb)

The notebook [`TSP_Baseline_ORTools.ipynb`](TSP_Baseline_ORTools.ipynb) evaluates
**Google OR-Tools** as a baseline solver on the validation datasets.

For each TSP size it solves a configurable sample of instances and reports:
- **Tour length** (mean, std, min, max)
- **Wall-clock time** per instance
- **Solver status** (solved / timeout / failed)
- Optimality report and visualisations

### Solver settings

| Parameter | Default |
|---|---|
| Instances per size | 128 |
| Parallel workers | 12 CPUs (configurable via `NUM_WORKERS`) |
| Time limit per instance | 10 s |
| First solution strategy | `PATH_CHEAPEST_ARC` |
| Metaheuristic | `GUIDED_LOCAL_SEARCH` |

### Output

Results are saved to `MyDrive/TSP_Data/baseline_results/`:
- `ortools_summary.csv` — summary stats per TSP size
- `ortools_per_instance.csv` — per-instance tour lengths & times
- `ortools_lengths_tsp_{n}.npy` — tour length arrays for gap computation
- `config.json` — solver configuration snapshot

## Stage 3 — Concorde Baseline (Exact Solver)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/an2ancan/combinatorics-ml/blob/main/TSP_Baseline_Concorde.ipynb)

The notebook [`TSP_Baseline_Concorde.ipynb`](TSP_Baseline_Concorde.ipynb) evaluates
**Concorde** — the gold-standard exact solver for TSP. It finds provably optimal
solutions via branch-and-cut, providing ground truth for optimality gap computation.

Best suited for TSP-10 through TSP-100; may be slow on larger instances.

| Parameter | Default |
|---|---|
| Instances per size | 128 |
| Time limit per instance | 300 s |

### Output

Results are saved to `MyDrive/TSP_Data/baseline_results/`:
- `concorde_summary.csv`, `concorde_per_instance.csv`
- `concorde_lengths_tsp_{n}.npy` — optimal tour lengths

## Stage 4 — LKH-3 Baseline (Lin-Kernighan Heuristic)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/an2ancan/combinatorics-ml/blob/main/TSP_Baseline_LKH3.ipynb)

The notebook [`TSP_Baseline_LKH3.ipynb`](TSP_Baseline_LKH3.ipynb) evaluates
**LKH-3** — the state-of-the-art heuristic solver. It produces near-optimal solutions
(typically within 0.5% of optimal) and scales to thousands of nodes. Primary baseline
for large TSP instances where Concorde is too slow.

| Parameter | Default |
|---|---|
| Instances per size | 128 |
| Time limit per instance | 60 s |
| LKH runs per instance | 1 |
| Max trials per run | 1000 |

### Output

Results are saved to `MyDrive/TSP_Data/baseline_results/`:
- `lkh3_summary.csv`, `lkh3_per_instance.csv`
- `lkh3_lengths_tsp_{n}.npy` — near-optimal tour lengths

---

## Local Setup & Running Notebooks

This project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency and environment management.

### Prerequisites

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup

1. Clone the repository and install all dependencies:
   ```bash
   uv sync
   ```

2. (Optional) Install optional solver extras like Concorde:
   ```bash
   uv sync --extra concorde
   ```

### Environment Configuration (.env)

Copy `.env.example` to `.env` to customize your local configuration:

```bash
cp .env.example .env
```

Key variables in `.env`:
* `RUN_LOCALLY=True` — Switches data/results paths from `/content/drive/...` to local directory `./TSP_Data`. (Defaults to `False` on Colab).
* `NUM_WORKERS=12` — Number of CPU workers for parallel baseline evaluations.
* `TSP_DATA_DIR="./TSP_Data"` — Local directory for generated datasets (excluded from git).
* `TSP_RESULTS_DIR="./TSP_Data/baseline_results"` — Local directory for baseline evaluations (excluded from git).
* `RCLONE_REMOTE_DATA="gdrive:TSP_Data"` — Remote target for data synchronization via `rclone`.

### Syncing Data with Google Drive / Cloud

1. Configure `rclone` locally for this repository (creates `.rclone.conf`, ignored by git):
   ```bash
   task rclone:config
   ```
   * Choose `n` (New remote), name it `gdrive`, choose `drive` (Google Drive), and complete browser authentication.

2. Verify connection:
   ```bash
   task rclone:ls
   ```

3. Pull or push datasets and baseline results:
   ```bash
   # Pull datasets from Google Drive to local ./TSP_Data
   task sync:data:pull

   # Push local datasets to Google Drive
   task sync:data:push

   # Pull baseline results from Google Drive
   task sync:results:pull

   # Push local baseline results to Google Drive
   task sync:results:push
   ```

### Running the Notebooks Locally

You can launch individual notebooks directly using [Task](https://taskfile.dev/):

```bash
task data-gen    # Open TSP_Data_Generation.ipynb in JupyterLab
task ortools     # Open TSP_Baseline_ORTools.ipynb in JupyterLab
task concorde    # Open TSP_Baseline_Concorde.ipynb in JupyterLab
task lkh3        # Open TSP_Baseline_LKH3.ipynb in JupyterLab
```

Or launch the entire workspace:

```bash
task lab         # Launch full JupyterLab workspace
```

Alternatively, run directly with `uv`:

```bash
uv run jupyter lab [notebook_name.ipynb]
```



