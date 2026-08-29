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
