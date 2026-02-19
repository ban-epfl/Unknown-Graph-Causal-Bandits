# Causal Bandit with Unknwon Structure


This repository provides the implementation of the AISTATS 2026 paper "Graph Learning is Suboptimal in Causal Bandits," featuring code to run and compare various **causal bandit algorithms** within a fixed environment instance.  
The code initializes a causal bandit environment with given structural parameters and runs selected algorithms, such as **Known_k_UCB+**, **Unknown_k_UCB+**, **Standard_UCB**, and **RAPS**, to evaluate their cumulative regret performance over multiple runs.

Each experiment logs and saves the **average cumulative regret** and its **standard deviation** across multiple independent iterations, enabling direct comparison of algorithms under identical conditions.

---

## 🧩 File Overview

- **`main.py`** — Command-line runner that initializes the environment, executes the learners, and saves results.  
- **`Environment.py`** — Defines the causal environment (`Env`), including sampling and mean-reward functions.  
- **`Known_k_UCB_plus_learner.py`** — Implementation of the empirical version of Known-\(k\) UCB+ learner.  
- **`Unknown_k_UCB_plus.py`** — Implementation of the empirical version of Unknown-\(k\) UCB+ learner.  
- **`Standard_UCB_learner.py`** — Implementation of the standard UCB learner.  
- **`RAPS_learner.py`** — Implementation of the RAPS algorithm.  
- **`results/`** — Directory for saved experiment outputs (`.npy` files).

---

## 💻 Command-Line Usage

Run the main file with configurable parameters:

```bash
python main.py \
  --n 8 \
  --k 1 \
  --l 3 \
  --m 3 \
  --T 2000 \
  --iter 10 \
  --parent-effect 0.7 \
  --algos ucbp ucbpp ucb RAPS \
  --save_path results

| Argument          | Description                                                 | Default   |
| ----------------- | ----------------------------------------------------------- | --------- |
| `--n`             | Number of non-reward variables                              | 8         |
| `--k`             | Number of parents of the reward node                        | 1         |
| `--l`             | Cardinality (number of possible values) of each variable    | 3         |
| `--m`             | Number of variables the learner can intervene on            | 3         |
| `--T`             | Horizon (number of rounds per run)                          | 2000      |
| `--iter`          | Number of independent runs on the same environment instance | 10        |
| `--parent-effect` | Strength of parent effect in the environment                | 0.7       |
| `--algos`         | Algorithms to run (`ucbp`, `ucbpp`, `ucb`, `RAPS`)          | all       |
| `--save_path`     | Directory to store results                                  | `results` |
```

Note that RAPS algorithm is only implemented for k=1.

📦 Output

Each selected algorithm produces one result file in the results/ directory:

```bash
avg_std_regret_<ALGO>_n<n>_k<k>_v<v>_m<m>_T<T>_it<iterations>.npy
```

Each file stores a 2×T NumPy array:

- Row 0: Mean cumulative regret across runs

- Row 1: Standard deviation of cumulative regret

If a file already exists, the script will load it instead of re-running the experiment.

📊 Example

Example: compare Known_k_UCB+ and Standard_UCB on a small instance.

```bash
python main.py --n 8 --k 2 --l 3 --m 3 --T 1000 --iter 20 --algos ucbp ucb
```
After the run, you’ll find files like:
- results/avg_std_regret_Known_k_UCB+_n8_k2_v3_m3_T1000_it20.npy
- results/avg_std_regret_Standard_UCB_n8_k2_v3_m3_T1000_it20.npy

📚 Citation

To cite this work, please use the following BibTeX entry:

```
@inproceedings{
shahverdikondori2026graph,
title={Graph Learning is Suboptimal in Causal Bandits},
author={Mohammad Shahverdikondori and Jalal Etesami and Negar Kiyavash},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=CF8vvfItK3}
}
```
