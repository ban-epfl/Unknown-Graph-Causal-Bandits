# main.py
import numpy as np
from typing import Callable, Dict, Any
from tqdm import tqdm
import os
import argparse

# Import implementations
from Environment import Env
from Known_k_UCB_plus_learner import known_k_UCB_plus
from Standard_UCB_learner import standard_UCB
from Unknown_k_UCB_plus import unknown_k_UCB_plus
from RAPS_learner import RAPS



def run_single_iteration(
    env: Env,
    algo_ctor: Callable[..., Any],
    algo_params: Dict[str, Any],
    T: int,
) -> np.ndarray:
    """
    Run one iteration (single run) of an algorithm on a fixed environment instance.

    Returns
    -------
    regrets : np.ndarray of shape (T,)
        Per-round instantaneous regret (best_mean - played_action_mean).
    """
    # Fresh algorithm instance for each run
    learner = algo_ctor(**algo_params)

    # Precompute best action and best mean for this algorithm's m
    m_for_algo = algo_params["m"]
    _, best_mean = env.best_action(m_for_algo)

    regrets = np.zeros(T, dtype=float)

    for t in range(T):
        action = learner.action()

        # Get a sample from the interventional distribution
        sample = env.sample(action)

        # Update learner
        learner.update(action, sample)

        # Compute regret
        played_mean = env.mean_reward(action)
        regrets[t] = best_mean - played_mean


    return regrets


def run_experiment_average_regret(
    env: Env,
    algo_ctor: Callable[..., Any],
    algo_params: Dict[str, Any],
    T: int,
    iterations: int,
) -> np.ndarray:
    """
    Run `iterations` independent runs on the *same environment instance*,
    return the per-round average regret over runs.
    """
    all_regrets = np.zeros((iterations, T), dtype=float)
    for r in tqdm(range(iterations), desc="Running iterations"):
        regrets = run_single_iteration(env, algo_ctor, algo_params, T)
        all_regrets[r] = regrets
    cum_regret = all_regrets.cumsum(axis=1)
    regret_std = cum_regret.std(axis=0)
    return cum_regret.mean(axis=0), regret_std


def main(input_algos, n, k, v, m, T, iterations, parent_effect, erdos_renyi_p, save_path):

    path = f"envs/env_n{n}_k{k}_v{v}_per{erdos_renyi_p}_pe{parent_effect}.json"
    env = Env(n=n, k=k, v=v, parent_effect=parent_effect, p_edge = erdos_renyi_p)
    # # save the environment
    # env.save(path)

    algos = {
        "ucbp": {
            "init": known_k_UCB_plus,
            "params": {
                "n": n,
                "k": k,
                "m": m,
                "v": v,
                "T": T
            },
            "label": "Known_k_UCB+",
            },
        "ucbpp":{
            "init": unknown_k_UCB_plus,
            "params": {
                "n": n,
                "m": m,
                "v": v,
                "T": T
            },
            "label": "Unknown_k_UCB+",
        },
        "ucb": {
            "init": standard_UCB,
            "params": {
                "n": n,
                "m": m,
                "v": v,
                "T": T
            },
            "label": "Standard_UCB",
            },
        "RAPS": {
            "init": RAPS,
            "params": {
                "n": n,
                "m": m,
                "v": v,
                "T": T,
                "epsilon": 0.05
            },
            "label": "RAPS",
            },
    }

    algos = {key: val for key, val in algos.items() if key in input_algos}


    # Run experiments
    for algo in algos.keys():
        algo_name = algos[algo]["label"]
        algo_init = algos[algo]["init"]
        algo_params = algos[algo]["params"]

        print(f"Running experiment for {algo_name}...")

        save_path = save_path + f"/avg_std_regret_{algo_name}_n{n}_k{k}_v{v}_m{m}_T{T}_it{iterations}.npy"

        # Run the experiment if the result file does not exist
        if not os.path.exists(save_path):
            avg_regret, regret_std = run_experiment_average_regret(env, algo_init, algo_params, T, iterations)
            # Save results
            res = np.vstack((avg_regret, regret_std))
            np.save(f"results/avg_std_regret_{algo_name}_n{n}_k{k}_v{v}_m{m}_T{T}_it{iterations}.npy", res)

        else: 
            print(f"Loading existing results from {save_path}")
            data = np.load(save_path)
            avg_regret = data[0]
        
        
        print(f"[{algo_name}] final avg cumulative regret: {avg_regret[-1]:.3f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal MAB experiments with unknown graph for various algorithms.")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of non-reward variables (default: 8)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of parents of the reward node (default: 1)")
    parser.add_argument("--l", type=int, default=3,
                        help="Cardinality of variables (1...l), reward is binary (default: 3)")
    parser.add_argument("--parent-effect", type=float, default=0.7, dest="parent_effect",
                        help="Parent-effect parameter used by Env to compute the conditional distributions (default: 0.7)")

    # Algorithm / horizon
    parser.add_argument("--m", type=int, default=3,
                        help="Intervention size m (default: 3)")
    parser.add_argument("--T", type=int, default=2000,
                        help="Horizon (rounds) (default: 2000)")
    parser.add_argument("--iter", type=int, default=10,
                        help="Number of runs on the same env instance (default: 10)")
    parser.add_argument("--algos", nargs="+", default=["ucbp", "ucbpp", "ucb", "RAPS"],
                        help="List of algorithms to run (default: all)")

    # I/O
    parser.add_argument("--save_path", type=str, default="results",
                        help="Directory to save outputs (default: results)")

    args = parser.parse_args()

    main(args.algos, args.n, args.k, args.l, args.m, args.T, args.iter, args.parent_effect, erdos_renyi_p=2/args.n, save_path=args.save_path)

