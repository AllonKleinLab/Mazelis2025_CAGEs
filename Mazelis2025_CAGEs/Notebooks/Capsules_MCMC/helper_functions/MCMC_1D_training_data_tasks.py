"""
Minimal public version of 1D training-data helpers used by script2_MCMC.py.

Contains only:
- get_parameter_grid_simulations
- mean_and_std_over_params

and their direct dependencies.
"""

import os
import pickle
from functools import partial
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp

from . import state_simulations_1D_v1 as ss


def run_and_process_1paramset_simulations(
    params: ss.TransitionParams,
    n_sims: int = 1000,
    divisions: List[int] = (0, 2, 4, 6),
):
    """
    Run simulations from both initial conditions and process into P(f_hi|params,div).
    """
    init0 = {"state_0": 1.0, "state_1": 0.0}
    init1 = {"state_0": 0.0, "state_1": 1.0}

    results0 = ss.run_simulations(params, init0, n_sims, list(divisions))
    results1 = ss.run_simulations(params, init1, n_sims, list(divisions))

    init0_dist: Dict[int, Dict[float, float]] = {}
    init1_dist: Dict[int, Dict[float, float]] = {}

    for div in divisions:
        n_cells = 2**div
        possible_values = np.linspace(0, 1, n_cells + 1)

        hist0, _ = np.histogram(
            results0[div], bins=n_cells + 1, range=(0, 1), density=True
        )
        hist1, _ = np.histogram(
            results1[div], bins=n_cells + 1, range=(0, 1), density=True
        )

        hist0 = hist0 / hist0.sum()
        hist1 = hist1 / hist1.sum()

        init0_dist[div] = dict(zip(possible_values, hist0))
        init1_dist[div] = dict(zip(possible_values, hist1))

    return init0_dist, init1_dist


def _process_param_set(
    param_tuple: Tuple[float, float], n_sims: int, divisions: List[int]
):
    """Helper for parallel execution over the parameter grid."""
    q0_1, q1_0 = param_tuple
    params = ss.TransitionParams(q0_1=q0_1, q1_0=q1_0)
    return (q0_1, q1_0, *run_and_process_1paramset_simulations(params, n_sims, divisions))


def _run_parameter_grid_simulations(
    q_min: float = 1e-4,
    q_max: float = 1.0,
    n_grid: int = 40,
    n_sims: int = 1000,
    divisions: List[int] = (0, 2, 4, 6),
    n_processes: int | None = None,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Core grid-simulation routine; see get_parameter_grid_simulations for public API.
    """
    q_vals = np.logspace(np.log10(q_min), np.log10(q_max), n_grid)

    init0_dfs: Dict[int, pd.DataFrame] = {}
    init1_dfs: Dict[int, pd.DataFrame] = {}

    for div in divisions:
        n_states = 2**div + 1
        fractions = [i / (2**div) for i in range(n_states)]
        columns = ["q0_1", "q1_0"] + fractions
        init0_dfs[div] = pd.DataFrame(columns=columns)
        init1_dfs[div] = pd.DataFrame(columns=columns)

    param_combinations = [(q0_1, q1_0) for q0_1 in q_vals for q1_0 in q_vals]
    process_param_set_partial = partial(
        _process_param_set, n_sims=n_sims, divisions=list(divisions)
    )

    if n_processes is None:
        n_processes = mp.cpu_count()

    with mp.Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_param_set_partial, param_combinations),
                total=len(param_combinations),
                desc="Running parallel simulations",
            )
        )

    for q0_1, q1_0, init0_dist, init1_dist in results:
        for div in divisions:
            row0 = [q0_1, q1_0] + list(init0_dist[div].values())
            row1 = [q0_1, q1_0] + list(init1_dist[div].values())
            init0_dfs[div].loc[len(init0_dfs[div])] = row0
            init1_dfs[div].loc[len(init1_dfs[div])] = row1

    return init0_dfs, init1_dfs


def get_parameter_grid_simulations(
    divisions: List[int],
    n_sims: int,
    q_min: float,
    q_max: float,
    n_grid: int,
    cache_file: str = "1D_model_param_grid_cache.pkl",
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Public API used in script2_MCMC.py.

    Returns cached simulation results if cache matches parameters, otherwise
    runs the grid and saves to cache.
    """
    cache_valid = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            cache_valid = (
                cache["divisions"] == divisions
                and cache["n_sims"] == n_sims
                and cache["q_min"] == q_min
                and cache["q_max"] == q_max
                and cache["n_grid"] == n_grid
            )
            if cache_valid:
                print("Using cached results.")
                return cache["init0_dfs"], cache["init1_dfs"]
        except (EOFError, KeyError):
            cache_valid = False

    init0_dfs, init1_dfs = _run_parameter_grid_simulations(
        q_min=q_min,
        q_max=q_max,
        n_grid=n_grid,
        n_sims=n_sims,
        divisions=divisions,
    )

    cache = {
        "divisions": divisions,
        "n_sims": n_sims,
        "q_min": q_min,
        "q_max": q_max,
        "n_grid": n_grid,
        "init0_dfs": init0_dfs,
        "init1_dfs": init1_dfs,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    return init0_dfs, init1_dfs


def mean_and_std_over_params(df: pd.DataFrame, show_heatmaps: bool = False) -> pd.DataFrame:
    """
    Calculate mean and std of f_hi under P(f_hi | params) for a given division.

    Matches the interface expected by script2_MCMC.py.
    """
    x_vals = df.columns[2:].astype(float)
    means = df.iloc[:, 2:].dot(x_vals) / df.iloc[:, 2:].sum(axis=1)
    m2 = df.iloc[:, 2:].dot(x_vals**2) / df.iloc[:, 2:].sum(axis=1)
    stds = ((m2 - means**2) ** 0.5).astype(float)

    df_out = pd.DataFrame(
        {"q0_1": df["q0_1"], "q1_0": df["q1_0"], "mean": means, "std": stds}
    )

    if show_heatmaps:
        q0_1_vals = sorted(df["q0_1"].unique())
        q1_0_vals = sorted(df["q1_0"].unique())
        mean_grid = np.zeros((len(q0_1_vals), len(q1_0_vals)))
        std_grid = np.zeros((len(q0_1_vals), len(q1_0_vals)))

        for i, q0_1 in enumerate(q0_1_vals):
            for j, q1_0 in enumerate(q1_0_vals):
                idx = (df["q0_1"] == q0_1) & (df["q1_0"] == q1_0)
                mean_grid[i, j] = means[idx].iloc[0]
                std_grid[i, j] = stds[idx].iloc[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(
            mean_grid,
            ax=ax1,
            xticklabels=np.round(q1_0_vals, 4),
            yticklabels=np.round(q0_1_vals, 4),
            cmap="viridis",
        )
        ax1.set_title("Mean")
        ax1.set_xlabel("q1_0")
        ax1.set_ylabel("q0_1")

        sns.heatmap(
            std_grid,
            ax=ax2,
            xticklabels=np.round(q1_0_vals, 4),
            yticklabels=np.round(q0_1_vals, 4),
            cmap="viridis",
        )
        ax2.set_title("Standard Deviation")
        ax2.set_xlabel("q1_0")
        ax2.set_ylabel("q0_1")

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        plt.tight_layout()
        plt.show()

    return df_out

