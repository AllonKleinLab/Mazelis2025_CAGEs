import state_simulations_1D_v1 as ss
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
from typing import List, Tuple, Dict, Any  # Make sure typing import is complete
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns





def run_and_process_1paramset_simulations(params, n_sims=1000, divisions=[0,2,4,6]):
    """
    Run simulations from both initial conditions and process results into probability distributions
    
    Parameters
    ----------
    params : TransitionParams
        Parameters for state transitions
    n_sims : int, optional
        Number of simulations to run
    divisions : list, optional
        Division numbers to simulate
        
    Returns
    -------
    tuple of dicts
        Two dictionaries (init0_dist, init1_dist) containing probability distributions
        for each division number. Each distribution has 2^k+1 bins for k divisions.
    """
    # Define initial conditions
    init0 = {'state_0': 1.0, 'state_1': 0.0}
    init1 = {'state_0': 0.0, 'state_1': 1.0}
    
    # Run simulations from both initial conditions
    results0 = ss.run_simulations(params, init0, n_sims, divisions)
    results1 = ss.run_simulations(params, init1, n_sims, divisions)
    
    # Process results into probability distributions
    init0_dist = {}
    init1_dist = {}
    
    for div in divisions:
        n_cells = 2**div
        possible_values = np.linspace(0, 1, n_cells + 1)
        
        # Create histograms with bins centered on possible values
        hist0, _ = np.histogram(results0[div], bins=n_cells+1, range=(0, 1), density=True)
        hist1, _ = np.histogram(results1[div], bins=n_cells+1, range=(0, 1), density=True)
        
        # Normalize to create probability distributions
        hist0 = hist0 / hist0.sum()
        hist1 = hist1 / hist1.sum()
        
        # Store distributions with their corresponding fraction values
        init0_dist[div] = dict(zip(possible_values, hist0))
        init1_dist[div] = dict(zip(possible_values, hist1))
    
    return init0_dist, init1_dist


def process_param_set(param_tuple, n_sims, divisions):
    """
    This is a helper function, bridging the gap between 
    run_parameter_grid_simulations
    and the run_and_process_1paramset_simulations function to
    allow for parallel processing of simulations.
    
    Parameters
    ----------
    param_tuple : tuple
        Tuple containing (q0_1, q1_0) transition probabilities
    n_sims : int
        Number of simulations per parameter set
    divisions : list
        List of division numbers to simulate
        
    Returns
    -------
    tuple
        Contains (q0_1, q1_0, init0_dist, init1_dist)
    """
    q0_1, q1_0 = param_tuple
    params = ss.TransitionParams(q0_1=q0_1, q1_0=q1_0)
    return (q0_1, q1_0, *run_and_process_1paramset_simulations(
        params, n_sims=n_sims, divisions=divisions
    ))

def run_parameter_grid_simulations(
    q_min: float = 1e-4,
    q_max: float = 1.0, 
    n_grid: int = 40,
    n_sims: int = 1000,
    divisions: List[int] = [0, 1, 2, 3, 4],
    n_processes: int = None
) -> Tuple[dict, dict]:
    """
    Run simulations over a grid of transition parameters and organize results into dataframes
    
    Parameters
    ----------
    q_min : float, optional
        Minimum transition probability, default 1e-4
    q_max : float, optional
        Maximum transition probability, default 1.0
    n_grid : int, optional
        Number of grid points in each dimension, default 40
    n_sims : int, optional
        Number of simulations per parameter set
    divisions : list, optional
        Division numbers to simulate
    n_processes : int, optional
        Number of processes to use for parallel computation. If None, uses cpu_count()
        
    Returns
    -------
    tuple of dicts
        Two dictionaries (init0_dfs, init1_dfs) containing pandas DataFrames
        for each division number. DataFrame columns are q0_1, q1_0, and 
        probability distributions.
    """
    
    # Create log-spaced parameter grid
    q_vals = np.logspace(np.log10(q_min), np.log10(q_max), n_grid)
    
    # Initialize dictionaries to store results
    init0_dfs = {}
    init1_dfs = {}
    
    # Initialize dataframes for each division number
    for div in divisions:
        n_states = 2**div + 1
        fractions = [i/(2**div) for i in range(n_states)]
        columns = ['q0_1', 'q1_0'] + fractions
        init0_dfs[div] = pd.DataFrame(columns=columns)
        init1_dfs[div] = pd.DataFrame(columns=columns)
    
    # Create parameter combinations
    param_combinations = [(q0_1, q1_0) for q0_1 in q_vals for q1_0 in q_vals]
    
    # Create partial function with fixed n_sims and divisions
    process_param_set_partial = partial(process_param_set, n_sims=n_sims, divisions=divisions)
    
    # Run parallel simulations
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_param_set_partial, param_combinations),
            total=len(param_combinations),
            desc="Running parallel simulations"
        ))
    
    # Process results
    for q0_1, q1_0, init0_dist, init1_dist in results:
        for div in divisions:
            # Convert distributions to row format
            row0 = [q0_1, q1_0] + list(init0_dist[div].values())
            row1 = [q0_1, q1_0] + list(init1_dist[div].values())
            
            # Append to respective dataframes
            init0_dfs[div].loc[len(init0_dfs[div])] = row0
            init1_dfs[div].loc[len(init1_dfs[div])] = row1
    
    return init0_dfs, init1_dfs


def get_parameter_grid_simulations(divisions, n_sims, q_min, q_max, n_grid, cache_file='1D_model_param_grid_cache.pkl'):
    """
    Get parameter grid simulation results, using cached results if available.
    
    Parameters
    ----------
    divisions : list of int
        Division numbers to simulate
    n_sims : int
        Number of simulations per parameter set
    q_min : float
        Minimum transition probability
    q_max : float
        Maximum transition probability  
    n_grid : int
        Number of grid points for each parameter
    cache_file : str, optional
        Path to cache file, by default 'param_grid_cache.pkl'
        
    Returns
    -------
    tuple of dicts
        Two dictionaries (init0_dfs, init1_dfs) containing pandas DataFrames
        for each division number
    """
    # Check if cache exists and matches parameters
    cache_valid = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                
            # Check if cached parameters match
            cache_valid = (
                cache['divisions'] == divisions and
                cache['n_sims'] == n_sims and
                cache['q_min'] == q_min and
                cache['q_max'] == q_max and
                cache['n_grid'] == n_grid
            )
            
            if cache_valid:
                print("Using cached results.")
                return cache['init0_dfs'], cache['init1_dfs']
                
        except (EOFError, KeyError):
            # Handle corrupted cache file
            cache_valid = False
    
    # Run simulations if cache invalid or missing
    init0_dfs, init1_dfs = run_parameter_grid_simulations(
        divisions=divisions,
        n_sims=n_sims,
        q_min=q_min,
        q_max=q_max,
        n_grid=n_grid
    )
    
    # Save results to cache
    cache = {
        'divisions': divisions,
        'n_sims': n_sims,
        'q_min': q_min,
        'q_max': q_max,
        'n_grid': n_grid,
        'init0_dfs': init0_dfs,
        'init1_dfs': init1_dfs
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
        
    return init0_dfs, init1_dfs


def mean_and_std_over_params(df, show_heatmaps=False):
    """
    Calculate mean and standard deviation from the probability distributions in the dataframe
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        - q0_1, q1_0 (transition parameters)
        - Additional columns with headers giving the fraction of cells in hi state
    show_heatmaps : bool, optional
        Whether to display heatmaps of the distributions, by default False
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: mean, std, q0_1, q1_0
    """
    # Convert column names to float for calculation
    x_vals = df.columns[2:].astype(float)
    
    # Calculate means
    means = df.iloc[:, 2:].dot(x_vals)/df.iloc[:, 2:].sum(axis=1)
    m2 = df.iloc[:, 2:].dot(x_vals**2)/df.iloc[:, 2:].sum(axis=1)

    stds = ((m2 - means**2)**0.5).astype(float)
    
    
    # Create output DataFrame
    df_out = pd.DataFrame({
        'q0_1': df['q0_1'],
        'q1_0': df['q1_0'],
        'mean': means,
        'std': stds,

    })
    
    if show_heatmaps:
        # Get unique parameter values
        q0_1_vals = sorted(df['q0_1'].unique())
        q1_0_vals = sorted(df['q1_0'].unique())
        
        # Create 2D arrays for heatmaps
        mean_grid = np.zeros((len(q0_1_vals), len(q1_0_vals)))
        std_grid = np.zeros((len(q0_1_vals), len(q1_0_vals)))
        
        # Fill the grids
        for i, q0_1 in enumerate(q0_1_vals):
            for j, q1_0 in enumerate(q1_0_vals):
                idx = (df['q0_1'] == q0_1) & (df['q1_0'] == q1_0)
                mean_grid[i, j] = means[idx].iloc[0]
                std_grid[i, j] = stds[idx].iloc[0]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot mean heatmap
        sns.heatmap(mean_grid, ax=ax1, 
                   xticklabels=np.round(q1_0_vals, 4),
                   yticklabels=np.round(q0_1_vals, 4),
                   cmap='viridis')
        ax1.set_title('Mean')
        ax1.set_xlabel('q1_0')
        ax1.set_ylabel('q0_1')
        
        # Plot std heatmap
        sns.heatmap(std_grid, ax=ax2,
                   xticklabels=np.round(q1_0_vals, 4),
                   yticklabels=np.round(q0_1_vals, 4),
                   cmap='viridis')
        ax2.set_title('Standard Deviation')
        ax2.set_xlabel('q1_0')
        ax2.set_ylabel('q0_1')
        # Invert y-axis for both heatmaps
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    return df_out

