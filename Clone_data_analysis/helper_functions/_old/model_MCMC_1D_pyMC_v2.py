import numpy as np
import pymc as pm
import state_simulations_1D_v1 as ss1D
from dataclasses import dataclass
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import pytensor.tensor as pt
import os
import pickle
import arviz as az
import matplotlib.pyplot as plt


def _run_single_simulation(params_tuple, timepoints, n_sims):
    q0_1, q1_0 = params_tuple
    params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
    
    results_init_0 = ss1D.run_simulations(
        params,
        {'state_0': 1.0, 'state_1': 0.0},  # all in state 0
        n_sims,
        timepoints
    )
    
    results_init_1 = ss1D.run_simulations(
        params,
        {'state_0': 0.0, 'state_1': 1.0},  # all in state 1
        n_sims,
        timepoints
    )
    
    # Ensure results are numpy arrays with shape (n_timepoints, n_sims)
    results_init_0 = np.array([np.array(x) for x in results_init_0])
    results_init_1 = np.array([np.array(x) for x in results_init_1])
    
    return (params_tuple, (results_init_0, results_init_1))


class CellInferencePyMC:
    def __init__(self, data_df, n_sims=1000, threshold_bounds=(1e-3, 1e-1),
                 verbose=True, n_cores=8, cache_file='simulation_grid_cache.pkl'):
        self.data_df = data_df
        self.n_sims = n_sims
        self.threshold_bounds = threshold_bounds
        self.verbose = verbose
        self.n_cores = min(n_cores, mp.cpu_count() - 1)
        self.cache_file = cache_file

        if self.verbose:
            print("Initializing CellInferencePyMC...")
            print(f"Number of simulations per parameter set: {n_sims}")
            print(f"Threshold bounds: {threshold_bounds}")
            print(f"Using {self.n_cores} CPU cores")
            print(f"Cache file: {cache_file}")
        
        self._setup_simulation_grid()

    def _setup_simulation_grid(self, n_grid=30, cache_file='simulation_grid_cache.pkl'):
        # Grid parameters
        grid_min = -7  # 10^-7
        grid_max = -2  # 10^-2
        
        # Include grid range in cache parameters
        cache_params = {
            'n_grid': n_grid,
            'n_sims': self.n_sims,
            'timepoints': list(self.data_df.index),
            'grid_min': grid_min,
            'grid_max': grid_max
        }
        
        if self.verbose:
            print(f"\nPre-computing simulation grid (n_grid={n_grid})...")
            print(f"Grid range: {grid_min} to {grid_max} (log10)")
            print(f"Total parameter combinations: {n_grid**2}")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if cached_data['params'] == cache_params:
                    if self.verbose:
                        print("Loading pre-computed simulation grid from cache...")
                    self.param_grid = cached_data['param_grid']
                    self.sim_results = cached_data['sim_results']
                    return
                else:
                    if self.verbose:
                        print("Cache parameters don't match, recomputing grid...")
                        print("Cached params:", cached_data['params'])
                        print("Current params:", cache_params)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache: {e}, recomputing grid...")
        else:
            if self.verbose:
                print("No cache file found, computing grid...")
        
        # Create logarithmically spaced grid
        q0_1_grid = np.logspace(grid_min, grid_max, n_grid).astype(float)
        q1_0_grid = np.logspace(grid_min, grid_max, n_grid).astype(float)

        self.param_grid = {
            'q0_1': q0_1_grid,
            'q1_0': q1_0_grid
        }

        # Create all pairs (q0_1, q1_0)
        param_combinations = [
            (float(q0_1), float(q1_0)) for q0_1 in q0_1_grid for q1_0 in q1_0_grid
        ]

        run_sim = partial(_run_single_simulation,
                          timepoints=list(self.data_df.index),
                          n_sims=self.n_sims)

        with mp.Pool(self.n_cores) as pool:
            if self.verbose:
                results = list(tqdm(pool.imap(run_sim, param_combinations),
                                    total=len(param_combinations)))
            else:
                results = pool.map(run_sim, param_combinations)

        # Store results in a dict keyed by (q0_1, q1_0)
        self.sim_results = dict(results)

        # Save to cache
        cache_data = {
            'params': cache_params,
            'param_grid': self.param_grid,
            'sim_results': self.sim_results
        }
        
        try:
            if self.verbose:
                print(f"Attempting to save cache to: {os.path.abspath(cache_file)}")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            if self.verbose:
                print("Successfully saved simulation grid to cache.")
                print(f"Cache file size: {os.path.getsize(cache_file) / 1024 / 1024:.1f} MB")
        except Exception as e:
            if self.verbose:
                print(f"Error saving cache: {e}")
                print(f"Full error details: {repr(e)}")

        if self.verbose:
            print("Pre-computation complete!")

    def build_model(self):
        if self.verbose:
            print("\nBuilding PyMC model...")

        # Calculate empirical proportions
        total_initial = self.data_df.loc[0, 'f_hi'] + self.data_df.loc[0, 'f_lo']
        emp_init_hi = self.data_df.loc[0, 'f_hi'] / total_initial
        emp_init_0 = 1 - emp_init_hi  # since init_0 is proportion in low state
        
        if self.verbose:
            print(f"Empirical initial proportions: hi={emp_init_hi:.3f}, lo={emp_init_0:.3f}")

        with pm.Model() as model:
            # Use log-uniform priors for transition rates
            log_q0_1 = pm.Uniform('log_q0_1', -7, -2)
            log_q1_0 = pm.Uniform('log_q1_0', -7, -2)
            
            # Transform to actual rates
            q0_1 = pm.Deterministic('q0_1', 10**log_q0_1)
            q1_0 = pm.Deterministic('q1_0', 10**log_q1_0)
            
            # Prior centered around empirical low proportion
            init_0 = pm.Beta('init_0', alpha=10*emp_init_0, beta=10*(1-emp_init_0))
            
            # Prior for threshold favoring lower values
            threshold = pm.Beta('threshold', alpha=2, beta=10)

            def potential(q0_1, q1_0, init_0, threshold):
                print(f"\nTesting parameters:")
                print(f"  q0_1={q0_1.eval():.2e}")
                print(f"  q1_0={q1_0.eval():.2e}")
                print(f"  init_0={init_0.eval():.3f}")
                print(f"  threshold={threshold.eval():.3f}")
                
                q0_1_grid_t = pt.as_tensor_variable(self.param_grid['q0_1'])
                q1_0_grid_t = pt.as_tensor_variable(self.param_grid['q1_0'])

                q0_1_idx = pt.argmin(pt.abs(q0_1_grid_t - q0_1))
                q1_0_idx = pt.argmin(pt.abs(q1_0_grid_t - q1_0))

                ll = 0
                for t_idx, t in enumerate(self.data_df.index):
                    q0_1_val = self.param_grid['q0_1'][q0_1_idx.eval()]
                    q1_0_val = self.param_grid['q1_0'][q1_0_idx.eval()]

                    results_init_0, results_init_1 = self.sim_results[(q0_1_val, q1_0_val)]
                    
                    states_0 = pt.as_tensor_variable(results_init_0[int(t_idx)])
                    states_1 = pt.as_tensor_variable(results_init_1[int(t_idx)])

                    p_hi_init_0 = pt.mean(states_0 > threshold)
                    p_hi_init_1 = pt.mean(states_1 > threshold)

                    p_hi = init_0 * p_hi_init_0 + (1 - init_0) * p_hi_init_1
                    p_lo = 1 - p_hi

                    counts = pt.as_tensor_variable([
                        self.data_df.loc[t, 'f_hi'],
                        self.data_df.loc[t, 'f_lo']
                    ])

                    #print(f"\nTime {t}:")
                    #print(f"  Model: p_hi={p_hi.eval():.3f}")
                    #data_prop = counts[0].eval() / (counts[0].eval() + counts[1].eval())
                    #print(f"  Data:  p_hi={data_prop:.3f}")

                    probs = pt.stack([p_hi, p_lo])
                    ll += pt.sum(counts * pt.log(probs + 1e-12))

                print(f"Log likelihood: {ll.eval():.3f}")
                return ll

            pm.Potential('likelihood', potential(q0_1, q1_0, init_0, threshold))

        if self.verbose:
            print("Model building complete!")
        return model

    def run_inference(self, n_draws, tune, n_chains=4, random_seed=42):
        """
        Run MCMC inference.
        
        Args:
            n_draws: Number of samples to draw
            tune: Number of tuning steps
            n_chains: Number of chains to run
            random_seed: Random seed for reproducibility
        """
        if self.verbose:
            print(f"\nRunning MCMC inference...")
        model = self.build_model()
        with model:
            trace = pm.sample(
                draws=n_draws,
                tune=tune,
                chains=n_chains,
                cores=min(n_chains, self.n_cores),
                random_seed=random_seed,
                progressbar=self.verbose,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True}  # Store log likelihood
            )
        return trace

    def analyze_results(self, trace):
        """
        Analyze MCMC results and return best-fitting parameters.
        
        Args:
            trace: PyMC trace object
        
        Returns:
            tuple: (best_params_dict, trace)
        """
        # Get posterior mean as our best estimate
        posterior = trace.posterior
        
        # Average over chains and draws
        best_params = {
            'init_0': float(posterior.init_0.mean()),
            'q0_1': float(posterior.q0_1.mean()),
            'q1_0': float(posterior.q1_0.mean()),
            'threshold': float(posterior.threshold.mean())
        }
        
        if self.verbose:
            print("\nBest-fit parameters (posterior mean):")
            for param, value in best_params.items():
                print(f"{param}: {value:.6f}")
        
        return best_params, trace

    def plot_fit_comparison(self, params_dict, n_sims=10000, save_path=None):
        """
        Plot comparison of model fit to data.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        trans_params = ss1D.TransitionParams(
            q0_1=params_dict['q0_1'],
            q1_0=params_dict['q1_0']
        )
        init_dict = {
            'state_0': params_dict['init_0'],
            'state_1': 1 - params_dict['init_0']
        }
        results = ss1D.run_simulations(
            trans_params,
            init_dict,
            n_sims,
            list(self.data_df.index)
        )

        print("Model predictions:")
        
        sim_props = []
        data_props = []
        data_sems = []
        times = []

        for t_idx, t in enumerate(self.data_df.index):
            sims = np.asarray(results[t], dtype=float).flatten()
            sim_prop = np.mean(sims > params_dict['threshold'])
            sim_props.append(sim_prop)

            n_hi = self.data_df.loc[t, 'f_hi']
            n_lo = self.data_df.loc[t, 'f_lo']
            total = n_hi + n_lo
            data_prop = n_hi / total
            sem = np.sqrt((data_prop * (1 - data_prop)) / total)
            
            data_props.append(data_prop)
            data_sems.append(sem)
            times.append(t)
            
            print(f"Time {t}: Model={sim_prop:.3f}, Data={data_prop:.3f} ± {sem:.3f}")

        plt.figure(figsize=(6, 4))
        plt.plot(times, sim_props, '-o', label='Model')
        plt.errorbar(times, data_props, yerr=data_sems, fmt='s', 
                    capsize=5, label='Data')
        plt.xlabel('Time')
        plt.ylabel('Proportion High')
        plt.legend()
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
        plt.close()

    def save_results(self, trace, params_dict, f1, cond, cell_type, base_dir="./figures/MCMC"):
        """
        Save MCMC results to a specified directory.
        
        Args:
            trace: PyMC trace object
            params_dict: Dictionary of best-fit parameters
            f1: Program identifier
            cond: Condition identifier
            cell_type: Cell type identifier
            base_dir: Base directory for output (default: "./figures/MCMC")
        """
        import os
        import arviz as az
        
        # Create output directory
        output_dir = os.path.join(base_dir, f"Program_{f1}_{cell_type}_{cond}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parameters to text file
        summary = az.summary(trace)
        with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
            f.write("Best-fit parameters:\n")
            for param, value in params_dict.items():
                f.write(f"{param}: {value:.6f}\n")
            f.write("\nParameter statistics:\n")
            f.write(summary.to_string())
        
        # Generate and save plot
        plot_path = os.path.join(output_dir, 'fit_comparison.pdf')
        self.plot_fit_comparison(params_dict, save_path=plot_path)
        
        if self.verbose:
            print(f"Results saved to {output_dir}")

    def get_best_fit(self, trace, n_samples=100):
        """
        Get the best-fitting parameters from the trace.
        
        Args:
            trace: PyMC trace object
            n_samples: Number of samples to evaluate
        
        Returns:
            Dictionary of best-fitting parameters
        """
        # Sample parameters from the trace
        samples = []
        posterior = trace.posterior  # Get the posterior samples
        n_total = len(posterior.chain) * len(posterior.draw)
        
        for i in range(n_samples):
            # Random chain and draw
            chain_idx = np.random.randint(len(posterior.chain))
            draw_idx = np.random.randint(len(posterior.draw))
            
            params = {
                'init_0': float(posterior.init_0[chain_idx, draw_idx]),
                'q0_1': float(posterior.q0_1[chain_idx, draw_idx]),
                'q1_0': float(posterior.q1_0[chain_idx, draw_idx]),
                'threshold': float(posterior.threshold[chain_idx, draw_idx])
            }
            samples.append((params, self._evaluate_likelihood(params)))
        
        # Find best parameters
        best_params, best_ll = max(samples, key=lambda x: x[1])
        if self.verbose:
            print(f"Best log likelihood: {best_ll:.2f}")
        
        return best_params

    def _evaluate_likelihood(self, params):
        """Helper function to compute likelihood for a set of parameters."""
        # Find closest grid points
        q0_1_grid = self.param_grid['q0_1']
        q1_0_grid = self.param_grid['q1_0']
        
        q0_1_idx = np.argmin(np.abs(q0_1_grid - params['q0_1']))
        q1_0_idx = np.argmin(np.abs(q1_0_grid - params['q1_0']))
        
        # Get pre-computed results
        q0_1_val = q0_1_grid[q0_1_idx]
        q1_0_val = q1_0_grid[q1_0_idx]
        results_init_0, results_init_1 = self.sim_results[(q0_1_val, q1_0_val)]
        
        ll = 0
        for t_idx, t in enumerate(self.data_df.index):
            # Get states for this timepoint
            states_0 = results_init_0[t_idx]
            states_1 = results_init_1[t_idx]
            
            # Combine based on initial conditions
            p_hi = params['init_0'] * np.mean(states_0 > params['threshold']) + \
                   (1 - params['init_0']) * np.mean(states_1 > params['threshold'])
            p_lo = 1 - p_hi
            
            # Get data counts
            n_hi = self.data_df.loc[t, 'f_hi']
            n_lo = self.data_df.loc[t, 'f_lo']
            
            ll += n_hi * np.log(p_hi + 1e-12) + n_lo * np.log(p_lo + 1e-12)
        
        return ll
