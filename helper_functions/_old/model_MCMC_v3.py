import emcee
import numpy as np
from state_simulations_v1 import TransitionParams, run_simulations
import multiprocessing
import time
from contextlib import contextmanager

class CellInference:
    def __init__(self, data_df, n_sims=1000, threshold_bounds=(1e-3, 1e-1), verbose=False, temp_scale=1.0):
        self.data_df = data_df
        self.n_sims = n_sims
        self.ndim = 12  # 8 transition rates + 3 initial probs + 1 threshold
        self.threshold_bounds = threshold_bounds  # Bounds for threshold parameter
        self.verbose = verbose
        self.temp_scale = temp_scale  # Scale factor for log probability (higher = flatter distribution)
        
    def log_prior(self, theta):
        # Unpack parameters
        trans_rates = theta[:8]
        init_probs = theta[8:11]
        threshold = theta[11]
        
        # Check transition rate bounds
        if not all(0 <= r <= 1 for r in trans_rates):
            return -np.inf
            
        # Check initial probability bounds and sum
        if not all(0 <= p <= 1 for p in init_probs):
            return -np.inf
        if np.sum(init_probs) > 1:
            return -np.inf
            
        # Check transition probability sums
        if (trans_rates[0] + trans_rates[1] > 1 or
            trans_rates[2] + trans_rates[3] > 1 or
            trans_rates[4] + trans_rates[5] > 1 or
            trans_rates[6] + trans_rates[7] > 1):
            return -np.inf
            
        # Check threshold bounds
        if not (self.threshold_bounds[0] <= threshold <= self.threshold_bounds[1]):
            return -np.inf
            
        return 0.0  # Uniform prior within constraints
        
    def log_likelihood(self, theta):
        try:
            # Unpack parameters
            trans_rates = theta[:8]
            init_probs = theta[8:11]
            threshold = theta[11]  # Get threshold from theta
            
            # Create parameter objects
            params = TransitionParams(
                q00_01=trans_rates[0], q00_10=trans_rates[1],
                q01_00=trans_rates[2], q01_11=trans_rates[3],
                q10_00=trans_rates[4], q10_11=trans_rates[5],
                q11_01=trans_rates[6], q11_10=trans_rates[7]
            )
            
            init_dict = {
                'state_00': init_probs[0],
                'state_01': init_probs[1],
                'state_10': init_probs[2],
                'state_11': 1 - np.sum(init_probs)
            }
            
            # Run simulations
            results = run_simulations(params, init_dict, self.n_sims, 
                                   list(self.data_df.index))
            
            # Calculate log likelihood
            ll = 0
            for t, sims in results.items():
                points = np.array(sims)
                probs = np.array([
                    np.mean((points[:,0] > threshold) & (points[:,1] > threshold)),  # hihi
                    np.mean((points[:,0] > threshold) & (points[:,1] <= threshold)), # hilo
                    np.mean((points[:,0] <= threshold) & (points[:,1] > threshold)), # lohi
                    np.mean((points[:,0] <= threshold) & (points[:,1] <= threshold)) # lolo
                ])
                
                # Get observed counts
                row = self.data_df.loc[t]
                counts = np.array([
                    row['f1f2_hi'],
                    row['f1_hi'],
                    row['f2_hi'],
                    row['f1f2_lo']
                ])
                
                # Multinomial log likelihood
                ll += np.sum(counts * np.log(probs + 1e-10))
                
            return ll
            
        except Exception as e:
            print(f"Error in likelihood calculation: {e}")
            return -np.inf
            
    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        # Scale the log probability by temperature factor
        return (lp + self.log_likelihood(theta)) / self.temp_scale
        
    def run_inference(self, n_walkers=32, n_steps=1000, n_cores=None, randomize_init=False):
        """
        Run MCMC inference with parallel processing.
        
        Parameters:
        -----------
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of steps for each walker
        n_cores : int, optional
            Number of CPU cores to use. If None, uses cpu_count() - 1
        randomize_init : bool, optional
            Whether to randomize initial positions
        """
        import multiprocessing
        import time
        from contextlib import contextmanager
        
        # Determine optimal number of cores if not specified
        if n_cores is None:
            n_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Check minimum number of walkers
        min_walkers = 2 * self.ndim
        if n_walkers < min_walkers:
            if self.verbose:
                print(f"Increasing number of walkers from {n_walkers} to {min_walkers} (minimum required)")
            n_walkers = min_walkers

        # Initialize walkers
        pos = np.random.rand(n_walkers, self.ndim)
        # Scale parameters appropriately
        pos[:, :8] *= 0.1  # Start with small transition rates
        pos[:, 8:11] *= 0.25  # Start with roughly equal initial probs
        pos[:, 11] = (self.threshold_bounds[1] - self.threshold_bounds[0]) * np.random.rand(n_walkers) + self.threshold_bounds[0]

        # Storage for monitoring convergence
        log_prob_history = []
        mean_log_prob_history = []
        std_log_prob_history = []
        
        if self.verbose:
            print(f"\nStarting MCMC sampling with:")
            print(f"- {n_walkers} walkers")
            print(f"- {n_steps} steps per walker")
            print(f"- {n_cores} CPU cores")
            print(f"- {n_walkers * n_steps} total samples")
        
        try:
            # Set up parallel processing
            with multiprocessing.Pool(processes=n_cores) as pool:
                start_time = time.time()
                
                # Initialize sampler with parallel processing
                sampler = emcee.EnsembleSampler(
                    n_walkers, 
                    self.ndim, 
                    self.log_probability,
                    pool=pool
                )
                
                # Run MCMC with progress monitoring
                for i, (pos, log_prob, _) in enumerate(sampler.sample(pos, iterations=n_steps, progress=True)):
                    # Store all log probabilities
                    log_prob_history.append(log_prob.copy())
                    
                    # Calculate and store statistics
                    mean_log_prob = np.mean(log_prob)
                    std_log_prob = np.std(log_prob)
                    mean_log_prob_history.append(mean_log_prob)
                    std_log_prob_history.append(std_log_prob)
                    
                    # Report progress
                    if self.verbose and i % 100 == 0:
                        elapsed_time = time.time() - start_time
                        steps_per_second = (i + 1) * n_walkers / elapsed_time
                        remaining_steps = n_steps - (i + 1)
                        estimated_remaining_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                        
                        print(f"\nStep {i+1}/{n_steps}:")
                        print(f"Mean log probability = {mean_log_prob:.2f} ± {std_log_prob:.2f}")
                        print(f"Sampling speed: {steps_per_second:.1f} steps/second")
                        print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
                
                end_time = time.time()
                
                if self.verbose:
                    total_time = end_time - start_time
                    print(f"\nSampling completed in {total_time/60:.1f} minutes")
                    print(f"Final mean log probability: {np.mean(sampler.get_log_prob()[-1]):.2f}")
                    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
                    
        except Exception as e:
            print(f"Error during sampling: {str(e)}")
            raise
        
        # Convert histories to numpy arrays for easier handling
        convergence_data = {
            'log_prob_history': np.array(log_prob_history),
            'mean_log_prob_history': np.array(mean_log_prob_history),
            'std_log_prob_history': np.array(std_log_prob_history),
            'steps': np.arange(n_steps),
            'runtime_seconds': end_time - start_time,
            'acceptance_fraction': np.mean(sampler.acceptance_fraction),
            'n_cores': n_cores
        }
        
        return sampler, convergence_data
        
    def analyze_results(self, sampler, convergence_data=None, burnin=None):
        # Automatically set burnin to 1/3 of total steps if not specified
        total_steps = sampler.get_chain().shape[0]
        if burnin is None:
            burnin = total_steps // 3
        
        # Ensure burnin isn't larger than our available steps
        if burnin >= total_steps:
            if self.verbose:
                print(f"Warning: burnin ({burnin}) is larger than total steps ({total_steps})")
                print(f"Setting burnin to {total_steps // 2} steps")
            burnin = total_steps // 2
        
        # Get samples and log probabilities
        samples = sampler.get_chain(discard=burnin, flat=True)
        log_probs = sampler.get_log_prob(discard=burnin, flat=True)
        
        # Filter out invalid values
        valid_mask = np.isfinite(log_probs)
        if self.verbose:
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                print(f"\nWarning: {n_invalid} samples with invalid log probabilities were filtered out")
            print(f"\nUsing {burnin} steps as burn-in period")
            print(f"Analyzing {len(samples)} samples after burn-in")
        
        # Calculate mean parameters
        mean_params = np.mean(samples, axis=0)
        
        params_dict = {
            "q00_01": mean_params[0],
            "q00_10": mean_params[1],
            "q01_00": mean_params[2],
            "q01_11": mean_params[3],
            "q10_00": mean_params[4],
            "q10_11": mean_params[5],
            "q11_01": mean_params[6],
            "q11_10": mean_params[7],
            "init_00": mean_params[8],
            "init_01": mean_params[9],
            "init_10": mean_params[10],
            "init_11": 1 - np.sum(mean_params[8:11]),
            "threshold": mean_params[11]  # Add threshold to params_dict
        }
        
        if convergence_data is not None and self.verbose:
            print("\nConvergence History Summary:")
            n_steps = len(convergence_data['mean_log_prob_history'])
            print(f"Initial mean log prob: {convergence_data['mean_log_prob_history'][0]:.2f}")
            print(f"Final mean log prob: {convergence_data['mean_log_prob_history'][-1]:.2f}")
            print(f"Log prob improvement: {convergence_data['mean_log_prob_history'][-1] - convergence_data['mean_log_prob_history'][0]:.2f}")
        
        if self.verbose:
            print("\nConvergence Statistics:")
            print(f"Mean log probability: {np.mean(log_probs):.2f}")
            print(f"Standard deviation of log probability: {np.std(log_probs):.2f}")
        
        return params_dict, samples, log_probs

    def plot_convergence_history(self, convergence_data, figsize=(10, 6), save_path=None, title=None):
        """
        Plot the convergence history of the MCMC chain.
        
        Parameters:
        -----------
        convergence_data : dict
            Dictionary containing convergence history data
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            If provided, save the plot to this path
        title : str, optional
            Custom title for the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        
        # Plot mean log probability
        plt.plot(convergence_data['steps'], 
                convergence_data['mean_log_prob_history'],
                label='Mean Log Probability')
        
        # Add uncertainty band
        plt.fill_between(
            convergence_data['steps'],
            convergence_data['mean_log_prob_history'] - convergence_data['std_log_prob_history'],
            convergence_data['mean_log_prob_history'] + convergence_data['std_log_prob_history'],
            alpha=0.2,
            label='±1 std dev'
        )
        
        plt.xlabel('Step')
        plt.ylabel('Log Probability')
        if title:
            plt.title(title)
        else:
            plt.title('MCMC Convergence History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved convergence plot to {save_path}")
        
        plt.show()
        
        # Print convergence statistics
        if self.verbose:
            print("\nConvergence Statistics:")
            print(f"Initial mean log prob: {convergence_data['mean_log_prob_history'][0]:.2f}")
            print(f"Final mean log prob: {convergence_data['mean_log_prob_history'][-1]:.2f}")
            print(f"Total improvement: {convergence_data['mean_log_prob_history'][-1] - convergence_data['mean_log_prob_history'][0]:.2f}")

    def plot_fit_comparison(self, params_dict, n_sims=None, figsize=(12, 8), save_path=None):
        """
        Plot the simulation results using fitted parameters compared to the actual data.
        
        Parameters:
        -----------
        params_dict : dict
            Dictionary of fitted parameters
        n_sims : int, optional
            Number of simulations to run (defaults to self.n_sims)
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            If provided, save the plot to this path
        """
        import matplotlib.pyplot as plt
        
        # Use class n_sims if not specified
        if n_sims is None:
            n_sims = self.n_sims
            
        # Create parameter objects for simulation
        params = TransitionParams(
            q00_01=params_dict['q00_01'], q00_10=params_dict['q00_10'],
            q01_00=params_dict['q01_00'], q01_11=params_dict['q01_11'],
            q10_00=params_dict['q10_00'], q10_11=params_dict['q10_11'],
            q11_01=params_dict['q11_01'], q11_10=params_dict['q11_10']
        )
        
        init_dict = {
            'state_00': params_dict['init_00'],
            'state_01': params_dict['init_01'],
            'state_10': params_dict['init_10'],
            'state_11': params_dict['init_11']
        }
        
        # Run simulations with fitted parameters
        results = run_simulations(params, init_dict, n_sims, list(self.data_df.index))
        
        # Calculate proportions from simulations
        sim_props = {}
        for t, sims in results.items():
            points = np.array(sims)
            sim_props[t] = {
                'hihi': np.mean((points[:,0] > params_dict['threshold']) & (points[:,1] > params_dict['threshold'])),
                'hilo': np.mean((points[:,0] > params_dict['threshold']) & (points[:,1] <= params_dict['threshold'])),
                'lohi': np.mean((points[:,0] <= params_dict['threshold']) & (points[:,1] > params_dict['threshold'])),
                'lolo': np.mean((points[:,0] <= params_dict['threshold']) & (points[:,1] <= params_dict['threshold']))
            }
        
        # Calculate proportions from data
        data_props = {}
        for t in self.data_df.index:
            row = self.data_df.loc[t]
            total = row.sum()
            data_props[t] = {
                'hihi': row['f1f2_hi'] / total,
                'hilo': row['f1_hi'] / total,
                'lohi': row['f2_hi'] / total,
                'lolo': row['f1f2_lo'] / total
            }
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        states = ['hihi', 'hilo', 'lohi', 'lolo']
        titles = ['Hi/Hi', 'Hi/Lo', 'Lo/Hi', 'Lo/Lo']
        
        for (state, title, ax) in zip(states, titles, axes.flat):
            # Plot simulation results
            sim_values = [sim_props[t][state] for t in sorted(results.keys())]
            ax.plot(sorted(results.keys()), sim_values, '-', label='Simulation')
            
            # Plot actual data
            data_values = [data_props[t][state] for t in sorted(results.keys())]
            ax.plot(sorted(results.keys()), data_values, 'o', label='Data')
            
            ax.set_title(f'State {title}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Proportion')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved fit comparison plot to {save_path}")
        
        plt.show()

    def run_multi_tempered_chains(self, n_attempts=3, n_temps=5, n_walkers=32, n_steps=1000, n_cores=8):
        """
        Run multiple parallel tempering chains from different starting points.
        
        Parameters:
        -----------
        n_attempts : int
            Number of different starting points to try
        n_temps : int
            Number of temperature levels per attempt
        n_walkers : int
            Number of walkers per temperature
        n_steps : int
            Number of steps per walker
        n_cores : int
            Number of CPU cores to use
        
        Returns:
        --------
        list of lists
            Results for each attempt and temperature combination
        """
        import time
        
        # Create temperature ladder
        temps = np.logspace(0, 4, n_temps)
        
        if self.verbose:
            print(f"\nStarting multi-tempered chains with:")
            print(f"- {n_attempts} different starting points")
            print(f"- {n_temps} temperature levels: {temps}")
            print(f"- {n_walkers} walkers per temperature")
            print(f"- {n_steps} steps per walker")
            print(f"- {n_cores} CPU cores")
        
        all_results = []
        start_time = time.time()
        
        for attempt in range(n_attempts):
            if self.verbose:
                print(f"\nStarting attempt {attempt + 1}/{n_attempts}")
                attempt_start = time.time()
            
            # Run parallel tempering for this attempt
            attempt_results = []
            
            for i, temp in enumerate(temps):
                if self.verbose:
                    print(f"\nRunning chain at temperature {temp:.2f} ({i+1}/{n_temps})")
                
                # Create new inference object with this temperature
                temp_inference = CellInference(
                    self.data_df,
                    n_sims=self.n_sims,
                    threshold_bounds=self.threshold_bounds,
                    verbose=False,
                    temp_scale=temp
                )
                
                # Run MCMC with randomized starting positions
                sampler, conv_data = temp_inference.run_inference(
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    n_cores=n_cores,
                    randomize_init=True  # This flag would need to be added to run_inference
                )
                
                # Analyze results
                params, samples, log_probs = temp_inference.analyze_results(
                    sampler,
                    conv_data,
                    burnin=n_steps // 3
                )
                
                attempt_results.append({
                    'temperature': temp,
                    'params': params,
                    'log_probs': log_probs,
                    'samples': samples,
                    'convergence_data': conv_data
                })
                
                if self.verbose:
                    elapsed = time.time() - attempt_start
                    print(f"Temperature {temp:.2f} completed in {elapsed/60:.1f} minutes")
                    print(f"Final log probability: {np.mean(log_probs):.2f}")
            
            all_results.append(attempt_results)
            
            if self.verbose:
                attempt_time = time.time() - attempt_start
                print(f"\nAttempt {attempt + 1} completed in {attempt_time/60:.1f} minutes")
        
        if self.verbose:
            total_time = time.time() - start_time
            print(f"\nAll attempts completed in {total_time/60:.1f} minutes")
        
        return all_results

    def analyze_multi_tempered_chains(self, all_results):
        """
        Analyze results from multiple parallel tempering runs.
        
        Parameters:
        -----------
        all_results : list of lists
            Results from run_multi_tempered_chains
        """
        print("\nMulti-Tempered Chains Analysis:")
        print("------------------------------")
        
        best_logprob = -np.inf
        best_params = None
        best_temp = None
        best_attempt = None
        
        for attempt, attempt_results in enumerate(all_results):
            print(f"\nAttempt {attempt + 1}:")
            
            for result in attempt_results:
                temp = result['temperature']
                params = result['params']
                log_probs = result['log_probs']
                mean_logprob = np.mean(log_probs)
                
                print(f"\n  Temperature {temp:.2f}:")
                print(f"  Mean log probability: {mean_logprob:.2f}")
                
                if mean_logprob > best_logprob:
                    best_logprob = mean_logprob
                    best_params = params
                    best_temp = temp
                    best_attempt = attempt + 1
        
        print(f"\nBest result found:")
        print(f"Attempt: {best_attempt}")
        print(f"Temperature: {best_temp:.2f}")
        print(f"Log probability: {best_logprob:.2f}")
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.3f}")
        
        return best_params

