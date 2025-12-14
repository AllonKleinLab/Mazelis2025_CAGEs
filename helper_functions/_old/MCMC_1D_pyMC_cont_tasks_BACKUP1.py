import os
os.environ["PYTHONPATH"] = os.getcwd()

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import cpu_count
import pymc as pm
import arviz as az
import torch
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

# Import the surrogate model module
import MCMC_1D_surrogate_model_tasks as mcmc1D_surrogate

class PyTorchSurrogateOp(Op):
    """
    Custom PyTensor Op that wraps a PyTorch surrogate model
    """
    def __init__(self, torch_model):
        self.torch_model = torch_model
        self.model = torch_model  # Keep reference to avoid garbage collection

    def make_node(self, params, p0):
        # Convert PyTensor tensors to PyTorch tensors
        params = pt.as_tensor_variable(params)
        p0 = pt.as_tensor_variable(p0)
        return Apply(self, [params, p0], [params.type()])

    def perform(self, node, inputs, outputs):
        # This is where we run the surrogate model
        params = inputs[0].reshape(1, 2)  # Reshape to [[q01, q10]]
        p0 = inputs[1]
        # Run model using predict method
        #print('Debug - params:', params)
        y = self.torch_model.predict(params, q0=float(p0))
        #print('Debug - y:', y)
        # Set output
        outputs[0][0] = y[0]  # Take first prediction since we only have one sample

    def grad(self, inputs, output_grads):
        # For Metropolis, we don't need gradients
        return [None, None]

class MCMCInference:
    def __init__(self, 
                 data: pd.DataFrame = None, 
                 surrogate_model_path: str = None,
                 model_config: Dict = None):
        """
        Initialize the MCMC inference class
        
        Args:
            data: DataFrame containing the data
            surrogate_model_path: Path to saved surrogate models
            model_config: Dictionary containing model configuration parameters
        """
        self.data = None
        self.model_config = model_config or {}
        self.trace = None
        self.model = None
        self.surrogate_models = {}
        self.surrogate_ops = {}  # Will store PyTensor Ops for each timepoint
        
        if data is not None:
            self.load_data(data)
        if surrogate_model_path is not None:
            self.load_surrogate_models(surrogate_model_path)
        
        # Debugging statements
        # print("MCMCInference initialized.")
        # print(f"Data: {self.data}")
        # print(f"Surrogate Models: {self.surrogate_models}")
        # print(f"Surrogate Ops: {self.surrogate_ops}")
        
    def load_data(self, data: pd.DataFrame):
        """
        Load and validate the single-cell data
        
        Args:
            data: DataFrame containing the data
                 Expected format: columns=['timepoint', 'score']
                 where each row represents a single colony measurement
        """
        expected_columns = ['timepoint', 'score']
        expected_timepoints = [0, 2, 4, 6]
        
        # Validate data format
        if not all(col in data.columns for col in expected_columns):
            raise ValueError(f"Data must have columns {expected_columns}")
        
        # Validate that we have data for expected timepoints
        if not all(t in data['timepoint'].unique() for t in expected_timepoints):
            raise ValueError(f"Data must have measurements for timepoints {expected_timepoints}")
            
        self.data = data
        # Store timepoints and sort them
        self.times = np.sort(data['timepoint'].unique())
        
        # For compatibility with existing code, we'll also store counts
        # We'll compute these dynamically when needed in other methods
        self.raw_data = data  # Store the original single-colony data

    def load_surrogate_models(self, model_path: str):
        """
        Load surrogate models and create PyTensor Ops for each
        """
        self.surrogate_models = mcmc1D_surrogate.load_model_dict(model_path)
        
        # Create PyTensor Ops for each model
        for t, model in self.surrogate_models.items():
            self.surrogate_ops[t] = PyTorchSurrogateOp(model)
            # Debugging statements
            # print(f"Loaded surrogate model for t={t}")
            # print(f"Surrogate Op for t={t}: {self.surrogate_ops[t]}")
            
        # Debugging statements
        # print(f"Loaded surrogate models for divisions: {list(self.surrogate_models.keys())}")
        # print(f"Surrogate Ops: {self.surrogate_ops}")
            
    def compute_px_distribution(self, t: int, p0: float, q01: float, q10: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P(x|t,params) and corresponding x values for a given timepoint and parameters
        
        Args:
            t: Time point (number of divisions)
            p0: Initial probability (passed as q0 to the surrogate model)
            q01, q10: Transition probabilities
            
        Returns:
            Tuple of (x_values, probabilities):
                x_values: Array of possible x values (0, 1/2^t, ..., 1)
                probabilities: Array of corresponding probabilities P(x)
        """
        if t == 0:
            return np.array([0, 1]), np.array([p0, 1-p0])
        
        if t not in self.surrogate_models:
            raise ValueError(f"No surrogate model available for t={t}")
            
        # Create parameter array for (q01, q10)
        params = np.array([[q01, q10]])
        
        # Get probabilities from surrogate model
        probs = self.surrogate_models[t].predict(params, q0=p0)
        
        # Generate corresponding x values: 0, 1/2^t, 2/2^t, ..., 1
        n_points = 2**t + 1
        x_values = np.linspace(0, 1, n_points)
        
        return x_values, probs[0]  # probs[0] because predict returns a 2D array



    def compute_log_likelihood(self, params, p0, a, sigma, threshold):
        """
        Compute log likelihood for observations at time t
        
        Args:
            params: Array of [q01, q10]
            p0: Initial probability
            a: Location parameter for log-normal
            sigma: Scale parameter for log-normal
            threshold: Threshold value
            
        Returns:
            Log likelihood for all observations at time t
        """
        # Initialize log likelihood
        logp = 0

        # Loop over each timepoint
        for t in self.times:
            # Get data for this timepoint
            x_t = self.data[self.data['timepoint'] == t]['score'].values
            
            # Get surrogate model predictions P(n,t)
            if t == 0:
                px = np.array([p0, 1-p0])
                n_values = np.array([0, 1])
            else:
                n_values = np.linspace(0, 1, 2**t + 1)
                px = self.surrogate_ops[t](params, p0)
        
            # Split computation based on threshold
            below_thresh = x_t < threshold
            above_thresh = ~below_thresh
        
            # For x < threshold, use P(n=0)
            logp += pt.sum(below_thresh) * pt.log(px[0])
            
            # For x ≥ threshold, compute all log-normal probabilities at once
            x_above = x_t[above_thresh]
            if len(x_above) > 0:
                # Compute log-normal probabilities for all x and n>0 combinations
                # The following operation gives an nxlen(x_above) matrix of probs
                # the operation .logp(..) gives log-probs, and .exp() gives probs
                lognormal_probs = pm.LogNormal.dist(
                    mu=pt.log(n_values[1:]).reshape(-1, 1) + a,
                    sigma=sigma
                ).logp(x_above).exp()
                
                # Multiply by P(n,t) and sum over n for each x
                # This line is \sum_x log [\sum_n P(x|n,t) P(n,t) ]
                logp += pt.sum(pt.log(pt.sum(lognormal_probs.T * px[1:], axis=1)))
        
        return logp

    def setup_model(self):
        """
        Set up PyMC model with the new parameterization using r and S
        """
        with pm.Model() as self.model:
            # Priors - using log transform for LogUniform
            log_S = pm.Uniform('log_S', lower=np.log(1e-4), upper=np.log(1e1))
            log_r = pm.Uniform('log_r', lower=np.log(1e-2), upper=np.log(1e2))
            S = pm.Deterministic('S', pm.math.exp(log_S))
            r = pm.Deterministic('r', pm.math.exp(log_r))
            
            p0 = pm.Uniform('p0', lower=0, upper=1)
            threshold = pm.Uniform('threshold', lower=0, upper=2e-3)
            
            # New parameters for modeling full NMF scores
            a0, sigma0 = self.estimate_lognormal_params()
            a = pm.Normal('a', mu=a0, sigma=3*sigma0)  # Location parameter for log-normal
            sigma = pm.HalfNormal('sigma', sigma=2*sigma0)  # Scale parameter for log-normal
            
            # Deterministic transformations
            exp_term = 1 - pm.math.exp(-S)
            q01 = pm.Deterministic('q01', exp_term / (1 + r))
            q10 = pm.Deterministic('q10', exp_term / (1 + 1/r))
            
            # Additional deterministic variables for rates
            r01 = pm.Deterministic('r01', S / (1 + r))
            r10 = pm.Deterministic('r10', S / (1 + 1/r))
            
            # Stack parameters for likelihood computation
            params = pm.math.stack([q01, q10], axis=1)
            
            # Compute log likelihood using new continuous model
            logp = self.compute_log_likelihood(params, p0, a, sigma, threshold)
            
            # Add the logp to the model
            pm.Potential("likelihood", logp)

    def run_inference(self, draws, tune, chains, cores, show_progressbar=True, compute_convergence_checks=True):
        """
        Run MCMC inference with refined step sizes and more chains
        """
        with self.model:
            # Metropolis steps with tune_interval specified
            step = pm.CompoundStep([
                pm.Metropolis(vars=[self.model.p0], scaling=0.1, tune_interval=100),
                pm.Metropolis(vars=[self.model.log_r], scaling=0.1, tune_interval=100),
                pm.Metropolis(vars=[self.model.log_S], scaling=0.1, tune_interval=100),
                pm.Metropolis(vars=[self.model.threshold], scaling=0.1, tune_interval=100)
            ])
            
            # Run inference with more chains and tuning
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step,
                progressbar=show_progressbar,  # Disable progress bar
                compute_convergence_checks=compute_convergence_checks  # Disable convergence warnings
            )

    def create_mock_data(self, 
                        r01: float = 0.2,
                        r10: float = 0.1,
                        p0: float = 0.5,
                        threshold: float = 1e-3,
                        a: float = -1.5,
                        sigma: float = 0.5,
                        n_colonies: int = 1000) -> pd.DataFrame:
        """
        Create mock data using provided parameters
        
        Args:
            r01: ON->OFF rate (default 0.2)
            r10: OFF->ON rate (default 0.1)
            p0: Initial ON probability (default 0.8)
            threshold: Score threshold (default 1e-3)
            a: Location parameter for log-normal (default -3.0)
            sigma: Scale parameter for log-normal (default 0.5)
            n_colonies: Number of colonies to simulate (default 1000)
            
        Returns:
            DataFrame with columns ['timepoint', 'score']
        """
        # Calculate S and r from rates
        S = r01 + r10
        r = r10 / r01
        
        # Calculate q01, q10
        exp_term = 1 - np.exp(-S)
        q01 = exp_term / (1 + r)
        q10 = exp_term / (1 + 1/r)
        
        # Initialize data storage
        all_data = []
        times = [0, 2, 4, 6]
        
        for t in times:
            # Get probability distribution P(n) from surrogate model
            if t == 0:
                px = np.array([p0, 1-p0])
                n_values = np.array([0, 1])
            else:
                n_values = np.linspace(0, 1, 2**t + 1)
                params = np.array([[q01, q10]])
                px = self.surrogate_ops[t].torch_model.predict(params, q0=p0)[0]
            
            # Generate "n" values for each colony
            n_samples = np.random.choice(n_values, size=n_colonies, p=px)
            
            # Generate scores
            scores = np.zeros(n_colonies)
            zero_mask = n_samples == 0
            nonzero_mask = ~zero_mask

            if np.any(zero_mask):
                # For n=0, generate values from 0 to threshold
                scores[zero_mask] = np.random.uniform(0, threshold, size=np.sum(zero_mask))

            if np.any(nonzero_mask):
                # For n>0, generate log-normal samples
                lognormal_samples = np.random.lognormal(
                    mean=np.log(n_samples[nonzero_mask]) + a,
                    sigma=sigma
                )
                scores[nonzero_mask] = lognormal_samples
            
            # Create DataFrame for this timepoint
            timepoint_data = pd.DataFrame({
                'timepoint': [t] * n_colonies,
                'score': scores
            })
            
            all_data.append(timepoint_data)
        
        # Combine all timepoints
        mock_data = pd.concat(all_data, ignore_index=True)
        
        return mock_data

    def plot_diagnostics(self, figsize=(8, 8), figsize_pair=(7, 7)):
        """
        Plot MCMC diagnostics including traces, posterior distributions, and pair plots
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
        
        # Get summary statistics
        summary = az.summary(self.trace, round_to=3)
        
        # Print parameter estimates
        print("\nParameter Estimates:")
        print("-" * 50)
        for param in ['p0', 'S', 'r', 'threshold', 'a', 'sigma']:
            est_mean = summary.loc[param, 'mean']
            est_hdi = (summary.loc[param, 'hdi_3%'], summary.loc[param, 'hdi_97%'])
            rhat = summary.loc[param, 'r_hat']
        
        # Create plots
        params = ['p0', 'S', 'r', 'threshold', 'a', 'sigma']
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(len(params), 2)
        
        for i, param in enumerate(params):
            # Create subplot for trace
            ax_trace = fig.add_subplot(gs[i, 0])
            ax_hist = fig.add_subplot(gs[i, 1])
            
            # Get the trace data
            trace_data = self.trace.posterior[param].values
            
            # Plot trace for each chain
            for chain in range(trace_data.shape[0]):
                ax_trace.plot(trace_data[chain], alpha=0.5)
            ax_trace.set_title(f'{param} trace')
            ax_trace.set_xlabel('Sample')
            ax_trace.set_ylabel(param)
            
            # Plot histogram
            if param in ['S', 'r']:
                # Plot in log space
                ax_hist.hist(np.log10(trace_data.flatten()), bins=30, density=True)
                ax_hist.set_xlabel(f'log10({param})')
                
                # Add mean and HDI
                ax_hist.axvline(np.log10(est_mean), color='r', 
                              linestyle='--', 
                              label=f'Mean: {est_mean:.2e}\nR-hat: {rhat:.3f}')
                ax_hist.axvline(np.log10(est_hdi[0]), color='g', linestyle=':')
                ax_hist.axvline(np.log10(est_hdi[1]), color='g', linestyle=':',
                              label='HDI')
            else:
                # Regular linear scale
                ax_hist.hist(trace_data.flatten(), bins=30, density=True)
                ax_hist.set_xlabel(param)
                
                # Add mean and HDI
                ax_hist.axvline(est_mean, color='r', 
                              linestyle='--',
                              label=f'Mean: {est_mean:.3f}\nR-hat: {rhat:.3f}')
                ax_hist.axvline(est_hdi[0], color='g', linestyle=':')
                ax_hist.axvline(est_hdi[1], color='g', linestyle=':',
                              label='HDI')
            
            ax_hist.set_title(f'{param} posterior')
            ax_hist.legend()
        
        plt.tight_layout()
        
        # After the trace plots, add pair plots
        plt.figure(figsize=figsize_pair)
        ax = az.plot_pair(
            self.trace, 
            var_names=['p0', 'S', 'r', 'threshold', 'a', 'sigma'],
            kind='kde',
            marginals=True,
            textsize=10,
            figsize=figsize_pair,
            kde_kwargs={'contour': True},
        )
        
        # Adjust tick label sizes
        for row in ax:
            for ax_i in row:
                if ax_i is not None:
                    ax_i.tick_params(labelsize=10)
        
        plt.tight_layout()

    def get_map_estimate(self):
        """
        Get parameter estimates using posterior mean
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
        
        print("Using posterior mean as estimate")
        summary = az.summary(self.trace)
        map_estimate = {
            'p0': float(summary.loc['p0', 'mean']),
            'S': float(summary.loc['S', 'mean']),
            'r': float(summary.loc['r', 'mean']),
            'threshold': float(summary.loc['threshold', 'mean']),
            'r01': float(summary.loc['r01', 'mean']),
            'r10': float(summary.loc['r10', 'mean'])
        }
        
        print("Parameter estimates:")
        for param, value in map_estimate.items():
            if param in ['S', 'r', 'r01', 'r10']:
                print(f"{param}: {value:.2e}")
            else:
                print(f"{param}: {value:.4f}")
        
        return map_estimate, float('nan')

    def plot_predictions(self, figsize=(6, 6)):
        """
        Plot model predictions against data in multiple panels
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
            
        # Get parameter estimates from trace
        summary = az.summary(self.trace)
        est_params = {
            'p0': float(summary.loc['p0', 'mean']),
            'q01': float(summary.loc['q01', 'mean']),
            'q10': float(summary.loc['q10', 'mean']),
            'threshold': float(summary.loc['threshold', 'mean']),
            'a': float(summary.loc['a', 'mean']),
            'sigma': float(summary.loc['sigma', 'mean'])
        }
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(2, 3)
        
        # Store data for summary plots
        empirical_nonzero = []
        predicted_nonzero = []
        empirical_means = []
        predicted_means = []
        
        # Plot histograms for each timepoint
        for i, t in enumerate(self.times):
            ax = fig.add_subplot(gs[i//3, i%3])
            
            # Get data for this timepoint
            x_t = self.data[self.data['timepoint'] == t]['score'].values
            
            # Plot empirical distribution
            mask = x_t > 0
            if np.any(mask):
                plt.hist(np.log10(est_params['threshold'] + x_t[mask]), 
                        bins=30, density=True, alpha=0.5, color='green', 
                        label='Data')
            
            # Generate predicted distribution
            if t == 0:
                px = np.array([est_params['p0'], 1-est_params['p0']])
                n_values = np.array([0, 1])
            else:
                n_values = np.linspace(0, 1, 2**t + 1)
                params = np.array([[est_params['q01'], est_params['q10']]])
                px = self.surrogate_ops[t].torch_model.predict(params, q0=est_params['p0'])[0]
            
            # Store fraction non-zero
            empirical_nonzero.append(np.mean(x_t > est_params['threshold']))
            predicted_nonzero.append(np.sum(px[1:]))  # Sum of all n>0 probabilities
            
            # Store means
            empirical_means.append(np.mean(x_t))
            predicted_mean = 0  # Add contribution from each n
            for n, p_n in zip(n_values, px):
                if n == 0:
                    continue
                # Expected value of lognormal with given n
                predicted_mean += p_n * np.exp(np.log(n) + est_params['a'] + 
                                             est_params['sigma']**2/2)
            predicted_means.append(predicted_mean)
            
            # Generate points for predicted distribution
            x_plot = np.logspace(np.log10(est_params['threshold']), 
                               np.log10(est_params['threshold'] + 1), 100)
            y_plot = np.zeros_like(x_plot)
            
            # Add contribution from each n>0
            for n, p_n in zip(n_values[1:], px[1:]):
                y_plot += p_n * np.exp(-(np.log(x_plot-est_params['threshold']) - 
                                       (np.log(n) + est_params['a']))**2 / 
                                     (2*est_params['sigma']**2)) / \
                         (x_plot-est_params['threshold']) / \
                         (est_params['sigma']*np.sqrt(2*np.pi))
            
            plt.plot(np.log10(x_plot), y_plot, 'k-', label='Model')
            plt.title(f't = {t}')
            plt.xlabel('log10(threshold + score)')
            plt.ylabel('Density')
            if i == 0:
                plt.legend()
        
        # Plot fraction non-zero
        ax = fig.add_subplot(gs[1, 0])
        plt.plot(self.times, empirical_nonzero, 'go', label='Data')
        plt.plot(self.times, predicted_nonzero, 'k-', label='Model')
        plt.xlabel('Time')
        plt.ylabel('Fraction non-zero')
        plt.legend()
        
        # Plot mean values
        ax = fig.add_subplot(gs[1, 1])
        plt.plot(self.times, empirical_means, 'go', label='Data')
        plt.plot(self.times, predicted_means, 'k-', label='Model')
        plt.xlabel('Time')
        plt.ylabel('Mean score')
        plt.legend()
        
        plt.tight_layout()

    def estimate_lognormal_params(self):
        """
        Estimate initial values for log-normal parameters a and sigma using all score data
        
        Returns:
            tuple: (a_init, sigma_init) - Initial estimates for location and scale parameters
        """
        # Get all scores
        scores = self.data['score'].values
        
        # Compute mean and std of log(x)
        log_scores = np.log(scores[scores > 0])  # Avoid log(0)
        
        # For lognormal distribution:
        # If Y = log(X), then Y ~ Normal(mu, sigma)
        # where mu = log(median(X)) = a
        # and sigma is std of log(X)
        
        a_init = np.mean(log_scores)
        sigma_init = np.std(log_scores)
        
        return a_init, sigma_init

def create_mcmc_summary_row(mcmc, f1, cond):
    """
    Create a summary row from MCMC trace results with full precision
    
    Args:
        mcmc: MCMCInference object with trace
        f1: program_id (list or int)
        cond: condition name (str)
    
    Returns:
        dict: Dictionary containing summary statistics for all parameters
    """
    # Convert f1 to integer if it's a list
    program_id = f1[0] if isinstance(f1, list) else f1
    
    # Get summary for HDI and std
    summary = az.summary(mcmc.trace)
    
    # Initialize row data
    row_data = {
        'program_id': program_id,
        'condition': cond,
    }
    
    # Parameters to process
    params = ['r01', 'r10', 'q01', 'q10', 'S', 'r', 'p0', 'threshold']
    
    # Get high precision means directly from trace
    for param in params:
        # Mean with full precision
        row_data[f'{param}_mean'] = float(mcmc.trace.posterior[param].mean())
        # Other stats from summary
        row_data[f'{param}_std'] = float(summary.loc[param, 'sd'])
        row_data[f'{param}_hdi_low'] = float(summary.loc[param, 'hdi_3%'])
        row_data[f'{param}_hdi_high'] = float(summary.loc[param, 'hdi_97%'])
    
    return row_data