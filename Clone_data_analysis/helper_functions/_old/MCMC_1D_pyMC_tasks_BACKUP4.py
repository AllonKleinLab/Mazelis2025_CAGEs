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
        Load and validate the clone count data
        
        Args:
            data: DataFrame containing the data
                 Expected format: index=[0,2,4,6], columns=['n_hi', 'n_lo']
        """
        expected_indices = [0, 2, 4, 6]
        expected_columns = ['n_hi', 'n_lo']
        
        # Validate data format
        if not all(idx in data.index for idx in expected_indices):
            raise ValueError(f"Data must have time points {expected_indices}")
        if not all(col in data.columns for col in expected_columns):
            raise ValueError(f"Data must have columns {expected_columns}")
            
        self.data = data
        # Convert to numpy arrays for PyMC
        self.times = self.data.index.values
        self.n_hi = self.data['n_hi'].values
        self.n_lo = self.data['n_lo'].values

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

    def compute_prob_high(self, t: int, p0, q01, q10, threshold):
        """
        Compute probability of being in high state (x > threshold)
        
        Args:
            t: Time point (number of divisions)
            p0, q01, q10: Model parameters
            threshold: Threshold value
            
        Returns:
            Probability of being in high state
        """
        if t == 0:
            # For t=0, analytical solution
            return 1 - p0 if isinstance(p0, float) else pm.math.switch(threshold < 1, 1 - p0, 0)
        
        # Use numpy for regular floats, pm.math for tensors
        if all(isinstance(x, float) for x in [p0, q01, q10]):
            x_values, px = self.compute_px_distribution(t, p0, q01, q10)
            high_mask = x_values > threshold
            return np.sum(px[high_mask])
        else:
            # For PyMC tensors, we need to evaluate the surrogate model
            # and sum probabilities differently
            params = np.array([[float(q01.eval()), float(q10.eval())]])
            x_values, px = self.compute_px_distribution(t, float(p0.eval()), 
                                                      params[0,0], params[0,1])
            high_mask = x_values > float(threshold.eval())
            return pm.math.constant(np.sum(px[high_mask]))

    def compute_log_likelihood(self, prob_high):
        """
        Compute the log likelihood of the data given the model probabilities
        
        Args:
            prob_high: Array of probabilities of being in high state at each time point
            
        Returns:
            Log likelihood value
        """
        # Ensure probabilities are valid
        prob_high = pm.math.clip(prob_high, 1e-10, 1 - 1e-10)
        prob_low = 1 - prob_high
        
        # Compute log likelihood
        log_likelihood = pm.math.sum(
            self.n_hi * pm.math.log(prob_high) +
            self.n_lo * pm.math.log(prob_low)
        )
        
        return log_likelihood

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
            threshold = pm.Uniform('threshold', lower=0.01, upper=0.1)
            
            # Deterministic transformations
            exp_term = 1 - pm.math.exp(-S)
            q01 = pm.Deterministic('q01', exp_term / (1 + r))
            q10 = pm.Deterministic('q10', exp_term - exp_term / (1 + r))
            
            # Additional deterministic variables for rates
            r01 = pm.Deterministic('r01', S / (1 + r))
            r10 = pm.Deterministic('r10', S / (1 + 1/r))
            
            # Likelihood computation remains the same
            params = pm.math.stack([q01, q10], axis=1)
            
            # Compute probabilities for each timepoint
            logp = 0
            for t_idx, t in enumerate(self.times):
                if t == 0:
                    # Analytical solution for t=0
                    prob_hi = 1 - p0
                else:
                    # Use surrogate model through PyTensor Op
                    px = self.surrogate_ops[t](params, p0)
                    # Sum probabilities where x > threshold
                    x_values = np.linspace(0, 1, 2**t + 1)
                    high_mask = x_values > threshold
                    prob_hi = pt.sum(px[high_mask])
                
                # Add to log likelihood
                prob_hi = pt.clip(prob_hi, 1e-10, 1 - 1e-10)
                logp += (self.n_hi[t_idx] * pt.log(prob_hi) + 
                        self.n_lo[t_idx] * pt.log(1 - prob_hi))
            
            # Add the logp to the model
            pm.Potential("likelihood", logp)

    def run_inference(self, draws, tune, chains, cores):
        """
        Run MCMC inference with refined step sizes and more chains
        """
        with self.model:
            # Metropolis steps with tune_interval specified
            step = pm.CompoundStep([
                pm.Metropolis(vars=[self.model.p0], scaling=1.0, tune_interval=100),
                pm.Metropolis(vars=[self.model.log_r], scaling=1.0, tune_interval=100),
                pm.Metropolis(vars=[self.model.log_S], scaling=1.0, tune_interval=100),
                pm.Metropolis(vars=[self.model.threshold], scaling=1.0, tune_interval=100)
            ])
            
            # Run inference with more chains and tuning
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step
            )

    def create_mock_data(self, true_params: Dict[str, float]) -> pd.DataFrame:
        """
        Create mock data using true parameters and surrogate models
        
        Args:
            true_params: Dictionary with keys 'p0', 'q01', 'q10', 'threshold'
                        containing the true parameter values
        
        Returns:
            DataFrame with mock data matching the expected format
        """
        # Extract parameters
        p0 = true_params['p0']
        q01 = true_params['q01']
        q10 = true_params['q10']
        threshold = true_params['threshold']
        
        # Initialize data storage
        mock_data = {'n_hi': [], 'n_lo': []}
        times = [0, 2, 4, 6]
        n_cells = 1000  # Total number of cells/clones to simulate
        
        for t in times:
            # Get probability of being in high state
            prob_high = self.compute_prob_high(t, p0, q01, q10, threshold)
            
            # Generate binomial samples
            n_hi = np.random.binomial(n_cells, prob_high)
            n_lo = n_cells - n_hi
            
            mock_data['n_hi'].append(n_hi)
            mock_data['n_lo'].append(n_lo)
            
        return pd.DataFrame(mock_data, index=times)

    def plot_diagnostics(self, figsize=(12, 10)):
        """
        Plot MCMC diagnostics including traces and posterior distributions
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
        
        # Get MAP estimates and log prob
        map_estimates, log_prob = self.get_map_estimate()
        
        # Get summary statistics
        summary = az.summary(self.trace, round_to=3)
        
        # Parameters to plot
        params = ['p0', 'S', 'r', 'threshold', 'r01', 'r10']
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(params), 2, figsize=figsize)
        
        for i, param in enumerate(params):
            # Plot trace manually
            axes[i, 0].plot(self.trace.posterior[param].values.T)
            axes[i, 0].set_title(f'{param} trace')
            
            # For rate parameters, use log scale on x-axis
            if param in ['S', 'r', 'r01', 'r10']:
                # Plot histogram in log space
                axes[i, 1].hist(np.log10(self.trace.posterior[param].values.flatten()), 
                              bins=30, density=True)
                axes[i, 1].set_xlabel(f'log10({param})')
                
                # Add MAP estimate and mean as vertical lines
                mean_val = summary.loc[param, 'mean']
                rhat = summary.loc[param, 'r_hat']
                axes[i, 1].axvline(np.log10(mean_val), color='g', linestyle='-', 
                                 label=f'Mean: {mean_val:.2e}\nR-hat: {rhat:.3f}')
                axes[i, 1].axvline(np.log10(map_estimates[param]), color='r', 
                                 linestyle='--', 
                                 label=f'MAP: {map_estimates[param]:.2e}')
            else:
                # Regular linear scale for other parameters
                axes[i, 1].hist(self.trace.posterior[param].values.flatten(), 
                              bins=30, density=True)
                axes[i, 1].set_xlabel(param)
                
                # Add MAP estimate and mean as vertical lines
                mean_val = summary.loc[param, 'mean']
                rhat = summary.loc[param, 'r_hat']
                axes[i, 1].axvline(mean_val, color='g', linestyle='-', 
                                 label=f'Mean: {mean_val:.3f}\nR-hat: {rhat:.3f}')
                axes[i, 1].axvline(map_estimates[param], color='r', 
                                 linestyle='--',
                                 label=f'MAP: {map_estimates[param]:.3f}')
            
            axes[i, 1].set_title(f'{param} posterior')
            axes[i, 1].legend()
        
        plt.tight_layout()
        return map_estimates

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

    def plot_predictions(self, data, est_params=None, figsize=(4, 4)):
        """
        Plot model predictions of high state fraction against observed data
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
            
        # Get parameter estimates if not provided
        if est_params is None:
            summary = az.summary(self.trace)
            est_params = {
                'p0': float(summary.loc['p0', 'mean']),
                'q01': float(summary.loc['q01', 'mean']),
                'q10': float(summary.loc['q10', 'mean']),
                'threshold': float(summary.loc['threshold', 'mean'])
            }
            print("\nUsing mean parameter values:")
            for param, value in est_params.items():
                print(f"{param}: {value:.3f}")
        
        # Calculate observed fractions and errors
        total_cells = data['n_hi'] + data['n_lo']
        f_hi_obs = data['n_hi'] / total_cells
        se = np.sqrt(f_hi_obs * (1 - f_hi_obs) / total_cells)
        
        # Generate predictions
        predictions = np.zeros(len(self.times))
        for i, t in enumerate(self.times):
            if t == 0:
                predictions[i] = 1 - est_params['p0']
            else:
                params = np.array([[est_params['q01'], est_params['q10']]])
                px = self.surrogate_ops[t].torch_model.predict(params, q0=est_params['p0'])
                x_values = np.linspace(0, 1, 2**t + 1)
                high_mask = x_values > est_params['threshold']
                predictions[i] = np.sum(px[0][high_mask])
        
        # Plot results
        plt.figure(figsize=figsize)
        plt.errorbar(self.times, f_hi_obs, yerr=se, 
                    fmt='o', color='blue', label='Observed', capsize=5)
        plt.plot(self.times, predictions, 'b--', label='Predicted')
        
        plt.xlabel('Time')
        plt.ylabel('Fraction in High State')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        
        # Debug print
        print("Predictions:", predictions)
        