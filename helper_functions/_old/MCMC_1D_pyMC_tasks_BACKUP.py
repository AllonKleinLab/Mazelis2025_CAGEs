import os
os.environ["PYTHONPATH"] = os.getcwd()

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import cpu_count

# PyMC imports with minimal configuration
import pymc as pm
import arviz as az

# Import the surrogate model module
import MCMC_1D_surrogate_model_tasks as mcmc1D_surrogate

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
        self.surrogate_models = None
        
        if data is not None:
            self.load_data(data)
        if surrogate_model_path is not None:
            self.load_surrogate_models(surrogate_model_path)
        
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
        Load surrogate models using the existing functionality
        
        Args:
            model_path: Path to saved surrogate models
        """
        try:
            self.surrogate_models = mcmc1D_surrogate.load_model_dict(model_path)
            print(f"Loaded surrogate models for divisions: {list(self.surrogate_models.keys())}")
        except Exception as e:
            raise ValueError(f"Error loading surrogate models: {str(e)}")
            
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

    def compute_prob_high(self, t: int, p0: float, q01: float, q10: float, threshold: float) -> float:
        """
        Compute probability of being in high state (x > threshold)
        
        Args:
            t: Time point (number of divisions)
            p0, q01, q10: Model parameters
            threshold: Threshold value
            
        Returns:
            Probability of being in high state
        """
        # Convert inputs to numpy if they're not already
        p0_val = float(p0)
        q01_val = float(q01)
        q10_val = float(q10)
        threshold_val = float(threshold)
        
        x_values, px = self.compute_px_distribution(t, p0_val, q01_val, q10_val)
        
        # Sum probabilities where x > threshold
        high_mask = x_values > threshold_val
        prob_high = np.sum(px[high_mask])
        
        return float(prob_high)  # Ensure we return a scalar

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
        Set up the PyMC model with Metropolis sampling
        """
        with pm.Model() as self.model:
            # Priors
            p0 = pm.Uniform('p0', lower=0, upper=1)
            q01 = pm.Exponential('q01', lam=1)
            q10 = pm.Exponential('q10', lam=1)
            threshold = pm.Exponential('threshold', lam=1)
            
            # Transform bounded parameters
            q01_bounded = pm.Deterministic('q01_bounded', 
                                         1e-4 + (0.95 - 1e-4) * pm.math.sigmoid(q01))
            q10_bounded = pm.Deterministic('q10_bounded', 
                                         1e-4 + (0.95 - 1e-4) * pm.math.sigmoid(q10))
            threshold_bounded = pm.Deterministic('threshold_bounded',
                                              1e-4 + (0.2 - 1e-4) * pm.math.sigmoid(threshold))
            
            # Custom likelihood using our PyTorch models
            def likelihood_func(p0_value, q01_value, q10_value, threshold_value):
                total_ll = 0.0
                for t_idx, t in enumerate(self.times):
                    prob_hi = self.compute_prob_high(t, p0_value, q01_value, q10_value, threshold_value)
                    prob_hi = np.clip(prob_hi, 1e-10, 1 - 1e-10)  # Numerical stability
                    total_ll += (float(self.n_hi[t_idx]) * np.log(prob_hi) + 
                               float(self.n_lo[t_idx]) * np.log(1 - prob_hi))
                return total_ll
            
            # Add likelihood to model
            pm.Potential('likelihood', 
                        likelihood_func(p0, q01_bounded, q10_bounded, threshold_bounded))
            
    def run_inference(self, 
                     draws: int = 1000,
                     tune: int = 1000,
                     chains: int = None,
                     cores: int = None):
        """
        Run the MCMC inference using Metropolis sampling
        """
        if chains is None:
            chains = max(2, cpu_count() - 1)
        if cores is None:
            cores = chains
            
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=pm.Metropolis(),
                return_inferencedata=True
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

