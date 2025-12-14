#import os
#os.environ["PYTHONPATH"] = os.getcwd()

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import cpu_count
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

# 🔹 Force PyTensor to use float64 globally
pytensor.config.floatX = "float64"

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
        params = pt.as_tensor_variable(params).astype("float64")
        p0 = pt.as_tensor_variable(p0).astype("float64")
        return Apply(self, [params, p0], [params.type()])

    def perform(self, node, inputs, outputs):
        # Convert inputs to float64
        params = inputs[0].reshape(1, 2).astype(np.float64)
        p0 = float(inputs[1])  # Ensure float64
        y = self.torch_model.predict(params, q0=p0)
        # Ensure output is float64
        outputs[0][0] = np.array(y[0], dtype=np.float64)

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
        
        # Define reasonable starting values (log_S ~ log(0.3) to prevent sticking at zero)
        self.init_vals = {
            "log_S": np.log(0.3),
            "log_r": np.log(1.0),
            "p0": 0.5,
            "threshold": 1e-3,
            "a": -1.5,
            "sigma": 0.5,
        }
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
        
        # Convert scores to tensors for each timepoint
        self.tensor_data = {}
        for t in self.times:
            self.tensor_data[t] = pt.as_tensor_variable(
                data[data['timepoint'] == t]['score'].values.astype(np.float64)
            )

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
            return np.array([0, 1], dtype=np.float64), np.array([p0, 1-p0], dtype=np.float64)
        
        if t not in self.surrogate_models:
            raise ValueError(f"No surrogate model available for t={t}")
            
        # Create parameter array for (q01, q10)
        params = np.array([[q01, q10]], dtype=np.float64)
        
        # Get probabilities from surrogate model
        probs = self.surrogate_models[t].predict(params, q0=p0)
        
        # Generate corresponding x values: 0, 1/2^t, 2/2^t, ..., 1
        n_points = 2**t + 1
        x_values = np.linspace(0, 1, n_points, dtype=np.float64)
        
        return x_values, np.array(probs[0], dtype=np.float64)

    def compute_log_likelihood(self, params, p0, a, sigma, threshold, s_min, bb):
        """
        Compute log likelihood for observations at time t
        """
        logp = 0

        # Loop over each timepoint
        for t in self.times:
            # Get tensor data for this timepoint
            x_t = self.tensor_data[t]
            
            # Get surrogate model predictions P(n,t)
            if t == 0:
                px = pt.stack([p0, 1-p0])
                n_values = np.array([0, 1], dtype=np.float64)
            else:
                n_values = np.linspace(0, 1, 2**t + 1, dtype=np.float64)
                px = self.surrogate_ops[t](params, p0).astype("float64")
                
            # Create masks for zero and non-zero states
            zero_mask = n_values == 0
            nonzero_mask = ~zero_mask
            
            # Split computation based on threshold
            below_thresh = pt.lt(x_t, threshold)
            above_thresh = pt.ge(x_t, threshold)
            
            # For x < threshold, use P(n=0)
            logp += pt.sum(below_thresh) * pt.log(px[0])
            
            # For x ≥ threshold, compute all log-normal probabilities at once
            x_above = x_t[pt.nonzero(above_thresh)[0]]
            
            # Get n values for non-zero states using mask
            n_nonzero = n_values[nonzero_mask]
            px_nonzero = px[nonzero_mask]
            
            # Compute log-normal probabilities
            #mu = pt.log(n_nonzero).reshape((-1, 1)) + a
            s2_nt = pt.log(1+(pt.exp(sigma**2)-1)/(n_nonzero.reshape((-1, 1))* pt.pow(2,t))) + s_min**2
            mu_nt = (bb*pt.log(n_nonzero)).reshape((-1, 1))  + (bb-1)*pt.log(2**t) + a + (sigma**2-s2_nt)/2
            lognormal_logprobs = -((pt.log(x_above) - mu_nt)**2)/(2*s2_nt) - pt.log(x_above) - 0.5*pt.log(s2_nt) 
            
            # Sum over n for each x using masked probabilities
            # tst1 = lognormal_logprobs.T
            # tst2 = pt.exp(tst1)
            # print("Type of tst1:", type(tst2))
            # print("Is tst2 a PyTensor variable?", isinstance(tst2, pt.Variable))
            # print("Type of lognormal_logprobs:", type(lognormal_logprobs))
            # print("Is lognormal_logprobs a PyTensor variable?", isinstance(lognormal_logprobs, pt.Variable))
            # print("Type of px:", type(px))
            # print("Is px a PyTensor variable?", isinstance(px, pt.Variable))
            # tst3 = tst2*px_nonzero
            # tst4 = pt.sum(tst3, axis=1)
            # tst5 = pt.log(tst4)
            #logp += pt.sum(tst5)

            logp += pt.sum(pt.log(pt.sum(pt.exp(lognormal_logprobs.T) * px_nonzero, axis=1))).astype("float64")
        
        return logp

    def setup_model(self):
        """
        Set up PyMC model with the new parameterization using r and S
        """
        with pm.Model() as self.model:
            # Priors - using log transform for LogUniform
            log_S = pm.Uniform('log_S', lower=np.log(1e-3), upper=np.log(1e1), 
                               dtype="float64", initval=np.log(1e-1))
            log_r = pm.Uniform('log_r', lower=np.log(1e-2), upper=np.log(1e2), 
                               dtype="float64", initval=0)
            S = pm.Deterministic('S', pm.math.exp(log_S))
            r = pm.Deterministic('r', pm.math.exp(log_r))
            
            p0 = pm.Uniform('p0', lower=0, upper=1, 
                            dtype="float64", initval=0.5)
            threshold = pm.Uniform('threshold', lower=5e-4, upper=5e-3, 
                                   dtype="float64", initval=1e-3)


            # New parameters for modeling full NMF scores
            a0, sigma0 = self.estimate_lognormal_params()
            a = pm.Normal('a', mu=a0-np.log(3), sigma=sigma0, 
                          dtype="float64", initval=a0-np.log(3))  # Location parameter for log-normal
            sigma = pm.HalfNormal('sigma', sigma=sigma0/2, 
                                  dtype="float64", initval=sigma0/2)  # Noise parameter for log-normal that drops with 1/N
            s_min = pm.HalfNormal('s_min', sigma=0.05, 
                                      dtype="float64", initval=0.05) # Noise parameter for log-normal that is N-independent
            b = pm.HalfNormal('b', sigma=1, 
                             dtype="float64", initval=0.5)
            bb = pm.Deterministic('bb', b + 1) # Non-linearity converting n to x, value of 1 is linear
            
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
            logp = self.compute_log_likelihood(params, p0, a, sigma, threshold, s_min, bb)

            # Add the logp to the model
            pm.Potential("likelihood", logp)

    def run_inference(self, draws, tune, chains, cores, show_progressbar=True, compute_convergence_checks=True):
        """
        Run MCMC inference with refined step sizes and more chains
        """
        with self.model:
            # Metropolis steps with tune_interval specified
            step = pm.CompoundStep([
                pm.Metropolis(vars=[self.model.p0], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.log_r], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.log_S], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.threshold], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.a], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.sigma], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.s_min], scaling=0.5, tune_interval=100),
                pm.Metropolis(vars=[self.model.b], scaling=0.5, tune_interval=100),
            ])
            
            print("About to start sampling")
        
            # def simple_callback(trace, draw):
            #     print("In callback, draw:", draw)

            # # Try to get initial values
            # for var in self.model.named_vars:
            #     try:
            #         var_obj = self.model.named_vars[var]
            #         if hasattr(var_obj, 'eval'):
            #             print(f"Initial value for {var}:", var_obj.eval())
            #     except Exception as e:
            #         print(f"Couldn't get value for {var}:", e)
            

            print("Calling pm.sample")
            # Run inference with more chains and tuning
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step,
                #initvals=[self.init_vals] * chains,  # Use same init values for all 4 chains
                progressbar=show_progressbar,  # Disable progress bar
                compute_convergence_checks=compute_convergence_checks # Disable convergence warnings
            )

    def create_mock_data(self, 
                        r01: float = 0.2,
                        r10: float = 0.1,
                        p0: float = 0.5,
                        threshold: float = 1e-3,
                        a: float = -1.5,
                        sigma: float = 0.5,
                        bb: float = 1.0,  # New parameter for nonlinear transformation
                        s_min: float = 0.1,  # New parameter for minimum noise
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
            bb: Non-linearity converting n to x (default 1.0)
            s_min: Minimum noise (default 0.1)
            
        Returns:
            DataFrame with columns ['timepoint', 'score']
        """
        # Calculate S and r from rates
        S = np.float64(r01 + r10)
        r = np.float64(r10 / r01)
        
        # Calculate q01, q10
        exp_term = 1 - np.exp(-S)
        q01 = np.float64(exp_term / (1 + r))
        q10 = np.float64(exp_term / (1 + 1/r))
        
        # Initialize data storage
        all_data = []
        times = [0, 2, 4, 6]
        
        for t in times:
            # Get probability distribution P(n) from surrogate model
            if t == 0:
                px = np.array([p0, 1-p0], dtype=np.float64)
                n_values = np.array([0, 1], dtype=np.float64)
            else:
                n_values = np.linspace(0, 1, 2**t + 1, dtype=np.float64)
                params = np.array([[q01, q10]], dtype=np.float64)
                px = np.array(self.surrogate_ops[t].torch_model.predict(params, q0=p0)[0], dtype=np.float64)
                px[0] = 1-np.sum(px[1:])
            
            # Generate "n" values for each colony
            n_samples = np.random.choice(n_values, size=n_colonies, p=px).astype("float64")
            
            # Generate scores
            scores = np.zeros(n_colonies, dtype="float64")
            zero_mask = n_samples == 0
            nonzero_mask = ~zero_mask

            if np.any(zero_mask):
                # For n=0, generate values from 0 to threshold
                scores[zero_mask] = np.random.uniform(0, threshold, size=np.sum(zero_mask))

            if np.any(nonzero_mask):
                # Updated calculations for non-zero values
                s_nt = np.sqrt(np.log(1 + (np.exp(est_params['sigma']**2)-1)/(n_values[:, None]*2**t)) + est_params['s_min']**2)
                mu_nt = (np.log(n_values[:, None]*2**t)* est_params['bb']) - pt.log(2**t) + est_params['a'] + (est_params['sigma']**2 - s_nt**2) / 2
                lognormal_samples = np.random.lognormal(
                    mean=mu_nt,
                    sigma=s_nt
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

    def plot_diagnostics(self, figsize=(6, 10), figsize_pair=(7, 7)):
        """
        Plot MCMC diagnostics including traces, posterior distributions, and pair plots
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
        
        # Get summary statistics
        summary = az.summary(self.trace, round_to=6)
        
        # Print parameter estimates
        print("\nParameter Estimates:")
        print("-" * 50)
        for param in ['p0', 'S', 'r', 'threshold', 'a', 'sigma', 'bb', 's_min']:
            est_mean = summary.loc[param, 'mean']
            est_hdi = (summary.loc[param, 'hdi_3%'], summary.loc[param, 'hdi_97%'])
            rhat = summary.loc[param, 'r_hat']
            print(f"{param}: {est_mean:.4f} ± {est_hdi[1] - est_hdi[0]:.4f} (R-hat: {rhat:.3f})")
        
        # Create plots
        params = ['p0', 'S', 'r', 'threshold', 'a', 'sigma', 'bb', 's_min']
        
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

        try:
            # After the trace plots, add pair plots with reduced parameter set
            plt.figure(figsize=figsize_pair)
            ax = az.plot_pair(
                self.trace, 
                var_names=['S', 'r', 'p0', 'a', 'threshold'],  # Reduced set of parameters
                kind='scatter',  # Changed from 'kde' to 'scatter'
                marginals=True,
                textsize=10,
                figsize=figsize_pair,
                scatter_kwargs={'alpha': 0.1, 's': 1},
            )
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not create pair plot due to error: {str(e)}")

    def get_map_estimate(self):
        """
        Get parameter estimates using posterior mean
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
        
        print("Using posterior mean as estimate")
        summary = az.summary(self.trace, round_to=6)
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

    def plot_predictions(self, figsize=(10, 5)):
        """
        Plot model predictions against data in multiple panels
        """
        if not hasattr(self, 'trace'):
            raise ValueError("No trace found. Run inference first.")
            
        # Get parameter estimates from trace with new parameters
        summary = az.summary(self.trace, round_to=6)
        est_params = {
            'p0': float(summary.loc['p0', 'mean']),
            'q01': float(summary.loc['q01', 'mean']),
            'q10': float(summary.loc['q10', 'mean']),
            'threshold': float(summary.loc['threshold', 'mean']),
            'a': float(summary.loc['a', 'mean']),
            'sigma': float(summary.loc['sigma', 'mean']),
            'bb': float(summary.loc['bb', 'mean']),
            's_min': float(summary.loc['s_min', 'mean'])
        }
        
        # Create figure with 2x4 subplots
        fig, axs = plt.subplots(2, 4, figsize=figsize)
        
        # Store data for summary plots
        empirical_nonzero = []
        predicted_nonzero = []
        empirical_means = []
        predicted_means = []
        empirical_sems = []
        predicted_E_pn = []
        
        x = self.data.score.values
        t_vals = self.data.timepoint.values

        # Plot histograms for each timepoint (t=0,2,4,6)
        for i, t in enumerate(self.times):
            ax = axs[0, i]  # First row, column i
            # Get data for this timepoint
            x_t = x[t_vals == t]
            

            # Generate predicted distribution
            if t == 0:
                px = np.array([est_params['p0'], 1-est_params['p0']])
                n_values = np.array([0, 1])
            else:
                n_values = np.linspace(0, 1, 2**t + 1)
                params = np.array([[est_params['q01'], est_params['q10']]])
                px = self.surrogate_ops[t].torch_model.predict(params, q0=est_params['p0'])[0]
            
            # Store fraction non-zero
            nonzero_mask = (x_t > est_params['threshold'])
            empirical_nonzero.append(np.mean(nonzero_mask))
            predicted_nonzero.append(np.sum(px[1:]))  # Sum of all n>0 probabilities
                        
            # Expectation on 
            E_pn_t = np.sum(n_values[1:] * px[1:])/(1-px[0])
            predicted_E_pn.append(E_pn_t)
            #print(t,E_pn_t, np.sum(px), np.sum(px[1:]))

            # Store non-zero means
            empirical_means.append(np.mean(x_t[nonzero_mask]))
            empirical_sems.append(np.std(x_t[nonzero_mask])/np.sqrt(len(x_t[nonzero_mask])))
            predicted_mean = 0  # Add contribution from each n
            for n, p_n in zip(n_values, px):
                if n == 0:
                    #predicted_mean += p_n * est_params['threshold']
                    continue
                else:
                    # Expected value of lognormal with given n
                    # Calculate n,t moments
                    s_nt = np.sqrt(np.log(1 + (np.exp(est_params['sigma']**2)-1)/(n*2**t)) + est_params['s_min']**2)
                    mu_nt = (np.log(n * 2**t)*est_params['bb']) - np.log(2**t) + est_params['a'] + (est_params['sigma']**2 - s_nt**2) / 2
                    predicted_mean += p_n * np.exp(mu_nt + s_nt**2/2)/(1-px[0])
            predicted_means.append(predicted_mean)
            

            # Plot empirical distribution
            hist_vals, _, _ = ax.hist(np.log10(est_params['threshold'] + x_t), 
                    bins=30, density=True, alpha=0.5, color='green', 
                    label='Data')

            # Generate points for predicted distribution
            x_plot = np.logspace(np.log10(est_params['threshold']), 
                               np.log10(est_params['threshold'] + 1), 100)
            y_plot = np.zeros_like(x_plot)
            
            # Add contribution from each n>0
            for n, p_n in zip(n_values[1:], px[1:]):
                s_nt = np.sqrt(np.log(1 + (np.exp(est_params['sigma']**2)-1)/(n*2**t)) + est_params['s_min']**2)
                mu_nt = (np.log(n*2**t)*est_params['bb']) - np.log(2**t) + est_params['a'] + (est_params['sigma']**2 - s_nt**2) / 2
                y_plot += p_n * np.exp(-(np.log(x_plot) - mu_nt)**2 / (2*s_nt**2)) / (s_nt*np.sqrt(2*np.pi))
            #y_plot[0] = px[0] # add zero-values
            scaling_factor = np.max(hist_vals[5:]) / np.max(y_plot[x_plot > est_params['threshold']])
            y_plot *= scaling_factor
            ax.plot(np.log10(x_plot), y_plot, 'k-', label='Model')
            ax.set_title(f't = {t}')
            ax.set_xlabel('log10(threshold + score)')
            ax.set_ylabel('Density')
            if i == 0:
                ax.legend()
        
        # Plot fraction non-zero
        ax = axs[1, 1]  # Second row
        ax.plot(self.times, empirical_nonzero, 'go', label='Data')
        ax.plot(self.times, predicted_nonzero, 'k-', label='Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction non-zero')
        ax.set_ylim(0, 1)
        ax.legend()
        
        # Plot mean values
        ax = axs[1, 2]  # Second row
        ax.errorbar(self.times, empirical_means, yerr=empirical_sems, fmt='go', label='Data', capsize=5)
        ax.plot(self.times, predicted_means, 'k-', label='Model')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean score (non-zero clones)')
        ax.legend()
        
        
        # Remove empty plots
        axs[1, 0].remove()
        #axs[1, 3].remove()
        
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
        print("a_init:", a_init)
        sigma_init = np.std(log_scores)
        print("sigma_init:", sigma_init)

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
    summary = az.summary(mcmc.trace, round_to=6)
    
    # Initialize row data
    row_data = {
        'program_id': program_id,
        'condition': cond,
    }
    
    # Parameters to process
    params = ['r01', 'r10', 'q01', 'q10', 'S', 'r', 'p0', 'threshold', 'a', 'sigma', 'bb', 's_min']
    
    # Get high precision means directly from trace
    for param in params:
        # Mean with full precision
        row_data[f'{param}_mean'] = float(mcmc.trace.posterior[param].mean())
        # Other stats from summary
        row_data[f'{param}_std'] = float(summary.loc[param, 'sd'])
        row_data[f'{param}_hdi_low'] = float(summary.loc[param, 'hdi_3%'])
        row_data[f'{param}_hdi_high'] = float(summary.loc[param, 'hdi_97%'])
    
    return row_data