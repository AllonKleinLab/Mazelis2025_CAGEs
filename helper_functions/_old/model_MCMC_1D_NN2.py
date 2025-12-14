import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import os
import state_simulations_1D_v1 as ss1D
from tqdm.auto import tqdm
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import random
import copy
import torch.nn.functional as F
from sklearn.model_selection import KFold
from multiprocessing import Pool
import pymc as pm
#import aesara.tensor as at
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time

class TransitionNetK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.n_bins = 2**k + 1
        
        # Define network architecture
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * self.n_bins)  # Remove Softmax from here
        )
    
    def forward(self, x):
        # Process input through network
        output = self.network(x)
        
        # Split output and apply Softmax separately to each part
        output_0 = F.softmax(output[:, :self.n_bins], dim=1)
        output_1 = F.softmax(output[:, self.n_bins:], dim=1)
        
        # Concatenate back together
        output = torch.cat([output_0, output_1], dim=1)
        
        if not hasattr(self, 'first_forward'):
            print("\nFirst forward pass:")
            print(f"Pre-softmax output shape: {output.shape}")
            print(f"Initial state 0 probabilities: {output[0,:self.n_bins]} (sum={output[0,:self.n_bins].sum():.4f})")
            print(f"Initial state 1 probabilities: {output[0,self.n_bins:]} (sum={output[0,self.n_bins:].sum():.4f})")
            self.first_forward = True
            
        return output

class SimulationDataset(Dataset):
    def __init__(self, sim_results, timepoints):
        self.inputs = []
        self.outputs = []
        
        print("\nSimulation Data Summary:")
        # Take first parameter set as example
        first_key = list(sim_results.keys())[0]
        q0_1, q1_0 = first_key
        results_init_0, results_init_1 = sim_results[first_key]
        
        print(f"\nFor parameters q0_1={q0_1:.2e}, q1_0={q1_0:.2e}:")
        for k in timepoints:
            n_bins = 2**k + 1
            print(f"\nk={k} divisions:")
            
            # For initial state 0
            counts_0 = np.zeros(n_bins)
            for i in range(n_bins):
                counts_0[i] = np.sum(results_init_0[k] == i/2**k)
            probs_0 = counts_0 / len(results_init_0[k])
            print(f"init_state=0 probabilities:")
            for i, p in enumerate(probs_0):
                if p > 0:  # Only show non-zero probabilities
                    print(f"  bin {i}/{2**k}: {p:.3f}")
            
            # For initial state 1
            counts_1 = np.zeros(n_bins)
            for i in range(n_bins):
                counts_1[i] = np.sum(results_init_1[k] == i/2**k)
            probs_1 = counts_1 / len(results_init_1[k])
            print(f"init_state=1 probabilities:")
            for i, p in enumerate(probs_1):
                if p > 0:  # Only show non-zero probabilities
                    print(f"  bin {i}/{2**k}: {p:.3f}")
        
        # Process all parameter sets for training
        for (q0_1, q1_0), (results_init_0, results_init_1) in sim_results.items():
            self.inputs.append([q0_1, q1_0])
            
            output_probs = []
            for k in timepoints:
                n_bins = 2**k + 1
                
                # For initial state 0
                counts_0 = np.zeros(n_bins)
                for i in range(n_bins):
                    counts_0[i] = np.sum(results_init_0[k] == i/2**k)
                probs_0 = counts_0 / len(results_init_0[k])
                output_probs.extend(probs_0)
                
                # For initial state 1
                counts_1 = np.zeros(n_bins)
                for i in range(n_bins):
                    counts_1[i] = np.sum(results_init_1[k] == i/2**k)
                probs_1 = counts_1 / len(results_init_1[k])
                output_probs.extend(probs_1)
            
            self.outputs.append(output_probs)
        
        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)
        self.inputs = torch.log10(self.inputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def simulate_trajectory(init_state, q0_1, q1_0, max_k):
    """Simulate a single trajectory."""
    trajectory = {0: init_state}
    current_state = init_state
    
    for k in range(1, max_k + 1):
        # Determine transition probability based on current state
        p = q0_1 if current_state == 0 else q1_0
        
        # Make transition decision
        if np.random.random() < p:
            current_state = 1 - current_state  # Flip state
            
        trajectory[k] = current_state
    
    return trajectory

def run_param_set(args):
    """Run simulations for a single parameter set."""
    params, n_sims, timepoints = args
    q0_1, q1_0 = params
    
    # Create parameter objects
    params_obj = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
    
    # Run simulations starting from state 0
    init_state_0 = {'state_0': 1.0, 'state_1': 0.0}
    sim_0 = ss1D.CellSimulation(params_obj, init_state_0)
    results_init_0 = {k: [] for k in timepoints}
    for _ in range(n_sims):
        for k in timepoints:
            results_init_0[k].append(sim_0.simulate_clone(k))
    
    # Run simulations starting from state 1
    init_state_1 = {'state_0': 0.0, 'state_1': 1.0}
    sim_1 = ss1D.CellSimulation(params_obj, init_state_1)
    results_init_1 = {k: [] for k in timepoints}
    for _ in range(n_sims):
        for k in timepoints:
            results_init_1[k].append(sim_1.simulate_clone(k))
    
    return (q0_1, q1_0), (results_init_0, results_init_1)

def analyze_surrogate_interpolation(model, sim_results, k=2, n_interp_points=100):
    """
    Analyze and visualize how well the surrogate model interpolates between training points.
    """
    # Get the parameter grid used for training
    q0_1_train = np.logspace(-4, -0.5, 40)  # Match training grid
    q1_0_train = np.logspace(-4, -0.5, 40)
    
    # Create finer grid for interpolation
    q0_1_interp = np.logspace(-4, -0.5, n_interp_points)
    q1_0_interp = np.logspace(-4, -0.5, n_interp_points)
    
    # Calculate statistics for simulation data
    sim_means = np.zeros((len(q0_1_train), len(q1_0_train)))
    sim_vars = np.zeros((len(q0_1_train), len(q1_0_train)))
    
    # Handle training data format
    k_data = sim_results[k]
    inputs = k_data['inputs']
    outputs = k_data['outputs']
    n_bins = k_data['n_bins']
    
    # Map training data to grid
    for idx in range(len(inputs)):
        q1_0, q0_1 = inputs[idx]
        i = np.abs(q0_1_train - q0_1).argmin()
        j = np.abs(q1_0_train - q1_0).argmin()
        
        # Get distribution for initial state 0
        dist = outputs[idx][:n_bins]
        bins = np.arange(n_bins)
        sim_means[i,j] = np.sum(dist * bins)
        sim_vars[i,j] = np.sum(dist * bins**2) - sim_means[i,j]**2
    
    # Calculate predictions from surrogate model
    model_means = np.zeros((n_interp_points, n_interp_points))
    model_vars = np.zeros((n_interp_points, n_interp_points))
    
    for i, q0_1 in enumerate(q0_1_interp):
        for j, q1_0 in enumerate(q1_0_interp):
            input_point = np.array([[q1_0, q0_1]])  # Shape (1, 2)
            pred_0, pred_1 = model.predict(input_point)  # Use predict method
            
            # Calculate statistics from predicted distribution
            bins = torch.arange(n_bins, dtype=torch.float32)
            model_means[i,j] = torch.sum(pred_0[0] * bins).item()  # [0] because pred_0 has batch dimension
            ex2 = torch.sum(pred_0[0] * bins**2).item()
            model_vars[i,j] = ex2 - model_means[i,j]**2
    
    # Add debug prints for simulation data
    print("\nChecking simulation data mapping:")
    test_j = len(q1_0_train)//2  # corresponds to q1_0≈4.9e-2
    print(f"For q1_0 = {q1_0_train[test_j]:.2e}:")
    print("q0_1 values vs means:")
    for i in range(len(q0_1_train)):
        print(f"q0_1 = {q0_1_train[i]:.2e}, mean = {sim_means[i,test_j]:.3f}")
    
    # Add debug prints for model predictions
    print("\nChecking model predictions:")
    print(f"For q1_0 = {q1_0_train[test_j]:.2e}:")
    test_inputs = np.array([[q1_0_train[test_j], q0_1] for q0_1 in q0_1_train])
    pred_0s, _ = model.predict(test_inputs)
    for i, q0_1 in enumerate(q0_1_train):
        bins = np.arange(model.n_bins)
        mean = np.sum(pred_0s[i].numpy() * bins)
        print(f"q0_1 = {q0_1:.2e}, predicted mean = {mean:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot means vs q0_1 for different q1_0
    ax = axes[0,0]
    for j in [0, len(q1_0_train)//4, len(q1_0_train)//2, 3*len(q1_0_train)//4]:
        q1_0_val = q1_0_train[j]
        ax.plot(q0_1_train, sim_means[:,j], 'o', label=f'Sim q1_0={q1_0_val:.1e}')
        ax.plot(q0_1_interp, model_means[:,j], '-', label=f'Model q1_0={q1_0_val:.1e}')
    ax.set_xscale('log')
    ax.set_xlabel('q0_1')
    ax.set_ylabel('Mean f1_hi')
    ax.legend()
    
    # Plot means vs q1_0 for different q0_1
    ax = axes[0,1]
    for i in [0, len(q0_1_train)//4, len(q0_1_train)//2, 3*len(q0_1_train)//4]:
        q0_1_val = q0_1_train[i]
        ax.plot(q1_0_train, sim_means[i,:], 'o', label=f'Sim q0_1={q0_1_val:.1e}')
        ax.plot(q1_0_interp, model_means[i,:], '-', label=f'Model q0_1={q0_1_val:.1e}')
    ax.set_xscale('log')
    ax.set_xlabel('q1_0')
    ax.set_ylabel('Mean f1_hi')
    ax.legend()
    
    # Similar plots for variance
    ax = axes[1,0]
    for j in [0, len(q1_0_train)//4, len(q1_0_train)//2, 3*len(q1_0_train)//4]:
        q1_0_val = q1_0_train[j]
        ax.plot(q0_1_train, sim_vars[:,j], 'o', label=f'Sim q1_0={q1_0_val:.1e}')
        ax.plot(q0_1_interp, model_vars[:,j], '-', label=f'Model q1_0={q1_0_val:.1e}')
    ax.set_xscale('log')
    ax.set_xlabel('q0_1')
    ax.set_ylabel('Variance f1_hi')
    ax.legend()
    
    ax = axes[1,1]
    for i in [0, len(q0_1_train)//4, len(q0_1_train)//2, 3*len(q0_1_train)//4]:
        q0_1_val = q0_1_train[i]
        ax.plot(q1_0_train, sim_vars[i,:], 'o', label=f'Sim q0_1={q0_1_val:.1e}')
        ax.plot(q1_0_interp, model_vars[i,:], '-', label=f'Model q0_1={q0_1_val:.1e}')
    ax.set_xscale('log')
    ax.set_xlabel('q1_0')
    ax.set_ylabel('Variance f1_hi')
    ax.legend()
    
    plt.tight_layout()
    return fig, (sim_means, sim_vars, model_means, model_vars)

class CellInferencePyMC:
    def __init__(self, data_df, trained_models, verbose=True):
        """
        Initialize inference with data and trained surrogate models.
        
        Args:
            data_df: DataFrame with counts at timepoints 0,2,4,6
            trained_models: Dictionary of trained neural networks for k
            verbose: Whether to print progress
        """
        self.data = data_df
        self.verbose = verbose
        self.timepoints = data_df.index.values  # these are directly the k values
        
        if self.verbose:
            print(f"Data timepoints (k values): {self.timepoints}")
        
        # Convert torch models to weight matrices and biases
        self.network_params = {}
        for k, model_dict in trained_models.items():
            model = model_dict['model']
            params = []
            for layer in model.network:
                if isinstance(layer, torch.nn.Linear):
                    w = layer.weight.detach().numpy()
                    b = layer.bias.detach().numpy()
                    params.append((w, b))
            self.network_params[k] = params
            
        if self.verbose:
            print(f"Loaded network parameters for k values: {list(self.network_params.keys())}")
            print(f"Will also fit k=0 using initial state proportion")
    
    def _forward_pass(self, x, k):
        """Implement neural network forward pass using PyMC operations."""
        # Get network parameters for this k
        params = self.network_params[k]
        
        # First layer
        h = x
        
        # Hidden layers
        for i, (w, b) in enumerate(params):
            h = pm.math.dot(h, w.T) + b
            # ReLU activation for all but last layer
            if i < len(params) - 1:
                h = pm.math.maximum(0, h)
        
        # Softmax for each half of the output separately
        n_bins = 2**k + 1
        logits_0 = h[:n_bins]
        logits_1 = h[n_bins:]
        
        probs_0 = pm.math.softmax(logits_0)
        probs_1 = pm.math.softmax(logits_1)
        
        return probs_0, probs_1
    
    def run_inference(self, n_draws=2000, tune=1000):
        """Run MCMC inference using PyMC."""
        
        with pm.Model() as model:
            # Priors
            log_q0_1 = pm.Normal('log_q0_1', mu=-5, sigma=2)
            log_q1_0 = pm.Normal('log_q1_0', mu=-5, sigma=2)
            init_0 = pm.Beta('init_0', alpha=2, beta=2)
            threshold = pm.Beta('threshold', alpha=2, beta=2)
            
            # Transform rates to natural scale
            q0_1 = pm.Deterministic('q0_1', pm.math.exp(log_q0_1))
            q1_0 = pm.Deterministic('q1_0', pm.math.exp(log_q1_0))
            
            # Get model predictions for each timepoint/k
            for k in self.timepoints:  # k is the same as timepoint
                if self.verbose:
                    print(f"Processing k={k}")
                    
                if k == 0:
                    # For k=0, prediction is just init_0
                    pred_prop = init_0
                else:
                    # Create input tensor for neural network
                    nn_input = pm.math.stack([q1_0, q0_1])
                    
                    # Get predictions from neural network
                    pred_0, pred_1 = self._forward_pass(nn_input, k)
                    
                    # Combine predictions based on initial state and threshold
                    n_bins = 2**k + 1
                    bins = np.arange(n_bins)  # Use numpy array
                    threshold_idx = threshold * (n_bins-1)
                    above_threshold = bins >= threshold_idx
                    
                    pred_prop = pm.math.sum(
                        init_0 * pred_0 * above_threshold +
                        (1-init_0) * pred_1 * above_threshold
                    )
                
                # Add likelihood for this timepoint/k
                n_cells = self.data.loc[k, 'f_hi'] + self.data.loc[k, 'f_lo']
                n_hi = self.data.loc[k, 'f_hi']
                pm.Binomial(f'obs_{k}', n=n_cells, p=pred_prop, observed=n_hi)
            
            # Run MCMC
            trace = pm.sample(
                draws=n_draws,
                tune=tune,
                return_inferencedata=True,
                progressbar=self.verbose
            )
        
        return trace

    def analyze_surrogate_interpolation(self, k=2, n_interp_points=100):
        """
        Analyze and visualize how well the surrogate model interpolates between training points.
        
        Args:
            k: Which division number to analyze
            n_interp_points: Number of interpolation points between training values
        """
        # Get the parameter grid used for training
        q0_1_train = np.logspace(-4, -0.5, 40)  # Match training grid
        q1_0_train = np.logspace(-4, -0.5, 40)
        
        # Create finer grid for interpolation
        q0_1_interp = np.logspace(-4, -0.5, n_interp_points)
        q1_0_interp = np.logspace(-4, -0.5, n_interp_points)
        
        # Calculate statistics for simulation data
        sim_means = np.zeros((len(q0_1_train), len(q1_0_train)))
        sim_vars = np.zeros((len(q0_1_train), len(q1_0_train)))
        
        for i, q0_1 in enumerate(q0_1_train):
            for j, q1_0 in enumerate(q1_0_train):
                if (q0_1, q1_0) in self.sim_results:
                    results_init_0, results_init_1 = self.sim_results[(q0_1, q1_0)]
                    outcomes = np.array(results_init_0[k])  # Using init_0 for now
                    sim_means[i,j] = np.mean(outcomes)
                    sim_vars[i,j] = np.var(outcomes)
        
        # Calculate predictions from surrogate model
        model_means = np.zeros((n_interp_points, n_interp_points))
        model_vars = np.zeros((n_interp_points, n_interp_points))
        
        for i, q0_1 in enumerate(q0_1_interp):
            for j, q1_0 in enumerate(q1_0_interp):
                nn_input = torch.tensor([q1_0, q0_1], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    pred_0, pred_1 = model(nn_input)
                    pred_0 = pred_0.squeeze(0)  # Remove batch dimension
                    
                # Calculate mean and variance for initial state 0
                bins = torch.arange(2**k + 1, dtype=torch.float32)
                model_means[i,j] = torch.sum(pred_0 * bins).item()
                ex2 = torch.sum(pred_0 * bins**2).item()
                model_vars[i,j] = ex2 - model_means[i,j]**2
        
        # Plot comparisons
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot means vs q0_1 for different q1_0
        ax = axes[0,0]
        for j in [0, len(q1_0_train)//4, len(q1_0_train)//2, 3*len(q1_0_train)//4]:
            q1_0_val = q1_0_train[j]
            ax.plot(q0_1_train, sim_means[:,j], 'o', label=f'Sim q1_0={q1_0_val:.1e}')
            ax.plot(q0_1_interp, model_means[:,j], '-', label=f'Model q1_0={q1_0_val:.1e}')
        ax.set_xscale('log')
        ax.set_xlabel('q0_1')
        ax.set_ylabel('Mean f1_hi')
        ax.legend()
        
        # Plot means vs q1_0 for different q0_1
        ax = axes[0,1]
        for i in [0, len(q0_1_train)//4, len(q0_1_train)//2, 3*len(q0_1_train)//4]:
            q0_1_val = q0_1_train[i]
            ax.plot(q1_0_train, sim_means[i,:], 'o', label=f'Sim q0_1={q0_1_val:.1e}')
            ax.plot(q1_0_interp, model_means[i,:], '-', label=f'Model q0_1={q0_1_val:.1e}')
        ax.set_xscale('log')
        ax.set_xlabel('q1_0')
        ax.set_ylabel('Mean f1_hi')
        ax.legend()
        
        # Similar plots for variance
        ax = axes[1,0]
        for j in [0, len(q1_0_train)//4, len(q1_0_train)//2, 3*len(q1_0_train)//4]:
            q1_0_val = q1_0_train[j]
            ax.plot(q0_1_train, sim_vars[:,j], 'o', label=f'Sim q1_0={q1_0_val:.1e}')
            ax.plot(q0_1_interp, model_vars[:,j], '-', label=f'Model q1_0={q1_0_val:.1e}')
        ax.set_xscale('log')
        ax.set_xlabel('q0_1')
        ax.set_ylabel('Variance f1_hi')
        ax.legend()
        
        ax = axes[1,1]
        for i in [0, len(q0_1_train)//4, len(q0_1_train)//2, 3*len(q0_1_train)//4]:
            q0_1_val = q0_1_train[i]
            ax.plot(q1_0_train, sim_vars[i,:], 'o', label=f'Sim q0_1={q0_1_val:.1e}')
            ax.plot(q1_0_interp, model_vars[i,:], '-', label=f'Model q0_1={q0_1_val:.1e}')
        ax.set_xscale('log')
        ax.set_xlabel('q1_0')
        ax.set_ylabel('Variance f1_hi')
        ax.legend()
        
        plt.tight_layout()
        return fig, (sim_means, sim_vars, model_means, model_vars)

def prepare_training_data(sim_results, timepoints):
    """
    Organize simulation results into training datasets for each k.
    """
    training_data = {}
    
    # Convert sim_results items to list for random sampling
    param_results = list(sim_results.items())
    
    # Randomly sample 5 examples to display
    sample_indices = np.random.choice(len(param_results), size=5, replace=False)
    
    # Skip k=0, process each k > 0
    for k in timepoints:
        if k == 0:
            continue
            
        print(f"\nProcessing k={k}")
        n_bins = 2**k + 1
        print(f"Number of bins per initial state: {n_bins}")
        
        inputs = []
        outputs = []
        
        # Process each parameter combination
        for i, ((q0_1, q1_0), (results_init_0, results_init_1)) in enumerate(param_results):
            inputs.append([q0_1, q1_0])
            
            n_sims = len(results_init_0[k])
            
            # For initial state 0
            counts_0 = np.zeros(n_bins)
            for value in results_init_0[k]:
                bin_index = int(value * 2**k)
                counts_0[bin_index] += 1
            probs_0 = counts_0 / n_sims
            
            # For initial state 1
            counts_1 = np.zeros(n_bins)
            for value in results_init_1[k]:
                bin_index = int(value * 2**k)
                counts_1[bin_index] += 1
            probs_1 = counts_1 / n_sims
            
            # Print details for randomly sampled examples
            if i in sample_indices:
                print(f"\nExample with q0_1={q0_1:.2e}, q1_0={q1_0:.2e}")
                print("Initial state 0 probabilities:")
                for j, p in enumerate(probs_0):
                    if p > 0.001:  # Only show non-negligible probabilities
                        print(f"  {j}/{2**k}: {p:.3f}")
                print("Initial state 1 probabilities:")
                for j, p in enumerate(probs_1):
                    if p > 0.001:  # Only show non-negligible probabilities
                        print(f"  {j}/{2**k}: {p:.3f}")
                print(f"Sum of probs (init 0): {np.sum(probs_0):.3f}")
                print(f"Sum of probs (init 1): {np.sum(probs_1):.3f}")
            
            combined_probs = np.concatenate([probs_0, probs_1])
            outputs.append(combined_probs)
        
        training_data[k] = {
            'inputs': np.array(inputs),
            'outputs': np.array(outputs),
            'n_bins': n_bins
        }
        
        print(f"\nFor k={k}:")
        print(f"Input shape: {training_data[k]['inputs'].shape}")
        print(f"Output shape: {training_data[k]['outputs'].shape}")
        
    return training_data

def prepare_training_data_gpr(sim_results, timepoints):
    """
    Prepare training data in a format suitable for GPR.
    
    Returns:
        Dictionary with keys for each k value, containing:
        - df: pandas DataFrame with columns:
            * q1_0, q0_1: input parameters
            * mean_0, var_0: statistics for initial state 0
            * mean_1, var_1: statistics for initial state 1
    """
    training_data = {}
    
    for k in timepoints:
        if k == 0:
            continue
            
        # Collect data for this k value
        data = []
        for params, results in sim_results.items():
            q0_1, q1_0 = params
            results_init_0, results_init_1 = results
            
            # Calculate statistics for initial state 0
            outcomes_0 = np.array(results_init_0[k])
            mean_0 = np.mean(outcomes_0)
            var_0 = np.var(outcomes_0)
            
            # Calculate statistics for initial state 1
            outcomes_1 = np.array(results_init_1[k])
            mean_1 = np.mean(outcomes_1)
            var_1 = np.var(outcomes_1)
            
            data.append({
                'q1_0': q1_0,
                'q0_1': q0_1,
                'mean_0': mean_0,
                'var_0': var_0,
                'mean_1': mean_1,
                'var_1': var_1
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Store in dictionary
        training_data[k] = {
            'df': df,
            'n_bins': 2**k + 1
        }
        
        print(f"\nFor k={k}:")
        print(f"Number of training points: {len(df)}")
        print("\nSample of training data:")
        print(df.head())
        print("\nSummary statistics:")
        print(df.describe())
    
    return training_data

class TransitionGPR:
    def __init__(self, k):
        self.k = k
        self.n_bins = 2**k + 1
        
        # Create GPRs for all but one bin for each initial state
        self.models_init_0 = []
        self.models_init_1 = []
        
        print("Initializing models...")
        t0 = time.time()
        for _ in range(self.n_bins - 1):
            # Simple kernel for log-space
            kernel = RBF(
                length_scale=[2.0, 2.0],
                length_scale_bounds=[(1.0, 5.0), (1.0, 5.0)]
            )
            
            self.models_init_0.append(GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=1,
                random_state=0,
                normalize_y=True,
                alpha=1e-4
            ))
            
            kernel = RBF(
                length_scale=[2.0, 2.0],
                length_scale_bounds=[(1.0, 5.0), (1.0, 5.0)]
            )
            
            self.models_init_1.append(GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=1,
                random_state=0,
                normalize_y=True,
                alpha=1e-4
            ))
        print(f"Initialization took {time.time() - t0:.2f} seconds")
    
    def train(self, inputs, outputs):
        """Train on log-transformed probabilities."""
        # Log transform both inputs and outputs
        log_inputs = np.log10(inputs)
        
        # Add small constant before log transform to handle zeros
        eps = 1e-10
        log_outputs = np.log10(outputs + eps)
        
        print("\nTraining GPR models...")
        for i in tqdm(range(self.n_bins - 1), desc="Training models for init state 0"):
            self.models_init_0[i].fit(log_inputs, log_outputs[:, i])
        
        for i in tqdm(range(self.n_bins - 1), desc="Training models for init state 1"):
            self.models_init_1[i].fit(log_inputs, log_outputs[:, i + self.n_bins])
    
    def predict(self, inputs):
        """Predict probabilities, transforming back from log space."""
        log_inputs = np.log10(inputs)
        
        # Get log predictions
        pred_0 = np.zeros((inputs.shape[0], self.n_bins))
        pred_1 = np.zeros((inputs.shape[0], self.n_bins))
        
        # Predict log probabilities for first n-1 bins
        for i in range(self.n_bins - 1):
            pred_0[:, i] = self.models_init_0[i].predict(log_inputs)
            pred_1[:, i] = self.models_init_1[i].predict(log_inputs)
        
        # Transform back from log space
        pred_0 = 10**pred_0
        pred_1 = 10**pred_1
        
        # Last bin probability from normalization
        pred_0[:, -1] = 1 - pred_0[:, :-1].sum(axis=1)
        pred_1[:, -1] = 1 - pred_1[:, :-1].sum(axis=1)
        
        # Ensure non-negative probabilities and normalize
        pred_0 = np.maximum(pred_0, 0)
        pred_1 = np.maximum(pred_1, 0)
        
        pred_0 /= pred_0.sum(axis=1, keepdims=True)
        pred_1 /= pred_1.sum(axis=1, keepdims=True)
        
        return torch.tensor(pred_0, dtype=torch.float32), torch.tensor(pred_1, dtype=torch.float32)

def train_k_model_gpr(k, training_data, **kwargs):
    """
    Train GPR model for specific k value.
    
    Args:
        k: Value of k to train for
        training_data: Dictionary containing training data
        **kwargs: Additional arguments (ignored for GPR)
    """
    print(f"\nTraining GPR model for k={k}")
    
    # Extract data for specific k value
    k_data = training_data[k]
    inputs = k_data['inputs']
    outputs = k_data['outputs']
    
    # Create and train model
    model = TransitionGPR(k)
    model.train(inputs, outputs)
    
    return model, None  # Return None for scores as we don't do cross-validation here

def compare_model_to_data(model, training_data, k, q01, q10):
    """
    Compare model predictions to training data for specific parameters.
    
    Args:
        model: Trained TransitionNetK model
        training_data: Dictionary containing training data
        k: Value of k to analyze
        q01, q10: Transition rates to analyze
    """
    import matplotlib.pyplot as plt
    
    # Get the training data for this k
    k_data = training_data[k]
    inputs = k_data['inputs']
    outputs = k_data['outputs']
    n_bins = 2**k + 1
    
    # Find the closest parameter set in the training data
    distances = np.sqrt((inputs[:,0] - q10)**2 + (inputs[:,1] - q01)**2)
    closest_idx = np.argmin(distances)
    actual_q10, actual_q01 = inputs[closest_idx]
    
    # Get model prediction for these parameters
    model_input = torch.tensor([[actual_q10, actual_q01]], dtype=torch.float32)
    with torch.no_grad():
        model_output = model(model_input).numpy()[0]
    
    # Get training data output
    data_output = outputs[closest_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for initial state 0
    mask_nonzero_0 = (data_output[:n_bins] > 0) & (model_output[:n_bins] > 0)
    ax1.scatter(data_output[:n_bins][mask_nonzero_0], 
               model_output[:n_bins][mask_nonzero_0],
               alpha=0.7)
    ax1.set_title(f'Initial State 0\nq10={actual_q10:.3e}, q01={actual_q01:.3e}')
    ax1.set_xlabel('Training Data Probability')
    ax1.set_ylabel('Model Prediction')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Add diagonal line
    lims_0 = [
        min(data_output[:n_bins][mask_nonzero_0].min(), 
            model_output[:n_bins][mask_nonzero_0].min()),
        max(data_output[:n_bins].max(), 
            model_output[:n_bins].max())
    ]
    ax1.plot(lims_0, lims_0, 'k--', alpha=0.5, zorder=0)
    
    # Plot for initial state 1
    mask_nonzero_1 = (data_output[n_bins:] > 0) & (model_output[n_bins:] > 0)
    ax2.scatter(data_output[n_bins:][mask_nonzero_1], 
               model_output[n_bins:][mask_nonzero_1],
               alpha=0.7)
    ax2.set_title(f'Initial State 1\nq10={actual_q10:.3e}, q01={actual_q01:.3e}')
    ax2.set_xlabel('Training Data Probability')
    ax2.set_ylabel('Model Prediction')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Add diagonal line
    lims_1 = [
        min(data_output[n_bins:][mask_nonzero_1].min(), 
            model_output[n_bins:][mask_nonzero_1].min()),
        max(data_output[n_bins:].max(), 
            model_output[n_bins:].max())
    ]
    ax2.plot(lims_1, lims_1, 'k--', alpha=0.5, zorder=0)
    
    # Add R² values
    def r2_score(y_true, y_pred):
        mask = (y_true > 0) & (y_pred > 0)
        if not np.any(mask):
            return 0
        y_true, y_pred = np.log10(y_true[mask]), np.log10(y_pred[mask])
        return np.corrcoef(y_true, y_pred)[0,1]**2
    
    r2_0 = r2_score(data_output[:n_bins], model_output[:n_bins])
    r2_1 = r2_score(data_output[n_bins:], model_output[n_bins:])
    
    ax1.text(0.05, 0.95, f'R² = {r2_0:.3f}', 
             transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    ax2.text(0.05, 0.95, f'R² = {r2_1:.3f}', 
             transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, (actual_q10, actual_q01)


