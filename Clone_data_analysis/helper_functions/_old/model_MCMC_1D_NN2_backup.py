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

class CellInferencePyMC:
    def __init__(self, data_df, sim_results, n_sims, n_cores=1, verbose=False):
        self.sim_results = sim_results
        self.n_sims = n_sims
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dictionary to store trained models for each k
        self.trained_models = {}
        
        # Create bins config, skipping k=0
        self.bins_config = {k: 2**k + 1 for k in data_df.index if k > 0}
    
    def train_all_models(self, n_folds=5, batch_size=32, epochs=100, 
                        learning_rate=1e-3, early_stop_patience=10):
        """Train models for all k values."""
        # Prepare training data
        training_data = prepare_training_data(self.sim_results, timepoints=self.bins_config.keys())
        
        # Train a model for each k
        for k in self.bins_config.keys():
            print(f"\nTraining model for k={k}")
            model, scores = train_k_model(
                k=k,
                training_data=training_data,
                n_folds=n_folds,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stop_patience=early_stop_patience
            )
            self.trained_models[k] = {
                'model': model,
                'cv_scores': scores
            }
            print(f"Model for k={k} trained and stored")
    
    def predict(self, k, init_state, q0_1, q1_0):
        """Make prediction using the model for specific k."""
        if k == 0:
            # Handle k=0 case deterministically
            return [1.0, 0.0] if init_state == 0 else [0.0, 1.0]
            
        if k not in self.trained_models:
            raise ValueError(f"No trained model for k={k}")
        
        model = self.trained_models[k]['model']
        inputs = torch.FloatTensor([[q0_1, q1_0]]).log10()
        
        with torch.no_grad():
            outputs = model(inputs)
            n_bins = 2**k + 1
            if init_state == 0:
                return outputs[0, :n_bins].numpy()
            else:
                return outputs[0, n_bins:].numpy()

    def debug_single_comparison(self):
        q0_1, q1_0 = 1e-4, 1e-4
        predicted_histograms = self.predict(q0_1, q1_0)

        params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
        results = ss1D.run_simulations(params, {'state_0': 1, 'state_1': 0}, self.n_sims, self.data_df.index)

        fig, axes = plt.subplots(2, len(self.bins_config), figsize=(12, 6))
        for idx, (time, n_bins) in enumerate(self.bins_config.items()):
            bins = np.linspace(0, 1, n_bins + 1)

            axes[0, idx].bar(bins[:-1], predicted_histograms[idx][:n_bins], width=1/n_bins, align="edge", alpha=0.5, label="Predicted")
            sim_hist, _ = np.histogram(results[time], bins=bins, density=True)
            axes[0, idx].bar(bins[:-1], sim_hist, width=1/n_bins, align="edge", alpha=0.5, label="Simulated")

            axes[0, idx].set_title(f"k={time}")
            axes[0, idx].legend()

        plt.tight_layout()
        return fig

    def compare_model_to_simulation(self, q0_1=None, q1_0=None, n_test_sims=10000):
        """
        Compare surrogate model predictions to direct simulations for a given parameter set.
        For k divisions, probabilities are discrete over n/2^k values and sum to 1.
        """
        # If no parameters provided, randomly select from grid
        if q0_1 is None or q1_0 is None:
            param_key = random.choice(list(self.sim_results.keys()))
            q0_1, q1_0 = param_key
        
        # Get model predictions
        predicted_histograms = self.predict(q0_1, q1_0)
        
        # Run new simulations
        params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
        results_init_0 = ss1D.run_simulations(
            params, {'state_0': 1.0, 'state_1': 0.0}, n_test_sims, list(self.bins_config.keys())
        )
        results_init_1 = ss1D.run_simulations(
            params, {'state_0': 0.0, 'state_1': 1.0}, n_test_sims, list(self.bins_config.keys())
        )
        
        model_probs = []
        sim_probs = []
        point_labels = []
        
        # Process each timepoint
        for time_idx, (time, n_bins) in enumerate(self.bins_config.items()):
            # For k divisions, we have 2^k + 1 bins (including edges)
            bins = np.linspace(0, 1, 2**time + 1)
            bin_centers = np.arange(2**time) / (2**time)  # Discrete values n/2^k
            
            # Get predicted probabilities from model
            pred_hist_init_0 = predicted_histograms[time_idx][:2**time]
            pred_hist_init_1 = predicted_histograms[time_idx][2**time:]
            
            # Get simulated counts and convert to probabilities
            sim_hist_init_0, _ = np.histogram(results_init_0[time], bins=bins)
            sim_hist_init_1, _ = np.histogram(results_init_1[time], bins=bins)
            
            # Convert to probabilities (should sum to 1)
            sim_hist_init_0 = sim_hist_init_0 / n_test_sims
            sim_hist_init_1 = sim_hist_init_1 / n_test_sims
            
            # Add points for init_state = 0
            for bin_idx, (pred, sim) in enumerate(zip(pred_hist_init_0, sim_hist_init_0)):
                model_probs.append(pred)
                sim_probs.append(sim)
                point_labels.append({
                    'divisions': time,
                    'init_state': 0,
                    'f1_hi': bin_centers[bin_idx]
                })
            
            # Add points for init_state = 1
            for bin_idx, (pred, sim) in enumerate(zip(pred_hist_init_1, sim_hist_init_1)):
                model_probs.append(pred)
                sim_probs.append(sim)
                point_labels.append({
                    'divisions': time,
                    'init_state': 1,
                    'f1_hi': bin_centers[bin_idx]
                })
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Convert to numpy arrays
        model_probs = np.array(model_probs)
        sim_probs = np.array(sim_probs)
        divisions = np.array([label['divisions'] for label in point_labels])
        init_states = np.array([label['init_state'] for label in point_labels])
        
        # Create scatter plot
        unique_divisions = sorted(set(divisions))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_divisions)))
        
        for div, color in zip(unique_divisions, colors):
            mask = divisions == div
            for init_state, marker in zip([0, 1], ['o', 's']):
                mask_combined = mask & (init_states == init_state)
                ax.scatter(sim_probs[mask_combined], model_probs[mask_combined],
                            c=[color], marker=marker, label=f'k={div}, init={init_state}',
                            alpha=0.6)
        
        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        # Customize plot
        ax.set_xlabel('Simulation Probability')
        ax.set_ylabel('Model Prediction')
        ax.set_title(f'Model vs Simulation\nq0_1={q0_1:.2e}, q1_0={q1_0:.2e}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Calculate and display R²
        r2 = np.corrcoef(sim_probs, model_probs)[0, 1]**2
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def predict_single(self, k, init_state, q0_1, q1_0):
        """
        Generate a single prediction for specific conditions.
        
        Args:
            k: Number of divisions
            init_state: Initial state (0 or 1)
            q0_1, q1_0: Transition parameters
        
        Returns:
            Array of probabilities for each possible state n/2^k
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")
        
        with torch.no_grad():
            inputs = torch.FloatTensor([[q0_1, q1_0]]).to(self.device)
            inputs = torch.log10(inputs)
            raw_outputs = self.model(inputs).cpu().numpy()[0]
            
            # Find the index for this k in our timepoints
            k_idx = list(self.bins_config.keys()).index(k)
            
            # Calculate start index for this k and init_state
            start_idx = 0
            for prev_k in list(self.bins_config.keys())[:k_idx]:
                start_idx += 2 * (2**prev_k)  # Account for both init states
            
            if init_state == 1:
                start_idx += 2**k
            
            # Extract probabilities for this specific k and init_state
            probs = raw_outputs[start_idx:start_idx + 2**k]
            
            return probs

    def diagnose_k0_predictions(self):
        """
        Diagnose k=0 predictions across different parameters and initial states.
        Also examines the training data structure for k=0 cases.
        """
        print("=== K=0 Model Diagnostics ===\n")
        
        # First, examine training data
        print("Training Data Structure for k=0:")
        dataset = SimulationDataset(self.sim_results, list(self.bins_config.keys()))
        
        # Look at a few training examples
        print("\nSample from training data (first 5 parameter sets):")
        for i in range(min(5, len(dataset))):
            inputs, outputs = dataset[i]
            q0_1 = 10**inputs[0].item()
            q1_0 = 10**inputs[1].item()
            k0_probs = outputs[:2]  # First two values should be k=0 probabilities
            print(f"\nParams q0_1={q0_1:.2e}, q1_0={q1_0:.2e}")
            print(f"k=0 training probs: {k0_probs}")
        
        print("\n=== Model Predictions ===")
        # Test various parameter values
        test_params = [
            (1e-6, 1e-6),
            (1e-4, 1e-4),
            (1e-2, 1e-2),
            (1e-6, 1e-2),
            (1e-2, 1e-6)
        ]
        
        for q0_1, q1_0 in test_params:
            print(f"\nParameters: q0_1={q0_1:.2e}, q1_0={q1_0:.2e}")
            
            # Test init_state = 0
            probs_0 = self.predict_single(0, 0, q0_1, q1_0)
            print(f"init_state=0: P(f1_hi = 0) = {probs_0[0]:.3f}, P(f1_hi = 1) = {probs_0[1]:.3f}")
            
            # Test init_state = 1
            probs_1 = self.predict_single(0, 1, q0_1, q1_0)
            print(f"init_state=1: P(f1_hi = 0) = {probs_1[0]:.3f}, P(f1_hi = 1) = {probs_1[1]:.3f}")
        
        # Examine model's raw outputs before any processing
        print("\n=== Raw Model Outputs ===")
        with torch.no_grad():
            for q0_1, q1_0 in test_params[:2]:  # Just look at first two parameter sets
                inputs = torch.FloatTensor([[q0_1, q1_0]]).to(self.device)
                inputs = torch.log10(inputs)
                raw_outputs = self.model(inputs).cpu().numpy()[0]
                print(f"\nRaw outputs for q0_1={q0_1:.2e}, q1_0={q1_0:.2e}:")
                print(f"First few values: {raw_outputs[:4]}")
        
        return None

    @staticmethod
    def generate_simulation_grid(n_sims, timepoints, q0_1_values, q1_0_values, 
                               cache_file=None, n_cores=1, verbose=False):
        """
        Generate simulation results for a grid of parameter values.
        """
        # Check if cache exists
        if cache_file and os.path.exists(cache_file):
            if verbose:
                print(f"Loading cached results from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if verbose:
            print(f"Generating simulation grid for {len(q0_1_values)}x{len(q1_0_values)} parameters")
            print(f"Using {n_cores} CPU cores")
        
        # Create parameter grid with additional arguments
        param_grid = [
            ((q0_1, q1_0), n_sims, timepoints)  # Pack all arguments together
            for q0_1 in q0_1_values
            for q1_0 in q1_0_values
        ]
        
        # Run simulations in parallel
        with Pool(n_cores) as pool:
            results = dict(pool.map(run_param_set, param_grid))
        
        # Cache results if requested
        if cache_file:
            if verbose:
                print(f"Caching results to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        
        return results

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

def train_k_model(k, training_data, batch_size=32, epochs=100, 
                 learning_rate=1e-3, early_stop_patience=10, val_fraction=0.2):
    """Train model for specific k value using systematic sampling for validation."""
    
    print("\n=== Starting Training ===")
    print(f"Training data keys: {training_data.keys()}")
    print(f"Training data[{k}] keys: {training_data[k].keys()}")
    
    # Extract data for specific k value
    k_data = training_data[k]
    inputs = k_data['inputs']
    outputs = k_data['outputs']
    
    # Verify probability structure
    n_bins = 2**k + 1
    print(f"\nVerifying probability structure (k={k}, n_bins={n_bins}):")
    print(f"First training example:")
    print(f"Input (q_10, q_01): {inputs[0]}")
    print(f"Output probabilities:")
    print(f"  Initial state 0: {outputs[0,:n_bins]} (sum={outputs[0,:n_bins].sum():.4f})")
    print(f"  Initial state 1: {outputs[0,n_bins:]} (sum={outputs[0,n_bins:].sum():.4f})")
    
    # Convert inputs and outputs to tensors
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs, dtype=torch.float32)
    
    # Systematic sampling for validation
    n_samples = len(inputs)
    val_stride = int(1 / val_fraction)  # e.g., 5 for 20% validation
    val_idx = torch.arange(0, n_samples, val_stride)
    train_idx = torch.tensor([i for i in range(n_samples) if i % val_stride != 0])
    
    # Split the data
    train_inputs = inputs[train_idx]
    train_outputs = outputs[train_idx]
    val_inputs = inputs[val_idx]
    val_outputs = outputs[val_idx]
    
    print(f"\nDataset sizes:")
    print(f"Total samples: {n_samples}")
    print(f"Validation stride: every {val_stride}th sample")
    print(f"Training: {len(train_inputs)} samples")
    print(f"Validation: {len(val_inputs)} samples")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_inputs, train_outputs)
    val_dataset = TensorDataset(val_inputs, val_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = TransitionNetK(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    def compute_loss(pred, target):
        n_bins = 2**k + 1
        pred_0, pred_1 = pred[:, :n_bins], pred[:, n_bins:]
        target_0, target_1 = target[:, :n_bins], target[:, n_bins:]
        
        # Debug first batch
        if not hasattr(compute_loss, 'first_call'):
            print("\nFirst loss computation:")
            print(f"Predictions (first example):")
            print(f"  Initial state 0: {pred_0[0]} (sum={pred_0[0].sum():.4f})")
            print(f"  Initial state 1: {pred_1[0]} (sum={pred_1[0].sum():.4f})")
            print(f"Targets (first example):")
            print(f"  Initial state 0: {target_0[0]} (sum={target_0[0].sum():.4f})")
            print(f"  Initial state 1: {target_1[0]} (sum={target_1[0].sum():.4f})")
            compute_loss.first_call = True
        
        loss_0 = criterion(pred_0.log(), target_0)
        loss_1 = criterion(pred_1.log(), target_1)
        return (loss_0 + loss_1) / 2
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_fold_model = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        n_batches = 0
        
        for batch_inputs, batch_targets in train_loader:
            if epoch == 0 and n_batches == 0:
                print("\nFirst batch processing:")
                outputs = model(batch_inputs)
                print(f"Model output (first example):")
                print(f"  Initial state 0: {outputs[0,:n_bins]} (sum={outputs[0,:n_bins].sum():.4f})")
                print(f"  Initial state 1: {outputs[0,n_bins:]} (sum={outputs[0,n_bins:].sum():.4f})")
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = compute_loss(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                outputs = model(batch_inputs)
                val_loss += compute_loss(outputs, batch_targets).item()
                n_val_batches += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss/n_batches:.4f}, val_loss={val_loss/n_val_batches:.4f}")
    
    best_model_idx = np.argmin(val_loss)
    best_model = model
    
    return best_model, val_loss

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
