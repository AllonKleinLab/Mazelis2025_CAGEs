import numpy as np
import pymc as pm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import state_simulations_1D_v1 as ss1D
from dataclasses import dataclass
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
import copy


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


class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()  # Add sigmoid to ensure outputs are between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x)


class SimulationDataset(Dataset):
    def __init__(self, param_grid, sim_results):
        self.inputs = []
        self.outputs = []
        
        for (q0_1, q1_0), (results_init_0, results_init_1) in sim_results.items():
            self.inputs.append([q0_1, q1_0])
            self.outputs.append(results_init_0)  # Just use results_init_0
            
        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)
        
        # Log transform inputs
        self.inputs = torch.log10(self.inputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def _run_simulation_for_params(params, data_df, n_sims):
    """Helper function to run simulations for a single parameter set."""
    q0_1, q1_0 = params
    trans_params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
    
    # Run simulations for both initial conditions
    results_0 = ss1D.run_simulations(
        trans_params,
        {'state_0': 1, 'state_1': 0},
        n_sims,
        list(data_df.index)
    )
    
    results_1 = ss1D.run_simulations(
        trans_params,
        {'state_0': 0, 'state_1': 1},
        n_sims,
        list(data_df.index)
    )
    
    return (q0_1, q1_0), (results_0, results_1)


class CellInferencePyMC:
    def __init__(self, data_df, n_sims=1000, threshold_bounds=(1e-3, 1e-1),
                 verbose=True, n_cores=8, cache_file='simulation_grid_cache.pkl', n_grid=30):
        """Initialize the inference model."""
        # First set all instance attributes
        self.data_df = data_df
        self.n_sims = n_sims
        self.threshold_bounds = threshold_bounds
        self.verbose = verbose
        self.n_cores = min(n_cores, mp.cpu_count() - 1)
        self.cache_file = cache_file
        self.n_grid = n_grid
        self.model = None  # Neural network model will be initialized later

        # Set up CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.verbose:
            print("Initializing CellInferencePyMC...")
            print(f"Using device: {self.device}")
            print(f"Number of simulations per parameter set: {n_sims}")
            print(f"Grid size: {n_grid}x{n_grid} = {n_grid**2} parameter combinations")
            print(f"Threshold bounds: {threshold_bounds}")
            print(f"Using {self.n_cores} CPU cores")
            print(f"Cache file: {cache_file}")
        
        self._setup_simulation_grid(n_grid=self.n_grid)

    def _setup_simulation_grid(self, n_grid=30):
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
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
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
                print(f"Attempting to save cache to: {os.path.abspath(self.cache_file)}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            if self.verbose:
                print("Successfully saved simulation grid to cache.")
                print(f"Cache file size: {os.path.getsize(self.cache_file) / 1024 / 1024:.1f} MB")
        except Exception as e:
            if self.verbose:
                print(f"Error saving cache: {e}")
                print(f"Full error details: {repr(e)}")

        if self.verbose:
            print("Pre-computation complete!")

    def train_surrogate(self, val_split=0.2, batch_size=32, epochs=100, 
                       learning_rate=1e-3, early_stop_patience=10, min_delta=1e-6):
        """Train neural network on simulation grid with regularization"""
        if self.verbose:
            print("\nPreparing to train surrogate model...")
            print(f"Training parameters:")
            print(f"  Validation split: {val_split}")
            print(f"  Batch size: {batch_size}")
            print(f"  Epochs: {epochs}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Early stopping patience: {early_stop_patience}")
            print(f"  Min delta: {min_delta}")
        
        # Create dataset with noise
        dataset = SimulationDataset(self.param_grid, self.sim_results)
        
        # Split into train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model with dropout
        self.model = TransitionNet().to(self.device)
        
        # Use L2 regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        if self.verbose:
            print("\nStarting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': [], 'val_max_error': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            max_error = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    max_error = max(max_error, torch.max(torch.abs(outputs - targets)).item())
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Store history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_max_error'].append(max_error)
            
            if epoch % 10 == 9:  # Print every 10 epochs
                if self.verbose:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {avg_val_loss:.6f}, '
                          f'Max Error: {max_error:.6f}')
            
            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        
        if self.verbose:
            print("\nTraining complete!")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Final maximum error: {training_history['val_max_error'][-1]:.6f}")
        
        return training_history
    
    def save_model(self):
        """Save trained model"""
        os.makedirs('./cache', exist_ok=True)
        torch.save(self.model.state_dict(), './cache/surrogate_model.pt')
    
    def load_model(self):
        """Load trained model"""
        self.model = TransitionNet().to(self.device)
        self.model.load_state_dict(torch.load('./cache/surrogate_model.pt'))
        self.model.eval()
    
    def predict(self, q0_1, q1_0):
        """Use trained model to predict proportions"""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            inputs = torch.FloatTensor([[q0_1, q1_0]]).to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()[0]

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

    def run_inference(self, n_draws=2000, tune=200):
        """Run MCMC inference using the trained surrogate model"""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Neural network surrogate model must be trained before running inference")
        
        if self.verbose:
            print("\nBuilding PyMC model...")
        
        with pm.Model() as model:
            # Use log-uniform priors for transition rates
            log_q0_1 = pm.Uniform('log_q0_1', -7, -2)
            log_q1_0 = pm.Uniform('log_q1_0', -7, -2)
            
            # Transform to actual rates
            q0_1 = pm.Deterministic('q0_1', 10**log_q0_1)
            q1_0 = pm.Deterministic('q1_0', 10**log_q1_0)
            
            # Prior for initial state proportion
            init_0 = pm.Beta('init_0', alpha=2, beta=2)
            
            def likelihood(q0_1, q1_0, init_0):
                # Prepare input for neural network
                inputs = torch.tensor([[q0_1, q1_0]], dtype=torch.float32)
                inputs = torch.log10(inputs)  # Log transform like in training
                
                # Get predictions from surrogate model
                self.model.eval()
                with torch.no_grad():
                    pred_props = self.model(inputs).numpy()[0]
                
                # Calculate expected counts
                total_cells = self.data_df['f_hi'] + self.data_df['f_lo']
                expected_hi = total_cells * pred_props
                expected_lo = total_cells * (1 - pred_props)
                
                # Calculate log likelihood using Poisson distribution
                ll = pm.Poisson.dist(mu=expected_hi).logp(self.data_df['f_hi']).sum() + \
                     pm.Poisson.dist(mu=expected_lo).logp(self.data_df['f_lo']).sum()
                
                return ll
            
            # Add likelihood to model
            pm.Potential('likelihood', likelihood(q0_1, q1_0, init_0))
            
            if self.verbose:
                print("Starting MCMC sampling...")
            
            # Run MCMC
            trace = pm.sample(
                draws=n_draws,
                tune=tune,
                return_inferencedata=True,
                progressbar=self.verbose
            )
        
        if self.verbose:
            print("\nMCMC sampling complete!")
        
        return trace

    def analyze_results(self, trace):
        """Analyze MCMC results and return parameter estimates"""
        # Get median parameter estimates
        params_dict = {
            'q0_1': float(trace.posterior['q0_1'].median()),
            'q1_0': float(trace.posterior['q1_0'].median()),
            'init_0': float(trace.posterior['init_0'].median())
        }
        
        if self.verbose:
            print("\nParameter estimates (median):")
            for param, value in params_dict.items():
                print(f"{param}: {value:.3e}")
            
        return params_dict, trace

    def plot_fit_comparison(self, params_dict, figsize=(8, 6)):
        """Plot comparison of data and model predictions"""
        # Get predictions from surrogate model
        inputs = torch.tensor([[params_dict['q0_1'], params_dict['q1_0']]], dtype=torch.float32)
        inputs = torch.log10(inputs)
        
        self.model.eval()
        with torch.no_grad():
            pred_props = self.model(inputs).numpy()[0]
        
        # Plot comparison
        plt.figure(figsize=figsize)
        timepoints = self.data_df.index
        
        # Plot data points
        total_cells = self.data_df['f_hi'] + self.data_df['f_lo']
        data_props = self.data_df['f_hi'] / total_cells
        plt.scatter(timepoints, data_props, color='black', label='Data')
        
        # Plot model predictions
        plt.plot(timepoints, pred_props, 'r-', label='Model')
        
        plt.xlabel('Time')
        plt.ylabel('Proportion')
        plt.legend()
        plt.title('Data vs Model Predictions')
        plt.grid(True)
        
        return plt.gcf()

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
            
            # Get data counts
            n_hi = self.data_df.loc[t, 'f_hi']
            n_lo = self.data_df.loc[t, 'f_lo']
            
            ll += n_hi * np.log(p_hi + 1e-12) + n_lo * np.log(1 - p_hi + 1e-12)
        
        return ll

    def test_surrogate(self, n_tests=50):
        """Test surrogate model predictions against direct simulations"""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Neural network surrogate model must be trained before testing")
        
        # Generate random test points in log space
        np.random.seed(42)  # For reproducibility
        log_qs = np.random.uniform(-7, -2, size=(n_tests, 2))
        test_points = 10**log_qs
        
        # Get predictions from surrogate
        inputs = torch.tensor(test_points, dtype=torch.float32)
        inputs_log = torch.log10(inputs)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(inputs_log).numpy()
        
        # Run direct simulations for comparison
        true_results = []
        for q0_1, q1_0 in test_points:
            params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
            init_dict = {'state_0': 1, 'state_1': 0}  # Using init_0 = 1 as in dataset
            sim_results = ss1D.run_simulations(params, init_dict, n_sims=1000, divisions=list(self.data_df.index))
            # Extract the proportion in state 1 at each timepoint
            props = [np.mean(res == 1) for res in sim_results]  # Changed this line
            true_results.append(props)
        
        true_results = np.array(true_results)
        
        # Calculate errors
        abs_errors = np.abs(predictions - true_results)
        mean_error = np.mean(abs_errors)
        max_error = np.max(abs_errors)
        
        if self.verbose:
            print(f"\nSurrogate Model Test Results (n={n_tests}):")
            print(f"Mean absolute error: {mean_error:.6f}")
            print(f"Max absolute error: {max_error:.6f}")
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Surrogate vs Direct Simulation')
        
        for i in range(4):  # For each timepoint
            ax = axes[i//2, i%2]
            ax.scatter(true_results[:, i], predictions[:, i], alpha=0.5)
            
            # Add diagonal line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            
            ax.set_xlabel('Direct Simulation')
            ax.set_ylabel('Surrogate Prediction')
            ax.set_title(f'Timepoint {self.data_df.index[i]}')
            ax.grid(True)
        
        plt.tight_layout()
        return fig, (mean_error, max_error)

    def debug_single_comparison(self):
        """Debug a single comparison between surrogate and direct simulation"""
        # Test with larger transition rates
        test_points = [
            (1e-4, 1e-4),  # Original very small rates
            (0.1, 0.1),    # Moderate rates
            (0.5, 0.2)     # Large asymmetric rates
        ]
        
        figs = []
        for q0_1, q1_0 in test_points:
            print(f"\nTesting point: q0_1={q0_1:.1e}, q1_0={q1_0:.1e}")
            
            # Get surrogate prediction
            inputs = torch.tensor([[q0_1, q1_0]], dtype=torch.float32)
            inputs_log = torch.log10(inputs)
            
            self.model.eval()
            with torch.no_grad():
                pred = self.model(inputs_log).numpy()[0]
            
            # Get direct simulation
            params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
            init_dict = {'state_0': 1, 'state_1': 0}
            
            print("\nRunning direct simulation...")
            sim_results = ss1D.run_simulations(params, init_dict, n_sims=1000, divisions=list(self.data_df.index))
            
            # Calculate statistics for each timepoint
            timepoints = sorted(sim_results.keys())
            sim_means = []
            sim_vars = []
            
            for t in timepoints:
                sim_means.append(np.mean(sim_results[t]))
                sim_vars.append(np.var(sim_results[t]))
            
            # Create comparison plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot means
            ax1.plot(timepoints, sim_means, 'bo-', label='Simulation')
            ax1.plot(timepoints, pred, 'ro-', label='Surrogate')
            ax1.set_xlabel('Divisions')
            ax1.set_ylabel('E[f_hi|params]')
            ax1.set_title(f'Mean comparison (q0_1={q0_1:.1e}, q1_0={q1_0:.1e})')
            ax1.legend()
            ax1.grid(True)
            
            # Plot variances
            ax2.plot(timepoints, sim_vars, 'bo-', label='Simulation')
            ax2.set_xlabel('Divisions')
            ax2.set_ylabel('Var[f_hi|params]')
            ax2.set_title('Variance comparison')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            figs.append(fig)
            
            print("\nStatistical comparison:")
            print("Timepoints:", timepoints)
            print("Simulation means:", sim_means)
            print("Surrogate predictions:", pred)
            print("Simulation variances:", sim_vars)
        
        return figs

    def debug_training_data(self):
        """Debug the training data used for the surrogate"""
        print("\nTraining data summary:")
        print(f"Number of training points: {len(self.train_X)}")
        
        # Check parameter ranges
        q0_1_range = (10**self.train_X[:, 0].min(), 10**self.train_X[:, 0].max())
        q1_0_range = (10**self.train_X[:, 1].min(), 10**self.train_X[:, 1].max())
        print(f"\nParameter ranges in training data:")
        print(f"q0_1: {q0_1_range}")
        print(f"q1_0: {q1_0_range}")
        
        # Check output ranges
        y_ranges = []
        for i in range(self.train_y.shape[1]):
            y_range = (self.train_y[:, i].min(), self.train_y[:, i].max())
            y_ranges.append(y_range)
        
        print("\nOutput ranges in training data:")
        for i, y_range in enumerate(y_ranges):
            print(f"Output {i} (t={self.data_df.index[i]}): {y_range}")
        
        # Plot some random training examples
        n_examples = 5
        indices = np.random.choice(len(self.train_X), n_examples, replace=False)
        
        print("\nRandom training examples:")
        for idx in indices:
            q0_1 = 10**self.train_X[idx, 0]
            q1_0 = 10**self.train_X[idx, 1]
            y = self.train_y[idx]
            print(f"\nExample {idx}:")
            print(f"Parameters: q0_1={q0_1:.2e}, q1_0={q1_0:.2e}")
            print(f"Outputs: {y}")
            
            # Run simulation with these parameters
            params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
            init_dict = {'state_0': 1, 'state_1': 0}
            sim_results = ss1D.run_simulations(params, init_dict, n_sims=1000, divisions=list(self.data_df.index))
            
            sim_means = [np.mean(sim_results[t]) for t in sorted(sim_results.keys())]
            print(f"Simulation means: {sim_means}")

    def generate_training_data(self):
        """Generate training data for surrogate model"""
        print("Generating training data grid...")
        
        # Create log-spaced grid for parameters
        q_min, q_max = 1e-4, 1e-1  # Parameter range
        q_grid = np.logspace(np.log10(q_min), np.log10(q_max), self.n_grid)
        
        # Create parameter combinations
        q0_1_grid, q1_0_grid = np.meshgrid(q_grid, q_grid)
        params_list = np.column_stack((q0_1_grid.flatten(), q1_0_grid.flatten()))
        
        # Generate simulations for each parameter set
        X = []  # Parameters
        y = []  # Simulation results
        
        for q0_1, q1_0 in params_list:
            params = ss1D.TransitionParams(q0_1=q0_1, q1_0=q1_0)
            init_dict = {'state_0': 1, 'state_1': 0}
            
            sim_results = ss1D.run_simulations(params, init_dict, n_sims=self.n_sims, 
                                             divisions=list(self.data_df.index))
            
            # Calculate means for each timepoint
            means = [np.mean(sim_results[t]) for t in sorted(sim_results.keys())]
            
            X.append([np.log10(q0_1), np.log10(q1_0)])
            y.append(means)
        
        self.train_X = np.array(X)
        self.train_y = np.array(y)
        
        print(f"Generated {len(self.train_X)} training points")
