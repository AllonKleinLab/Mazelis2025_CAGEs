import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ProbabilityDistributionDataset(Dataset):
    """Dataset class for probability distribution data"""
    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns:
            - First two columns: parameters (q0_1, q1_0)
            - Remaining columns: probability distribution values
        """
        # Convert parameters to log space for training
        self.parameters = torch.FloatTensor(np.log10(df.iloc[:, :2].values))
        self.distributions = torch.FloatTensor(df.iloc[:, 2:].values)
        
        # Store original linear parameters for reference
        self.linear_parameters = torch.FloatTensor(df.iloc[:, :2].values)
        
        # Store column names for later reference
        self.param_columns = df.columns[:2].tolist()
        self.dist_columns = df.columns[2:].tolist()
        
    def __len__(self):
        return len(self.parameters)
    
    def __getitem__(self, idx):
        return self.parameters[idx], self.distributions[idx]

class ProbabilityDistributionNet(nn.Module):
    """Neural network for predicting probability distributions"""
    def __init__(self, n_outputs: int, hidden_layers: List[int] = [64, 32]):
        """
        Parameters
        ----------
        n_outputs : int
            Number of probability values to predict
        hidden_layers : List[int]
            List of hidden layer sizes
        """
        super().__init__()
        
        # Build network layers
        layers = []
        prev_size = 2  # Input size (2 parameters)
        
        # Add hidden layers - removed batch normalization
        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU()
            ])
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, n_outputs))
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def plot_training_history(history: Dict[str, List[float]], 
                         title: str = "Training History",
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None):
    """
    Plot training and validation loss over epochs
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary containing 'train_loss' and 'val_loss' lists
    title : str, optional
        Plot title, by default "Training History"
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    save_path : Optional[str], optional
        If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

class SurrogateModel:
    """Main class for training and using the surrogate model"""
    def __init__(self, 
                 hidden_layers: List[int] = [64, 32],
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 n_folds: int = 5,
                 patience: int = 10):
        """
        Parameters
        ----------
        hidden_layers : List[int]
            Sizes of hidden layers
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        n_epochs : int
            Maximum number of epochs
        n_folds : int
            Number of folds for cross-validation
        patience : int
            Number of epochs to wait for improvement before early stopping
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.patience = patience
        self.model = None
        self.dataset = None
        
    def plot_history(self, title: str = None, save_path: Optional[str] = None):
        """
        Plot the training history
        
        Parameters
        ----------
        title : str, optional
            Custom title for the plot. If None, uses default
        save_path : Optional[str], optional
            If provided, save the plot to this path
        """
        if not hasattr(self, 'history'):
            raise ValueError("No training history available. Train the model first.")
            
        if title is None:
            title = "Training History"
            
        plot_training_history(self.history, title=title, save_path=save_path)
    
    def train(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Train the model using k-fold cross validation
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
            
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        self.dataset = ProbabilityDistributionDataset(df)
        n_outputs = len(df.columns) - 2
        
        # Initialize model
        self.model = ProbabilityDistributionNet(n_outputs, self.hidden_layers)
        
        # Setup k-fold cross validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Train with cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"Fold {fold + 1}/{self.n_folds}")
            
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size,
                sampler=train_sampler
            )
            val_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=val_sampler
            )
            
            # Training loop for this fold
            self._train_fold(train_loader, val_loader, history)
        
        # Store history in instance variable
        self.history = history
        
        return history
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between predicted and target distributions"""
        return torch.nn.functional.mse_loss(output, target)
    
    def _train_fold(self, 
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    history: Dict[str, List[float]]):
        """Train model on one fold"""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            for params, dist in train_loader:
                optimizer.zero_grad()
                output = self.model(params)
                loss = self._compute_loss(output, dist)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for params, dist in val_loader:
                    output = self.model(params)
                    loss = self._compute_loss(output, dist)
                    val_losses.append(loss.item())
            
            # Compute average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {
                    key: value.cpu().clone() 
                    for key, value in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_model_state)
                    break
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}")
                print(f"Train Loss: {avg_train_loss:.6f}")
                print(f"Val Loss: {avg_val_loss:.6f}")
    
    def predict(self, parameters: np.ndarray) -> np.ndarray:
        """
        Predict probability distributions for given parameters
        
        Parameters
        ----------
        parameters : np.ndarray
            Array of shape (n_samples, 2) containing parameter pairs in linear space
            
        Returns
        -------
        np.ndarray
            Predicted probability distributions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            # Convert input parameters to log space
            log_params = torch.FloatTensor(np.log10(parameters))
            predictions = self.model(log_params)
        
        return predictions.numpy()

    def plot_model_predictions(self, df: pd.DataFrame, n_interpolations: int = 5, figsize: Tuple[int, int] = (10, 6)):
        """
        Visualize model predictions for interpolated parameters between two random points
        
        Parameters
        ----------
        df : pd.DataFrame
            Original training data
        n_interpolations : int, optional
            Number of interpolation points between the two chosen parameters
        figsize : Tuple[int, int], optional
            Figure size
        """
        # Randomly select two parameter sets
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        params1 = df.iloc[idx1, :2].values
        params2 = df.iloc[idx2, :2].values
        
        # Create interpolated parameters in log space
        log_params1 = np.log10(params1)
        log_params2 = np.log10(params2)
        alphas = np.linspace(0, 1, n_interpolations)
        
        interpolated_log_params = np.array([
            log_params1 * (1-alpha) + log_params2 * alpha 
            for alpha in alphas
        ])
        
        # Convert back to linear space
        interpolated_params = 10**interpolated_log_params
        
        # Get predictions for all interpolated points
        predictions = self.predict(interpolated_params)
        
        # Create plot
        plt.figure(figsize=figsize)
        x_vals = np.array([float(x) for x in df.columns[2:]])
        
        # Plot true distributions for endpoints
        true_dist1 = df.iloc[idx1, 2:].values
        true_dist2 = df.iloc[idx2, 2:].values
        
        plt.plot(x_vals, true_dist1, 'o-', color='blue', label=f'True 1 (q0_1={params1[0]:.3e}, q1_0={params1[1]:.3e})')
        plt.plot(x_vals, true_dist2, 'o-', color='red', label=f'True 2 (q0_1={params2[0]:.3e}, q1_0={params2[1]:.3e})')
        
        # Plot interpolated predictions
        colors = plt.cm.viridis(np.linspace(0, 1, n_interpolations))
        for i, (params, pred, color) in enumerate(zip(interpolated_params, predictions, colors)):
            plt.plot(x_vals, pred, '--', color=color, 
                    label=f'Interpolated {i+1} (q0_1={params[0]:.3e}, q1_0={params[1]:.3e})')
        
        plt.xlabel('Fraction in high state')
        plt.ylabel('Probability')
        plt.title('True vs Interpolated Distributions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_random_prediction_with_neighbors(self, df: pd.DataFrame, n_samples: int = 1, figsize: Tuple[int, int] = (10, 6)):
        """
        Pick random parameters and compare model predictions with nearest neighbors in training data
        
        Parameters
        ----------
        df : pd.DataFrame
            Original training data
        n_samples : int, optional
            Number of random parameter sets to test
        figsize : Tuple[int, int], optional
            Figure size
        """
        # Get parameter ranges from training data
        q0_1_range = df['q0_1'].agg(['min', 'max']).values
        q1_0_range = df['q1_0'].agg(['min', 'max']).values
        
        # Generate random parameters in log space
        log_q0_1 = np.random.uniform(np.log10(q0_1_range[0]), np.log10(q0_1_range[1]), n_samples)
        log_q1_0 = np.random.uniform(np.log10(q1_0_range[0]), np.log10(q1_0_range[1]), n_samples)
        
        # Convert to linear space
        random_params = np.array(list(zip(10**log_q0_1, 10**log_q1_0)))
        
        # Setup plotting
        fig, axes = plt.subplots(n_samples, 1, figsize=(figsize[0], figsize[1]*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        x_vals = np.array([float(x) for x in df.columns[2:]])
        
        for idx, (ax, params) in enumerate(zip(axes, random_params)):
            # Get model prediction for random parameters
            pred_dist = self.predict(params.reshape(1, -1))[0]
            
            # Find two nearest neighbors in training data
            distances = np.sqrt(
                (np.log10(df['q0_1'].values) - np.log10(params[0]))**2 + 
                (np.log10(df['q1_0'].values) - np.log10(params[1]))**2
            )
            nearest_indices = np.argsort(distances)[:2]
            
            # Plot prediction and nearest neighbors
            ax.plot(x_vals, pred_dist, 'k--', label=f'Prediction (q0_1={params[0]:.3e}, q1_0={params[1]:.3e})')
            
            for i, idx in enumerate(nearest_indices):
                neighbor_params = df.iloc[idx, :2].values
                neighbor_dist = df.iloc[idx, 2:].values
                ax.plot(x_vals, neighbor_dist, 'o-', 
                       label=f'Neighbor {i+1} (q0_1={neighbor_params[0]:.3e}, q1_0={neighbor_params[1]:.3e})')
            
            ax.set_xlabel('Fraction in high state')
            ax.set_ylabel('Probability')
            ax.set_title(f'Random Parameter Set {idx+1}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def predict_df(self, parameters: np.ndarray = None, n_grid: int = None, q_min: float = None, q_max: float = None) -> pd.DataFrame:
        """
        Generate predictions in a DataFrame format matching the training data
        
        Parameters
        ----------
        parameters : np.ndarray, optional
            Array of shape (n_samples, 2) containing parameter pairs
            If None, generates a grid using n_grid, q_min, and q_max
        n_grid : int, optional
            Number of grid points for each parameter if parameters not provided
        q_min : float, optional
            Minimum parameter value if parameters not provided
        q_max : float, optional
            Maximum parameter value if parameters not provided
            
        Returns
        -------
        pd.DataFrame
            DataFrame with same format as training data:
            - First two columns: parameters (q0_1, q1_0)
            - Remaining columns: predicted probability distributions
        """
        if parameters is None:
            if any(x is None for x in [n_grid, q_min, q_max]):
                raise ValueError("Must provide either parameters or (n_grid, q_min, q_max)")
            
            # Create log-spaced parameter grid
            q_vals = np.logspace(np.log10(q_min), np.log10(q_max), n_grid)
            parameters = np.array([(q0_1, q1_0) 
                                 for q0_1 in q_vals 
                                 for q1_0 in q_vals])
        
        # Get predictions
        predictions = self.predict(parameters)
        
        # Create DataFrame
        df = pd.DataFrame(parameters, columns=['q0_1', 'q1_0'])
        
        # Add distribution columns using the same format as training data
        if hasattr(self.dataset, 'dist_columns'):
            dist_columns = self.dataset.dist_columns
        else:
            dist_columns = [f"{i}" for i in range(predictions.shape[1])]
        
        for i, col in enumerate(dist_columns):
            df[col] = predictions[:, i]
        
        return df

