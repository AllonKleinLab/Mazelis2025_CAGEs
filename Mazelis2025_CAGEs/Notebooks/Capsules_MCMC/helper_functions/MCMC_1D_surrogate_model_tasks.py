"""
Minimal public surrogate model helpers for 1D MCMC used in script2_MCMC.py.

Contains:
- SurrogateModel (PyTorch-based)
- save_model_dict / load_model_dict
"""

from typing import Dict, List, Tuple, Optional
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class ProbabilityDistributionDataset(Dataset):
    """
    Dataset wrapping P0/P1 distributions with symmetry augmentation.
    """

    def __init__(self, init0_df: pd.DataFrame, init1_df: pd.DataFrame):
        self.param_columns = init0_df.columns[:2].tolist()
        self.dist_columns = init0_df.columns[2:].tolist()

        params0 = np.log10(init0_df.iloc[:, :2].values)
        dist0 = init0_df.iloc[:, 2:].values

        params1 = np.log10(init1_df.iloc[:, :2].values)
        params1_swapped = params1[:, [1, 0]]
        dist1 = init1_df.iloc[:, 2:].values[:, ::-1]

        self.parameters = torch.FloatTensor(np.vstack([params0, params1_swapped]))
        self.distributions = torch.FloatTensor(np.vstack([dist0, dist1]))

        self.linear_parameters = np.vstack(
            [init0_df.iloc[:, :2].values, init1_df.iloc[:, [1, 0]].values]
        )

    def __len__(self) -> int:
        return len(self.parameters)

    def __getitem__(self, idx: int):
        return self.parameters[idx], self.distributions[idx]


class ProbabilityDistributionNet(nn.Module):
    """Simple MLP mapping (log10 q01, log10 q10) -> categorical distribution."""

    def __init__(self, n_outputs: int, hidden_layers: List[int] | None = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers: List[nn.Module] = []
        prev_size = 2
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, n_outputs))
        layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    plt.figure(figsize=figsize)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


class SurrogateModel:
    """
    Wrapper around ProbabilityDistributionNet with k-fold training and persistence.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 40,
        n_folds: int = 5,
        patience: int = 10,
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.patience = patience
        self.model: ProbabilityDistributionNet | None = None
        self.dataset: ProbabilityDistributionDataset | None = None
        self.history: Dict[str, List[float]] | None = None

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(output, target)

    def _train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        history: Dict[str, List[float]],
        verbose: bool = True,
    ):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.n_epochs):
            self.model.train()
            train_losses = []
            for params, dist in train_loader:
                optimizer.zero_grad()
                output = self.model(params)
                loss = self._compute_loss(output, dist)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for params, dist in val_loader:
                    output = self.model(params)
                    loss = self._compute_loss(output, dist)
                    val_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses))

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
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}")
                print(f"Train Loss: {avg_train_loss:.6f}")
                print(f"Val Loss: {avg_val_loss:.6f}")

    def train(
        self, init0_df: pd.DataFrame, init1_df: pd.DataFrame, verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the surrogate model using k-fold cross validation.
        """
        self.dataset = ProbabilityDistributionDataset(init0_df, init1_df)
        n_outputs = len(init0_df.columns) - 2
        self.model = ProbabilityDistributionNet(n_outputs, self.hidden_layers)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        history = {"train_loss": [], "val_loss": []}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            if verbose:
                print(f"Fold {fold + 1}/{self.n_folds}")

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=train_sampler
            )
            val_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=val_sampler
            )

            self._train_fold(train_loader, val_loader, history, verbose)

        self.history = history
        return history

    def plot_history(
        self, title: Optional[str] = None, save_path: Optional[str] = None
    ):
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        if title is None:
            title = "Training History"
        _plot_training_history(self.history, title=title, save_path=save_path)

    def predict(self, parameters: np.ndarray, q0: float = 0.5) -> np.ndarray:
        """
        Predict distributions P(f_hi | q01, q10, q0) for given parameters.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            log_params = torch.FloatTensor(np.log10(parameters))
            P0 = self.model(log_params)
            log_params_swapped = log_params[:, [1, 0]]
            P1_swapped = self.model(log_params_swapped)
            P1 = torch.flip(P1_swapped, dims=[1])
            predictions = q0 * P0 + (1 - q0) * P1
        return predictions.numpy()

    def predict_df(
        self,
        parameters: Optional[np.ndarray] = None,
        q0: float = 0.5,
        n_grid: Optional[int] = None,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper returning predictions in the same DataFrame format
        as the simulation grid.
        """
        if parameters is None:
            if any(x is None for x in [n_grid, q_min, q_max]):
                raise ValueError(
                    "Must provide either parameters or (n_grid, q_min, q_max)"
                )
            q_vals = np.logspace(np.log10(q_min), np.log10(q_max), n_grid)
            parameters = np.array(
                [(q0_1, q1_0) for q0_1 in q_vals for q1_0 in q_vals]
            )

        predictions = self.predict(parameters, q0)
        df = pd.DataFrame(parameters, columns=["q0_1", "q1_0"])

        if self.dataset is not None and hasattr(self.dataset, "dist_columns"):
            dist_columns = self.dataset.dist_columns
        else:
            dist_columns = [f"{i}" for i in range(predictions.shape[1])]

        for i, col in enumerate(dist_columns):
            df[col] = predictions[:, i]

        return df

    def save(self, filepath: str):
        """
        Save trained model and minimal metadata.
        """
        torch.save(self.model.state_dict(), f"{filepath}_model.pt")
        save_dict = {
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "n_folds": self.n_folds,
            "patience": self.patience,
            "history": self.history,
            "dataset_info": {
                "param_columns": self.dataset.param_columns if self.dataset else None,
                "dist_columns": self.dataset.dist_columns if self.dataset else None,
            },
        }
        with open(f"{filepath}_config.pkl", "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, filepath: str) -> "SurrogateModel":
        """
        Load a previously-saved surrogate model.
        """
        with open(f"{filepath}_config.pkl", "rb") as f:
            config = pickle.load(f)

        model = cls(
            hidden_layers=config["hidden_layers"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            n_folds=config["n_folds"],
            patience=config["patience"],
        )
        model.history = config["history"]

        model.dataset = type(
            "DummyDataset",
            (),
            {
                "param_columns": config["dataset_info"]["param_columns"],
                "dist_columns": config["dataset_info"]["dist_columns"],
            },
        )

        n_outputs = len(config["dataset_info"]["dist_columns"])
        model.model = ProbabilityDistributionNet(n_outputs, config["hidden_layers"])
        model.model.load_state_dict(torch.load(f"{filepath}_model.pt"))
        model.model.eval()
        return model


def save_model_dict(model_dict: Dict[int, SurrogateModel], base_path: str):
    """
    Save a dictionary of SurrogateModel instances keyed by division number.
    """
    os.makedirs(base_path, exist_ok=True)
    for div, model in model_dict.items():
        model.save(f"{base_path}/model_div_{div}")
    with open(f"{base_path}/divisions.pkl", "wb") as f:
        pickle.dump(list(model_dict.keys()), f)


def load_model_dict(base_path: str) -> Dict[int, SurrogateModel]:
    """
    Load a dictionary of SurrogateModel instances from disk.
    """
    with open(f"{base_path}/divisions.pkl", "rb") as f:
        divisions = pickle.load(f)
    model_dict: Dict[int, SurrogateModel] = {}
    for div in divisions:
        model_dict[div] = SurrogateModel.load(f"{base_path}/model_div_{div}")
    return model_dict

