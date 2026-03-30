import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import copy
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def calculate_correlation(pred, values):
    """Calculate correlation in a differentiable way"""
    pred_centered = pred - pred.mean()
    values_tensor = torch.FloatTensor(values)
    values_centered = values_tensor - values_tensor.mean()
    
    correlation = (pred_centered * values_centered).sum() / (
        torch.sqrt((pred_centered ** 2).sum()) * 
        torch.sqrt((values_centered ** 2).sum())
    )
    return correlation

def evaluate_all_samples_adversarial(X, values, samples, times, conditions, rf_verbose=1, normalize_features=False):
    # Suppress Intel MKL warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', message='Intel MKL WARNING')
    
    if normalize_features:
        # Normalize features to sum to 1 per cell
        X = X / X.sum(axis=1, keepdims=True)
        if rf_verbose:
            print("\nFeatures normalized to sum to 1 per cell")
    
    # Convert times to numeric if they're categorical
    times = pd.to_numeric(times)
    
    if rf_verbose:
        print("\nPreparing data:")
        print(f"Input shape: {X.shape}")
        print(f"Time points: {np.unique(times)}")
    
    # Select validation control group
    val_ctrl = np.random.choice(['Ctrl-1', 'Ctrl-2', 'Ctrl-3'])
    is_val = conditions == val_ctrl
    
    if rf_verbose:
        print(f"\nUsing {val_ctrl} as validation set")
        print(f"Training samples: {len(np.unique(samples[~is_val]))}")
        print(f"Validation samples: {len(np.unique(samples[is_val]))}")
    
    # Create masks for different sets
    mask_t0 = times == 0
    mask_later = times > 0
    
    # Training sets
    train_t0_mask = (~is_val) & mask_t0
    train_later_mask = (~is_val) & mask_later
    
    # Validation sets
    val_t0_mask = is_val & mask_t0
    val_later_mask = is_val & mask_later
    
    if rf_verbose:
        print(f"Training t0 cells: {train_t0_mask.sum()}")
        print(f"Training later cells: {train_later_mask.sum()}")
        print(f"Validation t0 cells: {val_t0_mask.sum()}")
        print(f"Validation later cells: {val_later_mask.sum()}")
    
    # Create dataloaders for training later timepoints
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(values)
    
    train_dataset = TensorDataset(X_tensor[train_later_mask], y_tensor[train_later_mask])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    lambda_decorr = 10.0
    
    class Predictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            self.output = nn.Linear(32, 1)
        
        def forward(self, x):
            features = self.features(x)
            return self.output(features).squeeze(), features
    
    predictor = Predictor(X.shape[1])
    optimizer = torch.optim.Adam(predictor.parameters())
    
    # Store t0 data for correlation calculation
    X_t0 = X_tensor[mask_t0]
    y_t0 = y_tensor[mask_t0]
    
    n_epochs = 40
    
    if rf_verbose:
        print("\nStarting training:")
        print(f"Lambda (decorrelation penalty): {lambda_decorr}")
        print(f"Number of epochs: {n_epochs}")
    
    # Track metrics over epochs
    train_metrics = []
    val_metrics = []
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        predictor.train()
        total_train_loss = 0
        n_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred, _ = predictor(batch_X)
            
            # MSE loss for later timepoints (training)
            mse_loss = F.mse_loss(pred, batch_y)
            
            # Calculate t0 correlations separately for train and validation
            t0_pred_train, _ = predictor(X_tensor[train_t0_mask])
            t0_pred_val, _ = predictor(X_tensor[val_t0_mask])
            
            t0_corr_train = calculate_correlation(t0_pred_train, values[train_t0_mask])
            t0_corr_val = calculate_correlation(t0_pred_val, values[val_t0_mask])
            
            # Combined decorrelation loss
            decorr_loss = t0_corr_train**2 + t0_corr_val**2
            
            # Total loss
            loss = mse_loss + lambda_decorr * decorr_loss
            
            if n_batches == 0 and epoch % 5 == 0 and rf_verbose:
                print(f"\nBatch losses - Epoch {epoch}:")
                print(f"MSE loss: {mse_loss.item():.4f}")
                print(f"Train T0 correlation: {t0_corr_train.item():.4f}")
                print(f"Val T0 correlation: {t0_corr_val.item():.4f}")
                
                # Also show later timepoint correlations
                with torch.no_grad():
                    later_pred_train, _ = predictor(X_tensor[train_later_mask])
                    later_pred_val, _ = predictor(X_tensor[val_later_mask])
                    train_later_corr = spearmanr(values[train_later_mask], 
                                               later_pred_train.numpy())[0]
                    val_later_corr = spearmanr(values[val_later_mask], 
                                             later_pred_val.numpy())[0]
                print(f"Train later correlation: {train_later_corr:.4f}")
                print(f"Val later correlation: {val_later_corr:.4f}")
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            n_batches += 1
        
        # Calculate validation loss
        predictor.eval()
        with torch.no_grad():
            val_pred, _ = predictor(X_tensor[val_later_mask])
            val_mse = F.mse_loss(val_pred, y_tensor[val_later_mask])
            
            t0_pred_val, _ = predictor(X_tensor[val_t0_mask])
            t0_corr_val = calculate_correlation(t0_pred_val, values[val_t0_mask])
            
            val_loss = val_mse + lambda_decorr * t0_corr_val**2
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(predictor)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if rf_verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Store metrics
        with torch.no_grad():
            train_metrics.append({
                'loss': total_train_loss / n_batches,
                't0_corr': t0_corr_train.item(),
                'later_corr': spearmanr(values[~is_val & mask_later], 
                                      predictor(X_tensor[~is_val & mask_later])[0].detach().numpy())[0]
            })
            
            val_metrics.append({
                'loss': val_loss.item(),
                't0_corr': t0_corr_val.item(),
                'later_corr': spearmanr(values[is_val & mask_later], 
                                      val_pred.detach().numpy())[0]
            })
        
        if rf_verbose and (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Train - MSE loss: {train_metrics[-1]['loss']:.4f}")
            print(f"         T0 correlation: {train_metrics[-1]['t0_corr']:.3f}")
            print(f"         Later correlation: {train_metrics[-1]['later_corr']:.3f}")
            print(f"  Val   - MSE loss: {val_metrics[-1]['loss']:.4f}")
            print(f"         T0 correlation: {val_metrics[-1]['t0_corr']:.3f}")
            print(f"         Later correlation: {val_metrics[-1]['later_corr']:.3f}")
    
    # Use best model for final predictions
    predictor = best_model
    
    # Calculate final predictions and feature importance
    predictor.eval()
    with torch.no_grad():
        predictions = predictor(X_tensor)[0].numpy()
    
    importance_dict = {
        't0': calculate_importance(predictor, X_tensor[mask_t0], values[mask_t0]),
        't1plus': calculate_importance(predictor, X_tensor[mask_later], values[mask_later])
    }
    importance_dict['difference'] = importance_dict['t1plus'] - importance_dict['t0']
    
    # Overall correlation
    corr = spearmanr(values, predictions)[0]
    
    return predictions, importance_dict, corr, predictor, {'train': train_metrics, 'val': val_metrics}

def train_and_evaluate_model(adata, evaluate_func=evaluate_all_samples_adversarial, 
                           verbose=True, normalize_features=False):
    """
    Train and evaluate a model on the data in adata
    
    Parameters:
    -----------
    adata : AnnData
        Input data
    evaluate_func : function
        Function to use for evaluation
    verbose : bool
        Whether to print progress
    normalize_features : bool
        Whether to normalize NMF features to sum to 1 per cell
    """
    X = adata.obsm['X_nmf']
    values = adata.obs['log_n_counts'].values
    samples = adata.obs['sample_key'].values
    times = adata.obs['timepoint'].values
    conditions = adata.obs['condition'].values
    
    predictions, importance_dict, corr, model, metrics = evaluate_func(
        X, values, samples, times, conditions, 
        rf_verbose=verbose,
        normalize_features=normalize_features
    )
    
    return predictions, importance_dict, corr, model, metrics

def analyze_clone_sizes(adata, target_col='log_n_counts', sample_col='sample_key', 
                       time_col='timepoint', method='rf', use_ranks=False,
                       verbose=True, rf_verbose=1):
    evaluate_func = {
        #'rf': evaluate_sample_rf,
        #'nn': evaluate_sample_nn,
        'adversarial': evaluate_all_samples_adversarial
    }[method]
    
    if verbose:
        print(f'\nUsing {method.upper()} method')
    
    
    X, values, samples, times = prepare_data(
        adata, target_col=target_col, sample_col=sample_col, time_col=time_col,
        use_ranks=use_ranks, verbose=verbose
    )
    print('Training model...')
    predictions, correlations, sample_importance, metrics = train_and_evaluate_model(
        adata, evaluate_func=evaluate_func,
        use_ranks=use_ranks, verbose=verbose, normalize_features=False
    )
    
    # plot = create_summary_figure(adata, predictions, values, times, 
    #                            sample_importance, correlations,
    #                            use_ranks=use_ranks)
    
    return predictions, correlations, sample_importance, metrics #, plot



def prepare_data(adata, target_col='log_n_counts', sample_col='sample_key', time_col='timepoint', 
                use_ranks=False, verbose=True):
    if verbose:
        print("Checking required columns...")
    required_cols = [target_col, sample_col, time_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if verbose:
        print("Extracting data matrices...")
    X = adata.obsm['X_nmf']
    values = adata.obs[target_col].values
    samples = adata.obs[sample_col].values
    times = adata.obs[time_col].values
    
    if verbose:
        print(f"Data shape: {X.shape}")
        print(f"Number of unique samples: {len(np.unique(samples))}")
        print(f"Number of timepoints: {len(np.unique(times))}")
    
    if use_ranks:
        if verbose:
            print("Converting values to ranks...")
        ranks = np.zeros(len(adata))
        for sample in np.unique(samples):
            mask = samples == sample
            ranks[mask] = pd.Series(values[mask]).rank(method='average').values
            ranks[mask] = (ranks[mask] - 1) / (np.sum(mask) - 1)
        values = ranks
        if verbose:
            print("Rank conversion complete")
    
    return X, values, samples, times

def create_summary_figures(adata, predictions, true_values, sample_importance):
    # First figure: scatter plots
    timepoints = adata.obs['timepoint'].values
    unique_timepoints = sorted(np.unique(timepoints))
    n_timepoints = len(unique_timepoints)
    
    fig1 = plt.figure(figsize=(10, 3))
    for i, t in enumerate(unique_timepoints):
        ax = plt.subplot(1, n_timepoints, i+1)
        mask = timepoints == t
        time_corr, _ = spearmanr(true_values[mask], predictions[mask])
        
        ax.scatter(true_values[mask], predictions[mask], alpha=0.1, c='blue')
        min_val = min(true_values[mask].min(), predictions[mask].min())
        max_val = max(true_values[mask].max(), predictions[mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Actual Log Colony Size', fontsize=10)
        ax.set_ylabel('Predicted Log Colony Size' if i == 0 else '', fontsize=10)
        ax.set_title(f'Day {t}\nr = {time_corr:.2f}', fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    
    # Calculate correlation direction for each feature
    X = adata.obsm['X_nmf']  # NMF features
    timepoints = adata.obs['timepoint'].values
    mask_t0 = timepoints == 0
    mask_later = ~mask_t0
    
    correlations_t0 = np.array([spearmanr(X[mask_t0, i], true_values[mask_t0])[0] 
                               for i in range(X.shape[1])])
    correlations_later = np.array([spearmanr(X[mask_later, i], true_values[mask_later])[0] 
                                 for i in range(X.shape[1])])
    
    # Second figure: importance plots
    fig2 = plt.figure(figsize=(8, 8))
    gs = fig2.add_gridspec(2, 1, height_ratios=[1, 1])
    
    # Top plot: Feature importance comparison
    ax = fig2.add_subplot(gs[0])
    if not sample_importance:
        ax.text(0.5, 0.5, 'No importance data available', ha='center', va='center', fontsize=14)
    else:
        importance_t0 = sample_importance['t0']
        importance_later = sample_importance['t1plus']
        importance_diff = sample_importance['difference']
        
        # Sort features by absolute difference in importance
        sorted_idx = np.argsort(np.abs(importance_diff))[::-1]
        features = [f'NMF_{i}' for i in range(len(importance_t0))]
        
        # Plot horizontal bars - limit to top 10 features for readability
        n_features_show = min(10, len(features))
        sorted_idx = sorted_idx[:n_features_show]
        y_pos = np.arange(n_features_show)
        width = 0.35
        
        # Add correlation direction to labels
        feature_labels = []
        for i in sorted_idx:
            label = f'NMF_{i}'
            if abs(correlations_later[i]) > 0.1:  # Only show direction if correlation is meaningful
                sign = '+' if correlations_later[i] > 0 else '-'
                label += f' ({sign})'
            feature_labels.append(label)
        
        ax.barh(y_pos - width/2, importance_later[sorted_idx], width, 
                label='Later timepoints', color='green', alpha=0.6)
        ax.barh(y_pos + width/2, importance_t0[sorted_idx], width,
                label='Time 0', color='red', alpha=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=10)
        ax.set_title('Feature Importance Comparison\n(top 10 features by difference, +/- shows correlation direction)', 
                    fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Bottom plot: Importance difference plot
    ax = fig2.add_subplot(gs[1])
    if not sample_importance:
        ax.text(0.5, 0.5, 'No importance data available', ha='center', va='center', fontsize=14)
    else:
        # Sort by difference and show top 10 positive and negative
        sorted_diff_idx = np.argsort(importance_diff)
        n_features_show = min(10, len(features)//2)
        show_idx = list(sorted_diff_idx[-n_features_show:]) + list(sorted_diff_idx[:n_features_show])
        
        # Create labels with correlation direction
        feature_labels = []
        for i in show_idx:
            label = f'NMF_{i}'
            if abs(correlations_later[i]) > 0.1:
                sign = '+' if correlations_later[i] > 0 else '-'
                label += f' ({sign})'
            feature_labels.append(label)
        
        # Create diverging bar plot
        y_pos = np.arange(len(show_idx))
        ax.barh(y_pos, importance_diff[show_idx], 
                color=['red' if x < 0 else 'green' for x in importance_diff[show_idx]],
                alpha=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=10)
        ax.set_xlabel('Importance Difference (Later - Time 0)', fontsize=10)
        ax.set_title('Features Most Different Between Timepoints\n(green = more important later, +/- shows correlation direction)', 
                    fontsize=11)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    plt.tight_layout()
    return fig1, fig2

def calculate_importance(model, X, y):
    """Calculate feature importance using permutation importance method"""
    model.eval()
    with torch.no_grad():
        base_pred, _ = model(X)
        base_pred = base_pred.numpy()
    
    # y is already a numpy array, so don't call .numpy() on it
    base_score = spearmanr(y, base_pred)[0]
    
    importance = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        X_permuted = X.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(len(X_permuted)), i]
        with torch.no_grad():
            perm_pred, _ = model(X_permuted)
            perm_pred = perm_pred.numpy()
        perm_score = spearmanr(y, perm_pred)[0]
        importance[i] = base_score - perm_score
    
    return importance

def analyze_model_predictions(model, adata, times, normalize_features=False):
    """
    Analyze how the model combines NMF factors and connect to underlying genes
    
    Parameters:
    -----------
    model : trained Predictor model
    adata : AnnData object
    times : array of timepoints
    normalize_features : bool, whether features were normalized
    
    Returns:
    --------
    dict containing analysis results
    """
    model.eval()
    X = adata.obsm['X_nmf']
    if normalize_features:
        X = X / X.sum(axis=1, keepdims=True)
    
    # Convert times to numeric if they're categorical
    times = pd.to_numeric(times)
    
    # Get model predictions for later timepoints
    mask_later = times > 0
    X_later = torch.FloatTensor(X[mask_later])
    
    with torch.no_grad():
        predictions = model(X_later)[0].numpy()
    
    # Fit linear model to explain predictions
    linear_model = LinearRegression()
    linear_model.fit(X[mask_later], predictions)
    
    # Get ordered weights for all NMF factors
    factor_weights = []
    for i in range(len(linear_model.coef_)):
        factor_weights.append({
            'factor': f'NMF_{i}',
            'coefficient': linear_model.coef_[i]
        })
    factor_weights = sorted(factor_weights, key=lambda x: abs(x['coefficient']), reverse=True)
    
    # Get combined gene weights using H*x
    nmf_components = adata.uns['nmf']['components']  # H matrix (factors × genes)
    gene_names = adata.uns['nmf']['highly_variable_genes']  # Correct gene names
    
    # Multiply H by the linear coefficients
    combined_gene_weights = np.dot(linear_model.coef_, nmf_components)  # (factors) × (factors × genes) -> (genes)
    
    # Get top genes by absolute weight
    gene_importance = []
    for i, gene in enumerate(gene_names):
        gene_importance.append({
            'gene': gene,
            'weight': combined_gene_weights[i]
        })
    gene_importance = sorted(gene_importance, key=lambda x: abs(x['weight']), reverse=True)
    
    # Calculate R² of linear approximation
    linear_predictions = linear_model.predict(X[mask_later])
    r2_score = np.corrcoef(predictions, linear_predictions)[0,1]**2
    
    results = {
        'linear_coefficients': linear_model.coef_,
        'factor_weights': factor_weights,
        'gene_weights': gene_importance,
        'linear_r2': r2_score,
        'intercept': linear_model.intercept_
    }
    
    return results

def print_factor_analysis(analysis_results, top_n_factors=15, top_n_genes=20):
    """Print readable summary of factor analysis"""
    print(f"\nLinear model R² = {analysis_results['linear_r2']:.3f}")
    
    print("\nAll NMF factor coefficients (sorted by importance):")
    for fw in analysis_results['factor_weights']:
        print(f"{fw['factor']}: {fw['coefficient']:.3f}")
    
    print("\nTop factors and their associated genes:")
    for fw in analysis_results['factor_weights'][:top_n_factors]:
        factor = fw['factor']
        if factor in analysis_results['factor_weights']:
            factor_info = analysis_results['factor_weights'][factor]
            print(f"\n{factor} (coefficient: {fw['coefficient']:.3f}):")
            print("Top genes:")
            for gene, loading in factor_info['top_genes'][:top_n_genes]:
                print(f"  {gene}: {loading:.3f}")
    
    print("\nTop genes from combined H*x analysis:")
    for g in analysis_results['gene_weights'][:top_n_genes]:
        print(f"{g['gene']}: {g['weight']:.3f}")

def plot_learning_curves(metrics):
    """
    Plot learning curves showing training and validation metrics over epochs
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing 'train' and 'val' lists of metrics per epoch
    
    Returns:
    --------
    fig : matplotlib figure
        Figure containing the learning curves
    """
    fig = plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(131)
    plt.plot([m['loss'] for m in metrics['train']], label='Train', color='blue', alpha=0.7)
    plt.plot([m['loss'] for m in metrics['val']], label='Validation', color='red', alpha=0.7)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # T0 correlation plot
    plt.subplot(132)
    plt.plot([m['t0_corr'] for m in metrics['train']], label='Train', color='blue', alpha=0.7)
    plt.plot([m['t0_corr'] for m in metrics['val']], label='Validation', color='red', alpha=0.7)
    plt.title('T0 correlation over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    
    # Later correlation plot
    plt.subplot(133)
    plt.plot([m['later_corr'] for m in metrics['train']], label='Train', color='blue', alpha=0.7)
    plt.plot([m['later_corr'] for m in metrics['val']], label='Validation', color='red', alpha=0.7)
    plt.title('Later correlation over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    
    plt.tight_layout()
    return fig

def plot_top_genes_vs_size(adata, gene_weights, times, predictions, n_genes=4):
    """
    Plot scatter plots of top genes vs clone size and predictions
    """
    times = pd.to_numeric(times)
    mask_t0 = times == 0
    mask_later = times > 0
    
    fig, axes = plt.subplots(3, n_genes, figsize=(2*n_genes, 6))
    
    for i in range(n_genes):
        gene = gene_weights[i]['gene']
        weight = gene_weights[i]['weight']
        
        # Get gene expression
        gene_expr = adata[:, gene].X.toarray().flatten()
        clone_size = adata.obs['log_n_counts'].values
        
        # t0 plot
        ax = axes[0, i]
        ax.scatter(gene_expr[mask_t0], clone_size[mask_t0], 
                  alpha=0.5, label='t=0', color='blue')
        ax.set_xlabel(gene)
        ax.set_ylabel('log clone size' if i==0 else '')
        r = np.corrcoef(gene_expr[mask_t0], clone_size[mask_t0])[0,1]
        ax.set_title(f't=0, r={r:.3f}')
        
        # Later timepoints actual size
        ax = axes[1, i]
        ax.scatter(gene_expr[mask_later], clone_size[mask_later], 
                  alpha=0.5, label='t>0', color='red')
        ax.set_xlabel(gene)
        ax.set_ylabel('actual log clone size' if i==0 else '')
        r = np.corrcoef(gene_expr[mask_later], clone_size[mask_later])[0,1]
        ax.set_title(f't>0 actual, r={r:.3f}')
        
        # Later timepoints predicted size
        ax = axes[2, i]
        ax.scatter(gene_expr[mask_later], predictions[mask_later], 
                  alpha=0.5, label='t>0', color='green')
        ax.set_xlabel(f'{gene}\nweight={weight:.3f}')
        ax.set_ylabel('predicted log clone size' if i==0 else '')
        r = np.corrcoef(gene_expr[mask_later], predictions[mask_later])[0,1]
        ax.set_title(f't>0 predicted, r={r:.3f}')
    
    plt.tight_layout()
    return fig

def plot_top_factors_vs_size(adata, factor_weights, times, predictions, n_factors=4):
    """
    Plot scatter plots of top NMF factors vs clone size and predictions
    
    Parameters:
    -----------
    adata : AnnData object
    factor_weights : list of dict
        From analysis_results['factor_weights']
    times : array
        Timepoint values
    predictions : array
        Model predictions
    n_factors : int
        Number of top factors to plot
    """
    times = pd.to_numeric(times)
    mask_t0 = times == 0
    mask_later = times > 0
    
    fig, axes = plt.subplots(3, n_factors, figsize=(2*n_factors, 6))
    
    for i in range(n_factors):
        factor = factor_weights[i]['factor']
        weight = factor_weights[i]['coefficient']
        factor_idx = int(factor.split('_')[1])  # Get numerical index from 'NMF_X'
        
        # Get factor values
        factor_expr = adata.obsm['X_nmf'][:, factor_idx]
        clone_size = adata.obs['log_n_counts'].values
        
        # t0 plot
        ax = axes[0, i]
        ax.scatter(factor_expr[mask_t0], clone_size[mask_t0], 
                  alpha=0.5, label='t=0', color='blue')
        ax.set_xlabel(factor)
        ax.set_ylabel('log clone size' if i==0 else '')
        r = np.corrcoef(factor_expr[mask_t0], clone_size[mask_t0])[0,1]
        ax.set_title(f't=0, r={r:.3f}')
        
        # Later timepoints actual size
        ax = axes[1, i]
        ax.scatter(factor_expr[mask_later], clone_size[mask_later], 
                  alpha=0.5, label='t>0', color='red')
        ax.set_xlabel(factor)
        ax.set_ylabel('actual log clone size' if i==0 else '')
        r = np.corrcoef(factor_expr[mask_later], clone_size[mask_later])[0,1]
        ax.set_title(f't>0 actual, r={r:.3f}')
        
        # Later timepoints predicted size
        ax = axes[2, i]
        ax.scatter(factor_expr[mask_later], predictions[mask_later], 
                  alpha=0.5, label='t>0', color='green')
        ax.set_xlabel(f'{factor}\nweight={weight:.3f}')
        ax.set_ylabel('predicted log clone size' if i==0 else '')
        r = np.corrcoef(factor_expr[mask_later], predictions[mask_later])[0,1]
        ax.set_title(f't>0 predicted, r={r:.3f}')
    
    plt.tight_layout()
    return fig

def fit_discriminative_linear_model(X, times, model, lambda_t0=0.0, lambda_reg=0.01, debug=False):
    """
    Analyze how the adversarial neural network combines NMF factors with explicit t0 anti-fitting
    
    Parameters:
    -----------
    X : array
        NMF factors matrix
    times : array
        Timepoints
    model : trained Predictor model
        The adversarial neural network model
    lambda_t0 : float
        Weight for t0 decorrelation penalty (higher values encourage zero correlation at t0)
    lambda_reg : float
        L2 regularization strength
    
    Returns:
    --------
    tuple:
        - weights : array of linear coefficients
        - offset : float bias term
        - metrics : dict of performance metrics with factor_weights as DataFrame
    """
    from sklearn.linear_model import Ridge
    from scipy.optimize import minimize
    
    times = pd.to_numeric(times)
    mask_t0 = times == 0
    mask_later = times > 0
    
    # Get neural network predictions
    X_tensor = torch.FloatTensor(X)
    model.eval()
    with torch.no_grad():
        nn_predictions, _ = model(X_tensor)
        nn_predictions = nn_predictions.numpy()
    
    # Get scale of predictions for normalization
    #pred_scale = np.std(nn_predictions[mask_later])
    n_later = np.sum(mask_later)
    #n_t0 = np.sum(mask_t0)
    
    def loss_function(params):
        weights = params[:-1]  # All but last parameter are weights
        offset = params[-1]    # Last parameter is the offset
        predictions = X @ weights + offset
        
        # MSE loss for t>0 predictions
        later_mse = np.sum((predictions[mask_later] - nn_predictions[mask_later])**2) / n_later
        
        # t0 correlation loss
        if lambda_t0 > 0:
            t0_pred = predictions[mask_t0]
            t0_target = nn_predictions[mask_t0]
            # Use covariance instead of correlation for smoother gradients
            t0_cov = np.cov(t0_pred, t0_target)[0,1]
            t0_loss = lambda_t0 * (t0_cov**2)
        else:
            t0_loss = 0
            
        # L2 regularization (only on weights, not offset)
        reg_loss = lambda_reg * np.sum(weights**2)
        
        # Add penalty for sum of weights being far from 1
        # This encourages the weights to sum to approximately 1 since NMF factors sum to 1
        weights_sum_penalty = 10.0 * (np.sum(weights) - 1.0)**2
        
        if debug:
            print(f"MSE: {later_mse:.4f}, T0: {t0_loss:.4f}, Reg: {reg_loss:.4f}, Sum penalty: {weights_sum_penalty:.4f}")
        
        return later_mse + t0_loss + reg_loss + weights_sum_penalty
    
    # Initialize with Ridge regression solution
    ridge = Ridge(alpha=lambda_reg)
    ridge.fit(X[mask_later], nn_predictions[mask_later])
    initial_weights = ridge.coef_
    initial_offset = ridge.intercept_
    
    # Adjust initial weights to sum to 1
    initial_weights = initial_weights / np.sum(initial_weights)
    initial_params = np.concatenate([initial_weights, [initial_offset]])
    
    # Optimize
    result = minimize(loss_function, initial_params, method='BFGS', tol=1e-8)
    if not result.success:
        print(f"Optimization warning: {result.message}")
    
    optimal_weights = result.x[:-1]
    optimal_offset = result.x[-1]
    
    # Calculate metrics
    linear_predictions = X @ optimal_weights + optimal_offset
    metrics = {
        't0_correlation': np.corrcoef(linear_predictions[mask_t0], nn_predictions[mask_t0])[0,1],
        'later_correlation': np.corrcoef(linear_predictions[mask_later], nn_predictions[mask_later])[0,1],
        't0_mse': np.mean((linear_predictions[mask_t0] - nn_predictions[mask_t0])**2),
        'later_mse': np.mean((linear_predictions[mask_later] - nn_predictions[mask_later])**2),
        'offset': optimal_offset,
        'weights_sum': np.sum(optimal_weights)
    }
    
    metrics['linear_r2'] = metrics['later_correlation']**2
    
    # Create factor weights DataFrame
    factor_weights = pd.DataFrame({
        'factor': [f'NMF_{i}' for i in range(len(optimal_weights))],
        'coefficient': optimal_weights,
        'abs_coef': np.abs(optimal_weights)
    })
    
    # Sort by absolute coefficient value
    factor_weights = factor_weights.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    
    metrics['factor_weights'] = factor_weights
    return optimal_weights, optimal_offset, metrics

def print_discriminative_model(metrics, adata, top_n_genes=20):
    """
    Print interpretable summary of the discriminative linear model including gene analysis
    """
    print("\nPerformance metrics:")
    print(f"t=0 correlation: {metrics['t0_correlation']:.3f}")
    print(f"t>0 correlation: {metrics['later_correlation']:.3f}")
    print(f"Linear model R²: {metrics['linear_r2']:.3f}")
    print(f"Offset term: {metrics['offset']:.3f}")
    
    # Get factor weights from DataFrame
    factor_weights_df = metrics['factor_weights']
    
    print("\nAll NMF factors (sorted by absolute coefficient):")
    for _, row in factor_weights_df.iterrows():
        sign = '+' if row['coefficient'] > 0 else ''
        print(f"{row['factor']}: {sign}{row['coefficient']:.3f}")
    
    # Gene analysis using all factors
    nmf_components = adata.uns['nmf']['components']
    gene_names = adata.uns['nmf']['highly_variable_genes']
    
    # Create coefficient vector maintaining original NMF factor order
    coef_vector = np.zeros(len(factor_weights_df))
    for _, row in factor_weights_df.iterrows():
        idx = int(row['factor'].split('_')[1])
        coef_vector[idx] = row['coefficient']
    
    # Calculate combined gene weights using all factors
    combined_gene_weights = np.dot(coef_vector, nmf_components)
    
    # Create gene weights DataFrame
    gene_weights_df = pd.DataFrame({
        'gene': gene_names,
        'weight': combined_gene_weights,
        'abs_weight': np.abs(combined_gene_weights)
    }).sort_values('abs_weight', ascending=False)
    
    print(f"\nTop {top_n_genes} positive genes (predict larger colonies):")
    positive_genes = gene_weights_df[gene_weights_df['weight'] > 0].head(top_n_genes)
    for _, row in positive_genes.iterrows():
        print(f"{row['gene']}: +{row['weight']:.3f}")
    
    print(f"\nTop {top_n_genes} negative genes (predict smaller colonies):")
    negative_genes = gene_weights_df[gene_weights_df['weight'] < 0].head(top_n_genes)
    for _, row in negative_genes.iterrows():
        print(f"{row['gene']}: {row['weight']:.3f}")
    
    print(f"\nFinal prediction formula:")
    print(f"prediction = (NMF factors × weights) + {metrics['offset']:.3f}")
    
    return gene_weights_df


