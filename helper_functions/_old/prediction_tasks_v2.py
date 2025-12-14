import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def prepare_data(adata, target_col='n_counts', sample_col='sample_key', time_col='timepoint', verbose=True):
    """
    Prepare data for modeling, converting target to within-sample ranks
    """
    # Verify columns exist
    required_cols = [target_col, sample_col, time_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if verbose:
        print(f"Found {adata.shape[0]} cells/clones and {adata.obsm['X_nmf'].shape[1]} NMF factors")
    
    # Extract NMF factors
    X = adata.obsm['X_nmf']
    
    # Convert target to ranks within each sample
    ranks = np.zeros(len(adata))
    samples = adata.obs[sample_col].values
    times = adata.obs[time_col].values
    
    if verbose:
        print(f"Converting {target_col} to within-sample ranks...")
        print(f"Total samples: {len(np.unique(samples))}")
    
    unique_samples = np.unique(samples)
    for i, sample in enumerate(unique_samples, 1):
        mask = samples == sample
        # Convert to ranks, handling ties with average
        sample_values = adata.obs[target_col].values[mask]
        ranks[mask] = pd.Series(sample_values).rank(method='average').values
        
        # Normalize ranks to 0-1 range within sample
        ranks[mask] = (ranks[mask] - 1) / (np.sum(mask) - 1)
        
        if verbose:
            print(f"\nSample {sample} (#{i}):")
            print(f"  Points in sample: {np.sum(mask)}")
            print(f"  Original values range: [{sample_values.min():.3f}, {sample_values.max():.3f}]")
            print(f"  Rank range: [{ranks[mask].min():.3f}, {ranks[mask].max():.3f}]")
            print(f"  Unique ranks: {len(np.unique(ranks[mask]))}")
    
    return X, ranks, samples, times

def evaluate_sample_rf(X, ranks, train_idx, test_idx, times=None, rf_verbose=0):
    """
    Train and evaluate model for a single sample using Random Forest
    """
    if rf_verbose > 0:
        print(f"\n    Training set size: {len(train_idx)}")
        print(f"    Test set size: {len(test_idx)}")
        print(f"    Training ranks range: [{ranks[train_idx].min():.3f}, {ranks[train_idx].max():.3f}]")
        print(f"    Testing ranks range: [{ranks[test_idx].min():.3f}, {ranks[test_idx].max():.3f}]")
    
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        verbose=rf_verbose,
        random_state=42
    )
    
    rf.fit(X[train_idx], ranks[train_idx])
    predictions = rf.predict(X[test_idx])
    
    # Calculate Spearman correlation
    corr, _ = spearmanr(ranks[test_idx], predictions)
    
    if rf_verbose > 0:
        print(f"    Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"    Correlation: {corr:.3f}")
    
    return predictions, rf.feature_importances_, corr

def evaluate_sample_nn(X, ranks, train_idx, test_idx, times=None, rf_verbose=0):
    """
    Train and evaluate model for a single sample using a neural network
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert data to tensors
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(ranks[train_idx])
    X_test = torch.FloatTensor(X[test_idx])
    y_test = torch.FloatTensor(ranks[test_idx])
    
    # Define network architecture
    class SimpleNet(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )
            
        def forward(self, x):
            return self.network(x).squeeze()
    
    # Initialize model and optimizer
    model = SimpleNet(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train the model
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    # Calculate correlation
    corr, _ = spearmanr(ranks[test_idx], predictions)
    
    if rf_verbose > 0:
        print(f"    Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"    Correlation: {corr:.3f}")
    
    # Get feature importance through permutation importance
    importance = np.zeros(X.shape[1])
    base_score = spearmanr(ranks[test_idx], predictions)[0]
    
    for i in range(X.shape[1]):
        X_permuted = X_test.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(len(X_permuted)), i]
        with torch.no_grad():
            perm_pred = model(X_permuted).numpy()
        perm_score = spearmanr(ranks[test_idx], perm_pred)[0]
        importance[i] = base_score - perm_score
    
    return predictions, importance, corr

def train_and_evaluate_model(X, ranks, samples, evaluate_func, n_splits=5, 
                           verbose=True, rf_verbose=0):
    """
    Train and evaluate model separately for each sample
    
    Parameters:
    -----------
    evaluate_func : callable
        Function to use for evaluation (either evaluate_sample_rf or evaluate_sample_nn)
    """
    predictions = np.zeros_like(ranks)
    sample_importance = {}  # Store importance per sample
    correlations = []
    
    if verbose:
        print("\nTraining models for each sample...")
    
    # Process each sample separately
    for sample in np.unique(samples):
        if verbose:
            print(f"\nProcessing sample: {sample}")
        
        mask = samples == sample
        X_sample = X[mask]
        ranks_sample = ranks[mask]
        
        if len(ranks_sample) < n_splits * 2:
            if verbose:
                print(f"Skipping sample {sample}: insufficient data")
            continue
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        sample_imp = np.zeros(X.shape[1])
        sample_correlations = []
        
        mask_indices = np.where(mask)[0]
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_sample), 1):
            if verbose:
                print(f"  Fold {fold}/{n_splits}")
            
            pred, imp, corr = evaluate_func(X_sample, ranks_sample, train_idx, test_idx, rf_verbose)
            predictions[mask_indices[test_idx]] = pred
            sample_imp += imp
            sample_correlations.append(corr)
        
        # Average importance for this sample
        sample_importance[sample] = sample_imp / n_splits
        correlations.extend(sample_correlations)
    
    if verbose:
        print("\nFinal prediction statistics:")
        print(f"  Overall range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"  Zero predictions: {np.sum(predictions == 0)} / {len(predictions)}")
        print(f"  NaN predictions: {np.sum(np.isnan(predictions))} / {len(predictions)}")
    
    return predictions, correlations, sample_importance

def create_summary_figure(adata, predictions, true_ranks, times, sample_importance, correlations):
    """
    Create a detailed summary figure with per-timepoint correlations and directional feature importance
    """
    # Get timepoints from adata
    timepoints = adata.obs['timepoint'].values
    unique_timepoints = sorted(np.unique(timepoints))
    n_timepoints = len(unique_timepoints)
    
    # Adjust figure size based on number of timepoints
    fig = plt.figure(figsize=(4*n_timepoints, 8))
    
    # Get raw colony sizes for both actual and predicted
    actual_sizes = adata.obs['log_n_counts'].values
    
    # Scale predictions to match actual sizes per sample
    scaled_predictions = np.zeros_like(predictions)
    for sample in np.unique(adata.obs['sample_key']):
        mask = adata.obs['sample_key'] == sample
        min_val = actual_sizes[mask].min()
        max_val = actual_sizes[mask].max()
        scaled_predictions[mask] = min_val + predictions[mask] * (max_val - min_val)
    
    # Panel A: Prediction accuracy split by timepoint
    for i, t in enumerate(unique_timepoints, 1):
        ax = plt.subplot(2, n_timepoints, i)
        mask = timepoints == t
        
        # Calculate correlation for this timepoint
        time_corr, _ = spearmanr(actual_sizes[mask], scaled_predictions[mask])
        
        # Plot using actual colony sizes
        ax.scatter(actual_sizes[mask], scaled_predictions[mask], alpha=0.1, c='blue')
        min_val = min(actual_sizes[mask].min(), scaled_predictions[mask].min())
        max_val = max(actual_sizes[mask].max(), scaled_predictions[mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Actual Log Colony Size')
        ax.set_ylabel('Predicted Log Colony Size' if i == 1 else '')
        ax.set_title(f'Day {t}\nr = {time_corr:.2f}')
    
    
    # Panel B: Directional Feature Importance
    ax = plt.subplot(2, 1, 2)
    n_factors = len(next(iter(sample_importance.values())))
    importance_df = pd.DataFrame(sample_importance).T
    importance_df.columns = [f'NMF_{i}' for i in range(n_factors)]
    
    # Get timepoints for each sample
    sample_timepoint_dict = dict(zip(adata.obs['sample_key'], adata.obs['timepoint']))
    importance_df['timepoint'] = importance_df.index.map(sample_timepoint_dict)
    
    # Calculate mean importance and correlation direction per timepoint
    mean_importance_by_time = []
    correlation_signs = []
    
    for t in unique_timepoints:
        # Get samples for this timepoint
        time_mask = timepoints == t
        
        # Calculate mean importance
        time_data = importance_df[importance_df['timepoint'] == t].drop('timepoint', axis=1)
        mean_importance = time_data.mean()
        mean_importance_by_time.append(mean_importance)
        
        # Calculate correlation direction for each factor
        factor_correlations = []
        for f in range(n_factors):
            corr, _ = spearmanr(adata.obsm['X_nmf'][time_mask, f], 
                               true_ranks[time_mask])
            factor_correlations.append(corr)
        correlation_signs.append(factor_correlations)
    
    # Create temporal importance heatmap with correlation signs
    temporal_importance = np.array(mean_importance_by_time)
    correlation_signs = np.array(correlation_signs)
    
    # Sort factors by overall absolute importance
    abs_importance = np.abs(temporal_importance).mean(axis=0)
    factor_order = abs_importance.argsort()[::-1]
    sorted_factors = [f'NMF_{factor_order[i]}' for i in range(n_factors)]
    
    # Multiply importance by correlation sign
    signed_importance = temporal_importance * np.sign(correlation_signs)
    
    # Create DataFrame for seaborn heatmap
    heatmap_df = pd.DataFrame(signed_importance[:, factor_order],
                             index=[f'Day {t}' for t in unique_timepoints],
                             columns=sorted_factors)
    
    # Create heatmap
    sns.heatmap(heatmap_df, ax=ax, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Signed Feature Importance'})
    
    ax.set_xlabel('NMF Factors')
    ax.set_title('Feature Importance Across Time\n(red = positive correlation with colony size)')
    
    plt.tight_layout()
    return fig



def create_summary_figure_nmf(adata, predictions, true_values, times, sample_importance, correlations, target_factor_index):
    """
    Create a summary figure to visualize NMF factor prediction results with signed feature importance,
    excluding the target factor from the labels.
    
    Parameters:
    -----------
    adata : AnnData
        Input annotated data object (used for timepoints and sample metadata).
    predictions : np.ndarray
        Predicted values of the NMF factor.
    true_values : np.ndarray
        True values of the NMF factor being predicted.
    times : np.ndarray
        Timepoint identifiers for each observation.
    sample_importance : dict
        Feature importance per sample.
    correlations : list
        Spearman correlations for each sample.
    target_factor_index : int
        Index of the NMF factor being predicted (to exclude from labels).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Summary visualization figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr

    # Extract unique timepoints
    unique_timepoints = sorted(np.unique(times))
    n_timepoints = len(unique_timepoints)
    
    # Create figure
    fig = plt.figure(figsize=(4 * n_timepoints, 8))

    # Panel A: Prediction accuracy split by timepoint
    for i, t in enumerate(unique_timepoints, 1):
        ax = plt.subplot(2, n_timepoints, i)
        mask = times == t

        # Calculate Spearman correlation for the timepoint
        time_corr, _ = spearmanr(true_values[mask], predictions[mask])

        # Scatter plot of true vs predicted values
        ax.scatter(true_values[mask], predictions[mask], alpha=0.5, c='blue')
        min_val = min(true_values[mask].min(), predictions[mask].min())
        max_val = max(true_values[mask].max(), predictions[mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)  # Diagonal line

        ax.set_xlabel('True NMF Usage')
        ax.set_ylabel('Predicted NMF Usage' if i == 1 else '')
        ax.set_title(f'Timepoint: {t}\nSpearman r = {time_corr:.2f}')

    # Panel B: Signed Feature Importance Heatmap
    ax = plt.subplot(2, 1, 2)
    n_factors = len(next(iter(sample_importance.values())))

    # Exclude target factor from NMF labels
    nmf_labels = [f'NMF_{i}' for i in range(n_factors + 1) if i != target_factor_index]
    
    # Convert sample_importance dict to DataFrame
    importance_df = pd.DataFrame(sample_importance).T
    importance_df.columns = nmf_labels

    # Add timepoint information
    sample_time_dict = dict(zip(adata.obs['sample_key'], adata.obs['timepoint']))
    importance_df['timepoint'] = importance_df.index.map(sample_time_dict)

    # Calculate mean importance for each timepoint
    mean_importance = []
    correlation_signs = []
    for t in unique_timepoints:
        time_mask = times == t
        time_data = importance_df[importance_df['timepoint'] == t].drop('timepoint', axis=1)
        mean_importance.append(time_data.mean())

        # Calculate correlation signs for each factor
        factor_corr_signs = []
        for f in range(n_factors + 1):
            if f == target_factor_index:
                continue
            corr, _ = spearmanr(adata.obsm['X_nmf'][time_mask, f], true_values[time_mask])
            factor_corr_signs.append(np.sign(corr))
        correlation_signs.append(factor_corr_signs)

    # Multiply mean importance by correlation signs to create signed importance
    signed_importance = np.array(mean_importance) * np.array(correlation_signs)

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(signed_importance, 
                              index=[f'Timepoint: {t}' for t in unique_timepoints],
                              columns=nmf_labels)

    # Plot heatmap
    sns.heatmap(heatmap_df, ax=ax, cmap='coolwarm', center=0, cbar_kws={'label': 'Signed Feature Importance'})
    ax.set_xlabel('NMF Factors (Excluding Target)')
    ax.set_title('Feature Importance by Timepoint\n(Signed by Correlation)')

    plt.tight_layout()
    return fig





def analyze_clone_sizes(adata, target_col='n_counts', sample_col='sample_key', 
                       time_col='timepoint', method='rf', verbose=True, rf_verbose=0):
    """
    Main analysis function using within-sample ranks
    
    Parameters:
    -----------
    adata : AnnData
        Input data
    target_col : str
        Column name for the target variable (default: 'n_counts')
    sample_col : str
        Column name for sample identifiers (default: 'sample_key')
    time_col : str
        Column name for timepoints (default: 'timepoint')
    method : str
        Prediction method to use: 'rf' for Random Forest or 'nn' for Neural Network
    verbose : bool
        Whether to print progress information
    rf_verbose : int
        Verbosity level for the underlying model
    """
    if method not in ['rf', 'nn']:
        raise ValueError("method must be either 'rf' or 'nn'")
    
    if verbose:
        print(f'\nUsing {method.upper()} method')
        print('Preparing data (converting to within-sample ranks):')
    
    # Prepare data
    X, ranks, samples, times = prepare_data(
        adata, target_col=target_col, sample_col=sample_col, time_col=time_col,
        verbose=verbose
    )
    
    if verbose:
        print(f'\nTraining and evaluating {method.upper()} model:')
    
    # Select evaluation function based on method
    evaluate_func = evaluate_sample_rf if method == 'rf' else evaluate_sample_nn
    
    # Train and evaluate model
    predictions, correlations, sample_importance = train_and_evaluate_model(
        X, ranks, samples, times, evaluate_func=evaluate_func,
        verbose=verbose, rf_verbose=rf_verbose
    )
    
    if verbose:
        print('\nGenerating visualization plots...')
    
    # Create visualizations
    plot = create_summary_figure(adata, predictions, ranks, times, 
                               sample_importance, correlations)
    
    # Print summary statistics
    print("\nModel Performance:")
    print(f"Median Spearman correlation: {np.median(correlations):.3f}")
    print(f"Mean Spearman correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
    
    return predictions, correlations, sample_importance, plot

def analyze_feature_importance(adata, feature_importance, samples, n_factors=None):
    """
    Analyze feature importance patterns across samples
    
    Parameters
    ----------
    adata : AnnData
        Original data object
    feature_importance : dict
        Dictionary mapping sample names to their feature importance arrays
    samples : array-like
        Array of sample names
    n_factors : int, optional
        Number of NMF factors. If None, inferred from data
    """
    if n_factors is None:
        n_factors = adata.obsm['X_nmf'].shape[1]
        
    # Create feature importance DataFrame
    importance_df = pd.DataFrame(feature_importance).T
    importance_df.columns = [f'NMF_{i}' for i in range(n_factors)]
    
    # Plot heatmap of feature importance
    plt.figure(figsize=(12, 8))
    sns.clustermap(importance_df, 
                  cmap='viridis',
                  xticklabels=True,
                  yticklabels=True,
                  col_cluster=True,
                  row_cluster=True)
    plt.title('Feature Importance Patterns Across Samples')
    
    # Calculate correlation between samples' feature importance patterns
    corr_matrix = importance_df.T.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title('Sample Similarity Based on Feature Importance Patterns')
    
    # Find most consistent important factors
    mean_importance = importance_df.mean()
    std_importance = importance_df.std()
    
    factor_stats = pd.DataFrame({
        'Mean_Importance': mean_importance,
        'Std_Importance': std_importance,
        'CV': std_importance / mean_importance
    }).sort_values('Mean_Importance', ascending=False)
    
    print("\nTop 5 Most Important Factors (averaged across samples):")
    print(factor_stats.head())
    
    print("\nMost Consistent Factors (lowest CV):")
    print(factor_stats.sort_values('CV').head())
    
    return importance_df, factor_stats


def analyze_top_factors(adata, importance_df, n_top=3):
    """
    Analyze the relationship between top factors and clone size
    
    Parameters
    ----------
    adata : AnnData
        Original data object
    importance_df : pandas.DataFrame
        DataFrame with feature importance per sample
    n_top : int
        Number of top factors to analyze
    """
    # Get top factors by mean importance
    top_factors = importance_df.mean().sort_values(ascending=False).head(n_top).index
    
    plt.figure(figsize=(15, 5))
    for i, factor in enumerate(top_factors, 1):
        plt.subplot(1, n_top, i)
        factor_idx = int(factor.split('_')[1])
        plt.scatter(adata.obsm['X_nmf'][:, factor_idx],
                   adata.obs['log_n_counts'],
                   alpha=0.1)
        plt.xlabel(factor)
        plt.ylabel('Log Clone Size')
        
        # Calculate correlation
        corr, _ = spearmanr(adata.obsm['X_nmf'][:, factor_idx],
                           adata.obs['log_n_counts'])
        plt.title(f'Correlation: {corr:.3f}')
    
    plt.tight_layout()
    return top_factors


def analyze_feature_importance_by_time(adata, feature_importance, sample_col='sample_key', time_col='timepoint'):
    """
    Create separate feature importance heatmaps for each timepoint
    """
    # Create DataFrame of feature importance
    n_factors = len(next(iter(feature_importance.values())))
    importance_df = pd.DataFrame(feature_importance).T
    importance_df.columns = [f'NMF_{i}' for i in range(n_factors)]
    
    # Get timepoint mapping
    sample_timepoint_dict = dict(zip(adata.obs[sample_col], adata.obs[time_col]))
    unique_timepoints = {}
    for sample in importance_df.index:
        # Get all timepoints for this sample name and take the first one
        timepoint = sample_timepoint_dict[sample]
        unique_timepoints[sample] = timepoint
    
    # Add timepoint information
    importance_df['timepoint'] = importance_df.index.map(unique_timepoints)
    
    # Create separate plots for each timepoint
    timepoints = sorted(importance_df['timepoint'].unique())
    n_times = len(timepoints)
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 6))
    
    for i, timepoint in enumerate(timepoints):
        time_data = importance_df[importance_df['timepoint'] == timepoint].drop('timepoint', axis=1)
        
        # Calculate mean importance for this timepoint
        mean_importance = time_data.mean()
        
        # Sort factors by importance
        factor_order = mean_importance.sort_values(ascending=False).index
        time_data = time_data[factor_order]
        
        # Create heatmap
        sns.heatmap(time_data, cmap='viridis', 
                   xticklabels=True, yticklabels=True,
                   ax=axes[i])
        axes[i].set_title(f'Day {timepoint}')
        
        # Print top factors for this timepoint
        print(f"\nTop 5 factors for Day {timepoint}:")
        print(mean_importance[factor_order].head())
    
    plt.tight_layout()
    
    # Create summary barplot
    plt.figure(figsize=(12, 5))
    summary_data = []
    
    for timepoint in timepoints:
        time_data = importance_df[importance_df['timepoint'] == timepoint].drop('timepoint', axis=1)
        mean_importance = time_data.mean()
        
        for factor in importance_df.columns[:-1]:  # Exclude timepoint column
            summary_data.append({
                'Timepoint': f'Day {timepoint}',
                'Factor': factor,
                'Importance': mean_importance[factor]
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plot
    plt.figure(figsize=(12, 5))
    sns.barplot(data=summary_df, x='Factor', y='Importance', hue='Timepoint')
    plt.xticks(rotation=45)
    plt.title('Feature Importance by Timepoint')
    plt.tight_layout()
    
    return importance_df




def analyze_nmf_prediction(adata, target_factor_index, sample_col='sample_key', time_col='timepoint', 
                           method='nn', verbose=True, rf_verbose=0, **kwargs):
    """
    Analyze and predict one NMF factor using the others without modifying the original adata object.
    
    Parameters:
    -----------
    adata : AnnData
        Input annotated data object containing the NMF factors in adata.obsm['X_nmf'].
    target_factor_index : int
        Index of the NMF factor to predict.
    sample_col : str
        Column name for sample identifiers (default: 'sample_key').
    time_col : str
        Column name for timepoints (default: 'timepoint').
    method : str
        Prediction method to use: 'rf' for Random Forest or 'nn' for Neural Network.
    verbose : bool
        Whether to print progress information.
    rf_verbose : int
        Verbosity level for the underlying Random Forest model.
    **kwargs : dict
        Additional parameters for the model evaluation function.
    
    Returns:
    --------
    predictions, correlations, sample_importance, plot : tuple
        Outputs of the prediction and evaluation pipeline.
    """
    if 'X_nmf' not in adata.obsm:
        raise ValueError("NMF factors not found in `adata.obsm['X_nmf']`.")

    if verbose:
        print(f"Predicting NMF factor {target_factor_index} using the others...")

    # Extract NMF matrix
    nmf_factors = adata.obsm['X_nmf']
    
    # Validate target factor index
    if target_factor_index < 0 or target_factor_index >= nmf_factors.shape[1]:
        raise ValueError(f"Invalid target_factor_index {target_factor_index}. Must be in range [0, {nmf_factors.shape[1] - 1}].")

    # Separate target factor and remaining factors
    target = nmf_factors[:, target_factor_index]
    predictors = np.delete(nmf_factors, target_factor_index, axis=1)

    # Extract sample and time information
    samples = adata.obs[sample_col].values
    times = adata.obs[time_col].values
    
    # Select the evaluation function based on the chosen method
    evaluate_func = evaluate_sample_rf if method == 'rf' else evaluate_sample_nn
    # Train and evaluate the model
    predictions, correlations, sample_importance = train_and_evaluate_model(
        predictors, target, samples, times, evaluate_func=evaluate_func, verbose=verbose, rf_verbose=rf_verbose, **kwargs
    )

    plot = create_summary_figure_nmf(adata, predictions, target, times, sample_importance, correlations, target_factor_index)

    # Print summary statistics
    print("\nModel Performance:")
    print(f"Median Spearman correlation: {np.median(correlations):.3f}")
    print(f"Mean Spearman correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")

    return predictions, correlations, sample_importance, plot


def evaluate_sample_adversarial(X, ranks, train_idx, test_idx, times, rf_verbose=0):
    """
    Train and evaluate model using adversarial approach
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # Convert data to tensors
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(ranks[train_idx])
    times_train = torch.FloatTensor(times[train_idx])
    X_test = torch.FloatTensor(X[test_idx])
    
    class Predictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.output = nn.Linear(32, 1)
            
        def forward(self, x):
            features = self.features(x)
            return self.output(features).squeeze(), features

    class Discriminator(nn.Module):
        def __init__(self, feature_size=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.net(x).squeeze()

    predictor = Predictor(X_train.shape[1])
    discriminator = Discriminator()
    
    pred_optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    
    pred_criterion = nn.MSELoss()
    disc_criterion = nn.BCELoss()
    
    # Create data loader
    train_dataset = TensorDataset(X_train, y_train, times_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    lambda_adv = 0.1  # Weight for adversarial loss
    
    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        for batch_X, batch_y, batch_times in train_loader:
            # Train discriminator
            disc_optimizer.zero_grad()
            _, features = predictor(batch_X)
            time0_labels = (batch_times == 0).float()
            disc_pred = discriminator(features.detach())
            disc_loss = disc_criterion(disc_pred, time0_labels)
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train predictor
            pred_optimizer.zero_grad()
            predictions, features = predictor(batch_X)
            pred_loss = pred_criterion(predictions, batch_y)
            
            # Adversarial loss: try to fool discriminator for non-time0 samples
            disc_pred = discriminator(features)
            adv_loss = disc_criterion(disc_pred, torch.zeros_like(disc_pred))
            
            # Combined loss: prediction accuracy - λ * discriminator fooling
            total_loss = pred_loss - lambda_adv * adv_loss
            total_loss.backward()
            pred_optimizer.step()
    
    # Make predictions
    predictor.eval()
    with torch.no_grad():
        predictions, _ = predictor(X_test)
        predictions = predictions.numpy()
    
    # Calculate correlation
    corr, _ = spearmanr(ranks[test_idx], predictions)
    
    # Feature importance through permutation
    importance = np.zeros(X.shape[1])
    base_score = corr
    for i in range(X.shape[1]):
        X_permuted = X_test.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(len(X_permuted)), i]
        with torch.no_grad():
            perm_pred, _ = predictor(X_permuted)
        perm_score = spearmanr(ranks[test_idx], perm_pred.numpy())[0]
        importance[i] = base_score - perm_score
    
    return predictions, importance, corr