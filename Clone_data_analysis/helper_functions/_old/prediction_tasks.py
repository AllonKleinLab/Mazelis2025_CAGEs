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

def evaluate_sample(X, ranks, train_idx, test_idx, rf_verbose=0):
    """
    Train and evaluate model for a single sample
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

def train_and_evaluate_model(X, ranks, samples, times, n_splits=5, verbose=True, rf_verbose=0):
    """
    Train and evaluate model separately for each sample, storing per-sample feature importance
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
            
            pred, imp, corr = evaluate_sample(X_sample, ranks_sample, train_idx, test_idx, rf_verbose)
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



def plot_results(true_ranks, predictions, times, sample_importance, n_factors, correlations):
    """
    Create visualization of results
    
    Parameters
    ----------
    true_ranks : array-like
        True rank values
    predictions : array-like
        Predicted rank values
    times : array-like
        Timepoint values
    sample_importance : dict
        Dictionary of feature importance per sample
    n_factors : int
        Number of NMF factors
    correlations : list
        List of correlation values
    """
    # Average feature importance across samples
    feature_importance = np.mean([imp for imp in sample_importance.values()], axis=0)
    
    plt.figure(figsize=(15, 10))
    
    # Plot predicted vs actual ranks
    plt.subplot(221)
    plt.scatter(true_ranks, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Rank (normalized)')
    plt.ylabel('Predicted Rank (normalized)')
    plt.title(f'Predicted vs Actual Ranks\n(n={len(predictions)}, zeros={np.sum(predictions == 0)})')
    
    # Plot feature importance
    plt.subplot(222)
    factor_names = [f'NMF_{i}' for i in range(n_factors)]
    importance_df = pd.DataFrame({
        'Factor': factor_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Factor'])
    plt.xlabel('Feature Importance')
    plt.title('NMF Factor Importance')
    
    # Plot correlation distribution
    plt.subplot(223)
    plt.hist(correlations, bins=20)
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Count')
    plt.title(f'Distribution of Prediction Performance\nMean={np.mean(correlations):.3f}')
    
    # Plot predictions by timepoint
    plt.subplot(224)
    for t in np.unique(times):
        mask = times == t
        plt.scatter(true_ranks[mask], predictions[mask], alpha=0.5, label=f'Time {t}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Rank (normalized)')
    plt.ylabel('Predicted Rank (normalized)')
    plt.title('Predictions by Timepoint')
    plt.legend()
    
    plt.tight_layout()
    return plt

def analyze_clone_sizes(adata, target_col='n_counts', sample_col='sample_key', 
                       time_col='timepoint', verbose=True, rf_verbose=0):
    """
    Main analysis function using within-sample ranks
    
    Returns
    -------
    predictions : array-like
        Predicted rank values
    correlations : list
        List of correlation values
    sample_importance : dict
        Dictionary of feature importance per sample
    plot : matplotlib.figure.Figure
        Visualization plot
    """
    if verbose:
        print('\nPreparing data (converting to within-sample ranks):')
    
    # Prepare data
    X, ranks, samples, times = prepare_data(
        adata, target_col=target_col, sample_col=sample_col, time_col=time_col,
        verbose=verbose
    )
    
    if verbose:
        print('\nTraining and evaluating Random Forest model:')
    
    # Train and evaluate model
    predictions, correlations, sample_importance = train_and_evaluate_model(
        X, ranks, samples, times, verbose=verbose, rf_verbose=rf_verbose
    )
    
    if verbose:
        print('\nGenerating visualization plots...')
    
    # Create visualizations
    plot = plot_results(ranks, predictions, times, 
                       sample_importance, X.shape[1],
                       correlations)
    
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




def create_summary_figure(adata, predictions, true_ranks, times, sample_importance, correlations):
    """
    Create a detailed summary figure with per-timepoint correlations and directional feature importance
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Get timepoints from adata
    timepoints = adata.obs['timepoint'].values
    unique_timepoints = sorted(np.unique(timepoints))
    
    # Panel A: Prediction accuracy split by timepoint
    for i, t in enumerate(unique_timepoints, 1):
        ax = plt.subplot(2, len(unique_timepoints), i)
        mask = timepoints == t
        
        # Calculate correlation for this timepoint
        time_corr, _ = spearmanr(true_ranks[mask], predictions[mask])
        
        ax.scatter(true_ranks[mask], predictions[mask], alpha=0.02, c='k')
        ax.plot([0, 1], [0, 1], 'r-', alpha=1.0, linewidth=1)
        ax.set_xlabel('Actual Relative Colony Size')
        ax.set_ylabel('Predicted Colony Size' if i == 1 else '')
        ax.set_title(f'Day {t}\nr = {time_corr:.2f}')
    
    # Panel B: Directional Feature Importance
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
    ax = plt.subplot(2, 1, 2)
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