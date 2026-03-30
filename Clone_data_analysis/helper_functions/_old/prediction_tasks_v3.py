
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(adata, target_col='log_n_counts', sample_col='sample_key', time_col='timepoint', 
                use_ranks=False, verbose=True):
    required_cols = [target_col, sample_col, time_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X = adata.obsm['X_nmf']
    values = adata.obs[target_col].values
    samples = adata.obs[sample_col].values
    times = adata.obs[time_col].values
    
    if verbose:
        print(f"Found {adata.shape[0]} cells/clones and {X.shape[1]} NMF factors")
    
    if use_ranks:
        if verbose:
            print(f"Converting {target_col} to within-sample ranks...")
            
        ranks = np.zeros(len(adata))
        for sample in np.unique(samples):
            mask = samples == sample
            ranks[mask] = pd.Series(values[mask]).rank(method='average').values
            ranks[mask] = (ranks[mask] - 1) / (np.sum(mask) - 1)
            
            if verbose:
                print(f"\nSample {sample}:")
                print(f"  Points: {np.sum(mask)}")
                print(f"  Range: [{values[mask].min():.3f}, {values[mask].max():.3f}]")
        values = ranks
    
    return X, values, samples, times

def evaluate_sample_rf(X, values, train_idx, test_idx, times=None, rf_verbose=0):
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5,
                             n_jobs=-1, verbose=rf_verbose, random_state=42)
    
    rf.fit(X[train_idx], values[train_idx])
    predictions = rf.predict(X[test_idx])
    corr, _ = spearmanr(values[test_idx], predictions)
    
    return predictions, rf.feature_importances_, corr

def evaluate_sample_nn(X, values, train_idx, test_idx, times=None, rf_verbose=0):
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(values[train_idx])
    X_test = torch.FloatTensor(X[test_idx])
    
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
    
    model = SimpleNet(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    importance = np.zeros(X.shape[1])
    base_score = spearmanr(values[test_idx], predictions)[0]
    for i in range(X.shape[1]):
        X_permuted = X_test.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(len(X_permuted)), i]
        perm_pred = model(X_permuted).numpy()
        perm_score = spearmanr(values[test_idx], perm_pred)[0]
        importance[i] = base_score - perm_score
    
    corr, _ = spearmanr(values[test_idx], predictions)
    return predictions, importance, corr

def evaluate_sample_adversarial(X, values, train_idx, test_idx, times, rf_verbose=0):
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(values[train_idx])
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
    
    train_loader = DataLoader(TensorDataset(X_train, y_train, times_train), 
                            batch_size=32, shuffle=True)
    
    lambda_adv = 0.1
    
    for epoch in range(100):
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
            
            disc_pred = discriminator(features)
            adv_loss = disc_criterion(disc_pred, torch.zeros_like(disc_pred))
            
            total_loss = pred_loss - lambda_adv * adv_loss
            total_loss.backward()
            pred_optimizer.step()
    
    predictor.eval()
    with torch.no_grad():
        predictions, _ = predictor(X_test)
        predictions = predictions.numpy()
    
    corr, _ = spearmanr(values[test_idx], predictions)
    
    importance = np.zeros(X.shape[1])
    base_score = corr
    for i in range(X.shape[1]):
        X_permuted = X_test.clone()
        X_permuted[:, i] = X_permuted[torch.randperm(len(X_permuted)), i]
        with torch.no_grad():
            perm_pred, _ = predictor(X_permuted)
        perm_score = spearmanr(values[test_idx], perm_pred.numpy())[0]
        importance[i] = base_score - perm_score
    
    return predictions, importance, corr

def train_and_evaluate_model(X, values, samples, times, evaluate_func, n_splits=5, 
                           use_ranks=False, verbose=True, rf_verbose=0):
    predictions = np.zeros_like(values)
    sample_importance = {}
    correlations = []
    
    for sample in np.unique(samples):
        if verbose:
            print(f"Training model for sample {sample}...")
        mask = samples == sample
        X_sample = X[mask]
        values_sample = values[mask]
        
        if len(values_sample) < n_splits * 2:
            if verbose:
                print(f"Skipping sample {sample}: insufficient data")
            continue
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        sample_imp = np.zeros(X.shape[1])
        sample_correlations = []
        mask_indices = np.where(mask)[0]
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_sample), 1):
            if verbose:
                print(f"Fold {fold} of {n_splits}...")
            pred, imp, corr = evaluate_func(X_sample, values_sample, train_idx, test_idx, 
                                          times=times[mask_indices], rf_verbose=rf_verbose)
            predictions[mask_indices[test_idx]] = pred
            sample_imp += imp
            sample_correlations.append(corr)
        
        sample_importance[sample] = sample_imp / n_splits
        correlations.extend(sample_correlations)
    
    return predictions, correlations, sample_importance

def create_summary_figure(adata, predictions, true_values, times, 
                        sample_importance, correlations, use_ranks=False):
    timepoints = adata.obs['timepoint'].values
    unique_timepoints = sorted(np.unique(timepoints))
    n_timepoints = len(unique_timepoints)
    
    fig = plt.figure(figsize=(4*n_timepoints, 8))
    
    if use_ranks:
        scaled_predictions = np.zeros_like(predictions)
        actual_sizes = adata.obs['log_n_counts'].values
        for sample in np.unique(adata.obs['sample_key']):
            mask = adata.obs['sample_key'] == sample
            min_val = actual_sizes[mask].min()
            max_val = actual_sizes[mask].max()
            scaled_predictions[mask] = min_val + predictions[mask] * (max_val - min_val)
    else:
        scaled_predictions = predictions
        actual_sizes = true_values
    
    for i, t in enumerate(unique_timepoints, 1):
        ax = plt.subplot(2, n_timepoints, i)
        mask = timepoints == t
        time_corr, _ = spearmanr(actual_sizes[mask], scaled_predictions[mask])
        
        ax.scatter(actual_sizes[mask], scaled_predictions[mask], alpha=0.1, c='blue')
        min_val = min(actual_sizes[mask].min(), scaled_predictions[mask].min())
        max_val = max(actual_sizes[mask].max(), scaled_predictions[mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Actual Log Colony Size')
        ax.set_ylabel('Predicted Log Colony Size' if i == 1 else '')
        ax.set_title(f'Day {t}\nr = {time_corr:.2f}')
    
    ax = plt.subplot(2, 1, 2)
    importance_df = pd.DataFrame(sample_importance).T
    importance_df.columns = [f'NMF_{i}' for i in range(len(next(iter(sample_importance.values()))))]
    
    sample_timepoint_dict = dict(zip(adata.obs['sample_key'], adata.obs['timepoint']))
    importance_df['timepoint'] = importance_df.index.map(sample_timepoint_dict)
    
    mean_importance_by_time = []
    correlation_signs = []
    
    for t in unique_timepoints:
        time_mask = timepoints == t
        time_data = importance_df[importance_df['timepoint'] == t].drop('timepoint', axis=1)
        mean_importance = time_data.mean()
        mean_importance_by_time.append(mean_importance)
        
        factor_correlations = []
        for f in range(importance_df.shape[1]-1):
            corr, _ = spearmanr(adata.obsm['X_nmf'][time_mask, f], true_values[time_mask])
            factor_correlations.append(corr)
        correlation_signs.append(factor_correlations)
    
    temporal_importance = np.array(mean_importance_by_time)
    correlation_signs = np.array(correlation_signs)
    
    abs_importance = np.abs(temporal_importance).mean(axis=0)
    factor_order = abs_importance.argsort()[::-1]
    sorted_factors = [f'NMF_{factor_order[i]}' for i in range(importance_df.shape[1]-1)]
    
    signed_importance = temporal_importance * np.sign(correlation_signs)
    
    heatmap_df = pd.DataFrame(signed_importance[:, factor_order],
                             index=[f'Day {t}' for t in unique_timepoints],
                             columns=sorted_factors)
    
    sns.heatmap(heatmap_df, ax=ax, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Signed Feature Importance'})
    
    ax.set_xlabel('NMF Factors')
    ax.set_title('Feature Importance Across Time\n(red = positive correlation with colony size)')
    
    plt.tight_layout()
    return fig

def analyze_clone_sizes(adata, target_col='log_n_counts', sample_col='sample_key', 
                       time_col='timepoint', method='rf', use_ranks=False,
                       verbose=True, rf_verbose=0):
    if method not in ['rf', 'nn', 'adversarial']:
        raise ValueError("method must be one of: 'rf', 'nn', 'adversarial'")
    
    evaluate_func = {
        'rf': evaluate_sample_rf,
        'nn': evaluate_sample_nn,
        'adversarial': evaluate_sample_adversarial
    }[method]
    
    if verbose:
        print(f'\nUsing {method.upper()} method')
    
    X, values, samples, times = prepare_data(
        adata, target_col=target_col, sample_col=sample_col, time_col=time_col,
        use_ranks=use_ranks, verbose=verbose
    )
    
    if verbose:
        print('Training and evaluating model...')
    predictions, correlations, sample_importance = train_and_evaluate_model(
        X, values, samples, times, evaluate_func=evaluate_func,
        use_ranks=use_ranks, verbose=verbose, rf_verbose=rf_verbose
    )
    
    plot = create_summary_figure(adata, predictions, values, times, 
                               sample_importance, correlations,
                               use_ranks=use_ranks)
    
    if verbose:
        print(f"\nModel Performance:")
        print(f"Mean correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
    
    return predictions, correlations, sample_importance