import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict

def normalize_nmf(adata):
    """Normalize NMF factors per cell"""
    norm_factors = adata.obsm['X_nmf'] / adata.obsm['X_nmf'].sum(axis=1)[:, np.newaxis]
    adata.obsm['X_nmfnm'] = norm_factors

def plot_nmf_correlation(adata, 
                         filt_column = ['timepoint','condition'], 
                         filt_values = [0,'Ctrl'], figsize=(3,3)):
    """Plot correlation heatmap for filtered cells"""
    # # Print available values for debugging
    # for col in filt_column:
    #     if col not in adata.obs.columns:
    #         raise ValueError(f"Column '{col}' not found in adata.obs. Available columns: {adata.obs.columns.tolist()}")
    #     print(f"Available values in {col}: {adata.obs[col].unique()}")
    
    # Filter cells
    mask = np.ones(len(adata), dtype=bool)
    for col, val in zip(filt_column, filt_values):
        current_mask = adata.obs[col] == val
        print(f"Number of cells matching {col}={val}: {current_mask.sum()}")
        mask &= current_mask
    
    print(f"Total number of cells after all filters: {mask.sum()}")
    
    # Check if we have any valid data
    if mask.sum() == 0:
        raise ValueError(f"No data points match the filter criteria: {dict(zip(filt_column, filt_values))}")
    
    # Get correlation matrix
    data = adata.obsm['X_nmfnm'][mask,:]
    corr = np.corrcoef(data.T)
    
    # Replace any NaN or infinite values
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Plot
    plt.figure(figsize=figsize)
    g = sns.clustermap(corr, cmap='RdBu_r', center=0, 
                xticklabels=range(corr.shape[0]),
                yticklabels=range(corr.shape[0]),
                dendrogram_ratio=(0.1, 0.1),  # Make dendrograms smaller (default is 0.2)
                cbar_pos=(1.02, 0.2, 0.03, 0.6))  # (left, bottom, width, height)
    g.ax_cbar.set_ylabel('Correlation',fontsize=20)
    g.figure.suptitle(f"NMF Factor Correlations\n{dict(zip(filt_column, filt_values))}")
    return g.figure

def get_factor_stats(adata, f1_set=[3], f2_set=[4], condition_col='condition',
                     threshold1=0.2, threshold2=0.2) -> Dict[str, pd.DataFrame]:
    """Calculate factor statistics over time for different conditions
    
    Args:
        adata: AnnData object
        f1_set: List of factor indices for first set
        f2_set: List of factor indices for second set
        condition_col: Column name for condition in adata.obs
        threshold1: Threshold for first factor set
        threshold2: Threshold for second factor set
    """
    results = {}
    timepoints = sorted(adata.obs['timepoint'].unique())
    
    for condition in adata.obs[condition_col].unique():
        data = []
        for time in timepoints:
            # Filter cells
            mask = (adata.obs[condition_col] == condition) & \
                   (adata.obs['timepoint'] == time)
            
            if not np.any(mask):
                continue
                
            factors = adata.obsm['X_nmfnm'][mask]
            # Sum factors within each set
            f1_vals = factors[:, f1_set].sum(axis=1)
            f2_vals = factors[:, f2_set].sum(axis=1)
            
            # Calculate statistics
            f1_high = ((f1_vals > threshold1) & (f2_vals <= threshold2)).mean() 
            f2_high = ((f2_vals > threshold2) & (f1_vals <= threshold1)).mean() 
            both_high = ((f1_vals > threshold1) & 
                        (f2_vals > threshold2)).mean() 
            both_low = ((f1_vals <= threshold1) & 
                       (f2_vals <= threshold2)).mean() 
            
            # Correlation
            corr, _ = stats.pearsonr(f1_vals, f2_vals)
            
            # Fisher's exact test
            contingency = np.array([
                [(f1_vals > threshold1) & (f2_vals > threshold2),
                 (f1_vals > threshold1) & (f2_vals <= threshold2)],
                [(f1_vals <= threshold1) & (f2_vals > threshold2),
                 (f1_vals <= threshold1) & (f2_vals <= threshold2)]
            ]).sum(axis=2)
            _, fisher_p = stats.fisher_exact(contingency)
            
            data.append({
                'timepoint': time,
                'f1_hi': f1_high,
                'f2_hi': f2_high,
                'f1f2_hi': both_high,
                'f1f2_lo': both_low,
                'corr': corr,
                'fisher_p': fisher_p,
                'n_cells': mask.sum()
            })
            
        results[condition] = pd.DataFrame(data)
    
    return results

def plot_factor_timeseries(stats_dict: Dict[str, pd.DataFrame], 
                          columns=None):
    """Plot time series for each metric across conditions"""
    if columns is None:
        columns = ['f1_hi', 'f2_hi', 'f1f2_hi', 'f1f2_lo', 
                  'corr', 'fisher_p', 'n_cells']
    
    fig, axes = plt.subplots(len(columns), 1, figsize=(3.5, 3*len(columns)))
    
    for i, col in enumerate(columns):
        ax = axes[i]
        for condition, df in stats_dict.items():
            if col == 'fisher_p':
                y = df[col].apply(lambda x: -np.log10(x))
                ylabel = '-log10(p)'
            else:
                y = df[col]
                ylabel = col
            ax.plot(df['timepoint'], y, 'o-', label=condition)
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.legend()
        
    plt.tight_layout()
    return fig

import numpy as np



def find_essential_factor_combinations(adata, thresh, min_frac=0.99):
    """Find essential factor combinations that are present in at least min_frac of cells
    
    Args:
        adata: AnnData object containing NMF results
        thresh: threshold for binarizing NMF values
        min_frac: minimum fraction of cells that must have the factor/combination (default: 0.99)
    """
    # Binarize NMF data
    nmf_binary = (adata.obsm['X_nmfnm'] > thresh).astype(int)
    adata.obsm['X_nmfbin'] = nmf_binary
    
    n_factors = nmf_binary.shape[1]
    essential_singles = []
    essential_pairs = []
    essential_triplets = []
    
    # Find single essential factors
    for i in range(n_factors):
        if np.mean(nmf_binary[:, i]) > min_frac:
            essential_singles.append(i)
    
    # Remove essential singles from consideration
    remaining_factors = list(set(range(n_factors)) - set(essential_singles))
    
    # Find essential pairs
    for i in range(len(remaining_factors)):
        for j in range(i + 1, len(remaining_factors)):
            factor1, factor2 = remaining_factors[i], remaining_factors[j]
            pair_sum = nmf_binary[:, factor1] + nmf_binary[:, factor2]
            if np.mean(pair_sum > 0) > min_frac:
                essential_pairs.append((factor1, factor2))
    
    # Remove factors in essential pairs
    used_in_pairs = set([x for pair in essential_pairs for x in pair])
    remaining_for_triplets = list(set(remaining_factors) - used_in_pairs)
    
    # Find essential triplets
    for i in range(len(remaining_for_triplets)):
        for j in range(i + 1, len(remaining_for_triplets)):
            for k in range(j + 1, len(remaining_for_triplets)):
                f1, f2, f3 = remaining_for_triplets[i], remaining_for_triplets[j], remaining_for_triplets[k]
                triplet_sum = nmf_binary[:, f1] + nmf_binary[:, f2] + nmf_binary[:, f3]
                if np.mean(triplet_sum > 0) > min_frac:
                    essential_triplets.append((f1, f2, f3))
    
    return {
        'singles': essential_singles,
        'pairs': essential_pairs,
        'triplets': essential_triplets
    }