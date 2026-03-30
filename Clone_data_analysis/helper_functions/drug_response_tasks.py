import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds




#########

def calculate_lfc(adata, timepoint, drug, control='Ctrl-1'):
    """
    Calculate Wilcoxon test statistics and log2 fold changes between drug and control conditions.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    timepoint : int/str
        Timepoint to analyze
    drug : str
        Name of drug condition
    control : str
        Name of control condition
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing gene names, log2FC, p-values, and adjusted p-values
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    # Subset data
    mask = (adata.obs['timepoint'] == timepoint) & (
        (adata.obs['condition'] == drug) | 
        (adata.obs['condition'] == control)
    )
    subset = adata[mask].copy()
    
    # Get expression data
    X = subset.layers['log'].toarray() if sparse.issparse(subset.layers['log']) else subset.layers['log']
    conditions = subset.obs['condition']
    
    drug_mask = conditions == drug
    control_mask = conditions == control
    
    # Initialize results storage
    results = []
    
    # Calculate statistics for each gene
    for idx, gene in enumerate(subset.var_names):
        drug_expr = X[drug_mask, idx]
        ctrl_expr = X[control_mask, idx]
        
        # Calculate mean difference and convert to log2FC
        mean_diff = np.mean(drug_expr) - np.mean(ctrl_expr)
        log2fc = mean_diff * np.log2(10)  # Convert from log10 to log2
        
        # Calculate Wilcoxon statistic and p-value
        stat, pval = stats.ranksums(drug_expr, ctrl_expr)
        
        # Calculate percent non-zero in drug group
        pct_nz = np.mean(drug_expr > 0)
        
        results.append({
            'gene': gene,
            'score': stat,
            'log2fc': log2fc,
            'pval': pval,
            'pct_nz_group': pct_nz
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # BH correction
    results_df['padj'] = multipletests(results_df['pval'], method='fdr_bh')[1]
    
    # Sort by absolute log fold change
    results_df = results_df.sort_values('score', ascending=False)
    
    return results_df

#######


# def perform_differential_expression(adata, timepoint, drug, control='Ctrl'):
#     """
#     Perform differential expression analysis using scanpy's built-in rank_genes_groups
#     """
#     # Subset data for specific timepoint and conditions
#     mask = (adata.obs['timepoint'] == timepoint) & (
#         (adata.obs['condition'] == drug) | (adata.obs['condition'] == control)
#     )
#     subset = adata[mask].copy()
    
#     # Run differential expression
#     sc.tl.rank_genes_groups(subset, 'condition', groups=[drug], 
#                            reference=control, method='t-test_overestim_var',
#                            pts=True,  # Calculate percentage of cells expressing genes
#                            layer='log'
#                            )
    
#     # Get results dataframe
#     results_df = sc.get.rank_genes_groups_df(subset, group=drug)
    
#     # Rename columns to match existing pipeline
#     results_df = results_df.rename(columns={
#         'names': 'gene',
#         'logfoldchanges': 'log2fc',
#         'pvals': 'pval',
#         'pvals_adj': 'padj',
#         'scores': 'stat'
#     })
    
#     return results_df

def plot_volcano(results_df, drug, timepoint, padj_threshold=0.05, fc_threshold=1, pts_threshold=0.05):
    """
    Create volcano plot for differential expression results
    """
    plt.figure(figsize=(4, 4))
    plt.scatter(results_df['log2fc'], -np.log10(results_df['padj']), alpha=0.5)
    
    # Highlight significant genes
    sig_genes = results_df[
        (results_df['padj'] < padj_threshold) & 
        (abs(results_df['log2fc']) > fc_threshold) &
        (results_df['pct_nz_group'] > pts_threshold)
    ]
    plt.scatter(sig_genes['log2fc'], -np.log10(sig_genes['padj']), 
                color='red', alpha=0.5)
    
    plt.axhline(-np.log10(padj_threshold), color='gray', linestyle='--')
    plt.axvline(fc_threshold, color='gray', linestyle='--')
    plt.axvline(-fc_threshold, color='gray', linestyle='--')
    
    plt.xlabel('log2 Fold Change')
    plt.ylabel('-log10 Adjusted P-value')
    plt.title(f'Volcano Plot: {drug} Timepoint {timepoint}')
    return plt

def assess_clone_heterogeneity(adata, sig_genes, drug, timepoint, verbose=False, showPlot=False):
   """
   Assess heterogeneity of significant genes using scanpy's variance normalization
   """
   # Subset data for specific condition and timepoint
   mask = (adata.obs['condition'] == drug) & (adata.obs['timepoint'] == timepoint)
   subset = adata[mask].copy()
   
   # Calculate normalized variances using seurat_v3 method
   sc.pp.highly_variable_genes(subset, 
                             layer='raw',
                             flavor='seurat_v3',
                             n_top_genes=None,
                             span=0.8
                             )
   
   # Get comprehensive metrics for all genes
   var_df = pd.DataFrame({
       'gene': subset.var_names,
       'mean': subset.var['means'],
       'variance': subset.var['variances'],
       'variance_norm': subset.var['variances_norm'],
       'highly_variable': subset.var['highly_variable'],
       'DEG': subset.var_names.isin(sig_genes)
   })
   
   # Calculate percentile ranks for variances
   var_df['variance_percentile'] = var_df['variance_norm'].rank(pct=True) * 100
   
   # Fisher exact test for enrichment
   contingency = pd.crosstab(var_df['DEG'], var_df['highly_variable'])
   odds_ratio, pvalue = stats.fisher_exact(contingency)

   if verbose:
       print("\nFisher's Exact Test Results:")
       print("Contingency table:")
       print(contingency)
       print(f"Odds ratio: {odds_ratio:.2f}")
       print(f"P-value: {pvalue:.2e}")
   
   # Create plot
   plt.figure(figsize=(4, 4))
   
   def safe_log10(x):
       min_pos = np.min(x[x > 0]) / 10 if np.any(x > 0) else 1e-10
       x_safe = np.where(x > 0, x, min_pos)
       return np.log10(x_safe)
   
   # Plot histogram of non-DEGs
   if showPlot:
       non_deg_mask = ~var_df['DEG']
       valid_variances = var_df.loc[non_deg_mask, 'variance_norm'][var_df.loc[non_deg_mask, 'variance_norm'] > 0]
       if len(valid_variances) > 0:
           plt.hist(safe_log10(valid_variances), 
                   bins=50, alpha=0.5, 
                   label=f'Non-DEGs (n={len(valid_variances)})', 
                   density=True)
       
       # Overlay DEGs
       deg_mask = var_df['DEG']
       valid_deg_variances = var_df.loc[deg_mask, 'variance_norm'][var_df.loc[deg_mask, 'variance_norm'] > 0]
       if len(valid_deg_variances) > 0:
           plt.hist(safe_log10(valid_deg_variances), 
                   bins=50, alpha=0.5, 
                   label=f'DEGs (n={len(valid_deg_variances)})', 
                   density=True)
       
       plt.xlabel('log10(Normalized Variance)')
       plt.ylabel('Density')
       plt.title(f'Clone Heterogeneity: {drug} Timepoint {timepoint}')
       plt.legend()
   
   # Get list of genes both DE and highly variable
   highly_variable_sig_genes = var_df[
       var_df['DEG'] & var_df['highly_variable']
   ]['gene'].tolist()
   
   # Print summary statistics
   if verbose:
       print("\nSummary Statistics:")
       print(f"Total genes: {len(var_df)}")
       print(f"DEGs: {sum(var_df['DEG'])}")
       print(f"Highly variable genes: {sum(var_df['highly_variable'])}")
       print(f"Genes both DE and highly variable: {len(highly_variable_sig_genes)}")
   
   return {
       'plot': plt,
       'var_df': var_df,
       'highly_variable_sig_genes': highly_variable_sig_genes,
       'fisher_results': {
           'contingency': contingency,
           'odds_ratio': odds_ratio,
           'pvalue': pvalue
       }
   }

def collect_heterogeneity_results(adata, sig_genes, drugs, timepoints):
    """
    Collect heterogeneity analysis results for all drugs and timepoints
    """
    all_results = {}
    for drug in drugs:
        all_results[drug] = {}
        for timepoint in timepoints:
            all_results[drug][timepoint] = assess_clone_heterogeneity(adata, sig_genes, drug, timepoint)
    return all_results

def analyze_heterogeneity_results(all_results):
    """
    Analyze collected heterogeneity results
    
    Parameters:
    -----------
    all_results : dict
        Nested dictionary of results from collect_heterogeneity_results
        
    Returns:
    --------
    dict containing:
        - hvg_degs_by_drug : dict of highly variable DEGs per drug
        - summary_df : DataFrame with statistics per drug/timepoint
        - combined_plot : matplotlib figure with combined variance plot
    """
    # 1. Get union of highly variable DEGs per drug
    hvg_degs_by_drug = {}
    for drug in all_results:
        hvg_degs_all_times = set()
        for timepoint in all_results[drug]:
            hvg_degs_all_times.update(all_results[drug][timepoint]['highly_variable_sig_genes'])
        hvg_degs_by_drug[drug] = sorted(list(hvg_degs_all_times))
    
    # 2. Create summary DataFrame
    summary_data = []
    for drug in all_results:
        for timepoint in all_results[drug]:
            result = all_results[drug][timepoint]
            var_df = result['var_df']
            fisher_results = result['fisher_results']
            
            summary_data.append({
                'drug': drug,
                'timepoint': timepoint,
                'total_genes': len(var_df),
                'total_HVG': sum(var_df['highly_variable']),
                'total_DEG': sum(var_df['DEG']),
                'HVG_and_DEG': len(result['highly_variable_sig_genes']),
                'odds_ratio': fisher_results['odds_ratio'],
                'pvalue': fisher_results['pvalue']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Collect all variance values for DEGs and background
    all_deg_variances = []
    all_background_variances = []
    
    for drug in all_results:
        for timepoint in all_results[drug]:
            var_df = all_results[drug][timepoint]['var_df']
            
            # Get valid variances for DEGs
            deg_vars = var_df.loc[var_df['DEG'], 'variance_norm']
            deg_vars = deg_vars[deg_vars > 0]
            all_deg_variances.extend(deg_vars)
            
            # Get valid variances for background (non-DEGs)
            bg_vars = var_df.loc[~var_df['DEG'], 'variance_norm']
            bg_vars = bg_vars[bg_vars > 0]
            all_background_variances.extend(bg_vars)
    
    # Create combined plot with single histogram for each category
    plt.figure(figsize=(4, 4))
    
    # Combine all data into single histograms
    bins = np.linspace(
        min(np.log10(min(all_background_variances)), np.log10(min(all_deg_variances))),
        max(np.log10(max(all_background_variances)), np.log10(max(all_deg_variances))),
        50
    )
    
    # Single histogram for background genes
    plt.hist(np.log10(all_background_variances), 
            bins=100, alpha=0.5, 
            label=f'All genes', 
            density=True)
    
    # Single histogram for DEGs
    plt.hist(np.log10(all_deg_variances), 
            bins=100, alpha=0.5, 
            label=f'DEGs', 
            density=True)
    
    plt.xlabel('log10(Between-clone normalized Variance)')
    plt.ylabel('Density')
    plt.xlim(-0.75,0.75)
    plt.title('Variance Distribution of DEGs vs Background')
    plt.legend()
    
    return {
        'hvg_degs_by_drug': hvg_degs_by_drug,
        'summary_df': summary_df,
        'combined_plot': plt
    }


def determine_nmf_components(X, max_components=50):
    """
    Determine optimal number of NMF components using eigenvalue spectrum
    """
    # Calculate eigenvalue spectrum
    U, s, Vt = svds(X, k=max_components)
    
    # Generate random matrix of same size
    X_random = np.random.normal(size=X.shape)
    U_r, s_r, Vt_r = svds(X_random, k=max_components)
    
    # Find where real eigenvalues cross random
    crossing_point = np.where(s < s_r)[0][0]
    return crossing_point

def perform_nmf_analysis(adata, variable_genes, drug, timepoint, n_components=None):
    """
    Perform NMF analysis on variable genes
    """
    # Subset data
    drug_cells = (adata.obs['condition'] == drug) & (adata.obs['timepoint'] == timepoint)
    X = adata[drug_cells][:,variable_genes].X
    
    if n_components is None:
        n_components = determine_nmf_components(X)
    
    # Perform NMF
    model = NMF(n_components=n_components, init='nndsvdar', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    
    # Create results dictionary
    nmf_results = {
        'W': W,  # Cell loadings
        'H': H,  # Gene loadings
        'reconstruction_error': model.reconstruction_error_,
        'n_components': n_components
    }
    
    return nmf_results

# def main(adata, drugs=['Vorinostat', 'Decitabine', '5-azacytidine'], 
#          timepoints=[2, 4, 6]):
#     """
#     Main analysis pipeline
#     """
#     all_results = {}
#     for drug in drugs:
#         drug_results = {}
#         for timepoint in timepoints:
#             # 1. Differential Expression
#             de_results = perform_differential_expression(adata, timepoint, drug)
#             volcano_plot = plot_volcano(de_results, drug, timepoint)
            
#             # Get significant genes
#             sig_genes = de_results[
#                 (de_results['padj'] < 0.05) & 
#                 (abs(de_results['log2fc']) > 1)
#             ]['gene'].tolist()
            
#             # 2. Clone Heterogeneity
#             het_results = assess_clone_heterogeneity(
#                 adata, sig_genes, drug, timepoint
#             )
            
#             # 3. NMF Analysis for heterogeneous genes
#             if len(het_results['highly_variable_sig_genes']) > 0:
#                 nmf_results = perform_nmf_analysis(
#                     adata, 
#                     het_results['highly_variable_sig_genes'], 
#                     drug, 
#                     timepoint
#                 )
#             else:
#                 nmf_results = None
            
#             drug_results[timepoint] = {
#                 'de_results': de_results,
#                 'volcano_plot': volcano_plot,
#                 'sig_genes': sig_genes,
#                 'heterogeneity_pval': het_pval,
#                 'heterogeneity_plot': het_plot,
#                 'nmf_results': nmf_results
#             }
        
#         all_results[drug] = drug_results
    
#     return all_results


####### 




