# Functions used in processing memory-seq data sets from Ignas Mazelis' work
# 
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import glob
import re
import gc

from sklearn.decomposition import NMF

import psutil
import time
import traceback
import sys

from openpyxl import Workbook
from openpyxl.styles import Font

#from scipy import sparse
#from scipy.linalg import svd

# List of routines in this file:
# 

####################################################################################

# CODE BLOCK 1: FUNCTIONS TO IMPORT THE ORIGINAL DATA AND ANNOTATE IT

####################################################################################

def load_h5ad_files(directory='.', cell_type_prefixes=None):
    """
    Load h5ad files from directory, optionally filtering by cell type prefix.
    
    Args:
        directory (str): Directory containing h5ad files
        cell_type_prefixes (str or list): Single cell type prefix or list of prefixes to load.
                                        If None, loads all files.
    """
    h5ad_files = glob.glob(f'{directory}/*.h5ad')
    adata_dict = {}
    
    # Convert single string prefix to list for consistent handling
    if isinstance(cell_type_prefixes, str):
        cell_type_prefixes = [cell_type_prefixes]
    
    for file in h5ad_files:
        key = file.split('/')[-1].replace('.h5ad', '')
        celltype, _, _ = parse_filename(key)
        
        # Skip files that don't match the cell type prefix filter
        if cell_type_prefixes and not any(celltype.startswith(prefix) for prefix in cell_type_prefixes):
            continue
            
        adata = sc.read_h5ad(file)
        adata.obs_names_make_unique(join='_')
        celltype, condition, replicate = parse_filename(key)
        adata.obs['condition'] = condition
        adata.obs['replicate'] = replicate
        for tp in adata.obs['timepoint'].unique():
            tp_key = f"{key}_t={tp}"
            adata_dict[tp_key] = adata[adata.obs['timepoint'] == tp].copy()
    
    return adata_dict

def parse_filename(filename):
    """Parse celltype, condition and replicate from filename."""
    base = filename.split('_t=')[0]  # Remove timepoint suffix
    parts = base.split('_')
    celltype = parts[0]
    
    if parts[1].endswith(('1', '2', '3')):
        condition = parts[1][:-1]
        replicate = int(parts[1][-1])
    else:
        condition = parts[1]
        replicate = 1
        
    return celltype, condition, replicate

def parse_timepoint(timepoint_str):
    """Extract numerical day value from timepoint string."""
    return int(timepoint_str[-1])

def get_umi_stats(adata):
    """Calculate UMI statistics from matrix."""
    umi_counts = adata.X.sum(axis=1)  # Shape (n_cells, 1)
    return (float(umi_counts.mean()), 
            float(np.median(umi_counts, axis=0)[0,0]),
            float(umi_counts.min()),
            float(umi_counts.max()))

def create_sample_info(adata_dict):
    """Create DataFrame with sample information."""
    records = []
        
    for key, adata in adata_dict.items():
        celltype, condition, replicate = parse_filename(key)
        timepoint = parse_timepoint(key.split('t=')[1])
        mean_umi, median_umi, min_umi, max_umi = get_umi_stats(adata)
        
        record = {
            'key': key,
            'celltype': key.split('_')[0],
            'condition': condition,
            'replicate': replicate,
            'timepoint': timepoint,
            'n_cells': adata.n_obs,
            'mean_umi': mean_umi,
            'median_umi': median_umi,
            'min_umi': min_umi,
            'max_umi': max_umi
        }
        records.append(record)
    sample_info = pd.DataFrame(records)
    sample_info.set_index('key',inplace=True)

    return sample_info



def read_batch_data(file_path):
    """
    Read an Excel file containing experiment batch data
    
    Parameters:
    file_path (str): Path to the Excel file
    
    Returns:
    pandas.DataFrame: DataFrame with 'Key' as index and specified columns
    """
    # Define expected columns
    expected_columns = [
        'Key',
        'AnnData file',
        'Library prep',
        'Seeding date',
        'Seeding flask (source cells)'
    ]
    
    try:
        # Read the Excel file
        df = pd.read_excel(
            file_path,
            index_col='Key'  # Set 'Key' as index
        )
        
        # Verify all expected columns are present
        missing_cols = [col for col in expected_columns if col not in df.columns and col != 'Key']
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find Excel file at: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")


def integrate_batch_data(sample_info, batch_info):

    sample_info = pd.concat([sample_info, batch_info],axis=1)
    sample_info = sample_info.sort_values(['Library prep','celltype','condition'])
    
    # # Fix replicates:
    # sample_info = sample_info.drop(columns=['replicate'])
    # unique_pairs = pd.DataFrame({
    #     'Library prep': sample_info['Library prep'],
    #     'Seeding date': sample_info['Seeding date']
    # }).drop_duplicates()
    # unique_pairs['replicate'] = range(1, len(unique_pairs) + 1)
    # sample_info = sample_info.merge(
    #     unique_pairs, 
    #     on=['Library prep', 'Seeding date'], 
    #     how='left'
    # )

    return sample_info

# Example usage:
# df = read_experiment_data('path_to_your_file.xlsx')
# print(df.head())

# def clean_gene_names(adata):
#     """
#     Clean gene names in AnnData object by removing species-specific suffixes.
#     Handles both '_mm' and '_hg' suffixes, as well as their variations.
    
#     Parameters
#     ----------
#     adata : anndata.AnnData
#         The AnnData object containing gene names to be cleaned
        
#     Returns
#     -------
#     anndata.AnnData
#         The modified AnnData object with cleaned gene names
        
#     Notes
#     -----
#     This function:
#     - Creates a copy of the AnnData object to avoid modifying the original
#     - Removes '_mm' and '_hg' suffixes from gene names
#     - Updates both var_names and var index
#     - Checks for and warns about duplicate gene names after cleaning
    
#     Examples
#     --------
#     >>> import scanpy as sc
#     >>> adata = sc.read_h5ad('my_data.h5ad')
#     >>> adata = clean_gene_names(adata)
#     """
#     import pandas as pd
    
#     # Create a copy to avoid modifying the original
#     adata = adata.copy()
    
#     # Function to remove suffixes
#     def remove_suffix(name):
#         if name.endswith('_mm') or name.endswith('_hg'):
#             return name[:-3]
#         return name
    
#     # Clean the names
#     new_names = [remove_suffix(name) for name in adata.var_names]
    
#     # Check for duplicates after cleaning
#     if len(set(new_names)) < len(new_names):
#         duplicate_names = pd.Series(new_names)[pd.Series(new_names).duplicated()].unique()
#         print(f"Warning: Found {len(duplicate_names)} duplicate gene names after cleaning:")
#         print(duplicate_names)
        
#     # Update the var_names and index
#     adata.var_names = new_names
#     adata.var.index = new_names
    
#     return adata

# Optional: Extended version with more flexible suffix handling
def clean_gene_names_flexible(adata, suffixes=None, case_sensitive=True):
    """
    More flexible version of clean_gene_names that allows custom suffix specification.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing gene names to be cleaned
    suffixes : list, optional
        List of suffixes to remove. Defaults to ['_mm', '_hg']
    case_sensitive : bool, optional
        Whether to treat suffixes as case-sensitive. Defaults to True
        
    Returns
    -------
    anndata.AnnData
        The modified AnnData object with cleaned gene names
    """
    import pandas as pd
    
    
    # Default suffixes if none provided
    if suffixes is None:
        suffixes = ['_mm', '_hg']
    
    # Function to remove suffixes
    def remove_suffix(name):
        original_name = name
        if not case_sensitive:
            name = name.lower()
            suffixes_to_check = [suffix.lower() for suffix in suffixes]
        else:
            suffixes_to_check = suffixes
            
        for suffix in suffixes_to_check:
            if name.endswith(suffix):
                return original_name[:-len(suffix)]
        return original_name
    
    # Clean the names
    new_names = [remove_suffix(name) for name in adata.var_names]
    
    # Check for duplicates after cleaning
    if len(set(new_names)) < len(new_names):
        duplicate_names = pd.Series(new_names)[pd.Series(new_names).duplicated()].unique()
        print(f"Warning: Found {len(duplicate_names)} duplicate gene names after cleaning:")
        print(duplicate_names)
        
    # Update the var_names and index
    adata.var_names = new_names
    adata.var.index = new_names
    


####################################################################################

# CODE BLOCK 2: FUNCTIONS TO PRE-PROCESS DATA

####################################################################################


def unify_adata(adata_dict, sample_info, filter_genes=None):
    """Integrate all data for a specific celltype."""
    # Get all samples for this celltype

    for k in adata_dict.keys():
        adata_dict[k].obs['sample_key'] = k
        adata_dict[k].obs['condition'] = sample_info.loc[k,'condition']
        adata_dict[k].obs['replicate'] = sample_info.loc[k,'replicate']
        adata_dict[k].obs['timepoint'] = sample_info.loc[k,'timepoint']
        adata_dict[k].obs['Library'] = sample_info.loc[k,'Library prep']
        adata_dict[k].obs['Seeding'] = sample_info.loc[k,'Seeding flask (source cells)']
    
    
    # Concatenate all samples for this celltype
    adatas = [adata_dict[k] for k in adata_dict.keys()]
    
    adata_concat = sc.concat(adatas, join='outer')
    del adatas
    
    if filter_genes is not None:
        filter_to_gene_set(adata_concat, filter_genes)
    
    adata_concat.obs_names_make_unique(join='_')
    # Convert to categorical if they aren't already
    adata_concat.obs['Library'] = adata_concat.obs['Library'].astype('category')
    adata_concat.obs['Seeding'] = adata_concat.obs['Seeding'].astype('category')
    adata_concat.obs['timepoint'] = adata_concat.obs['timepoint'].astype('category')
    
    return adata_concat



def filter_to_gene_set(adata, gene_set):
    """
    Filter annData in-place to only keep genes present in the provided gene set.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix to filter. Will be modified in-place.
    gene_set : set or list-like
        Set of gene names to keep
        
    Notes
    -----
    - Modifies the AnnData object in-place to save memory
    - Genes in gene_set that are not in adata.var_names will be ignored
    - The order of genes in the output will match their order in the original adata
    
    Example usage:
    -----
    my_genes = {'CCNB1', 'GAPDH', 'CD19', 'MS4A1'}
    filter_to_gene_set(adata, my_genes)  # adata is modified in-place
    
    """
    # Convert gene_set to a set if it isn't already
    gene_set = set(gene_set)
    
    # Find intersection with existing genes (maintaining original order)
    genes_to_keep = [gene for gene in adata.var_names if gene in gene_set]
    
    if len(genes_to_keep) == 0:
        raise ValueError("No genes from the provided gene_set were found in the data")
    
    # Store original stats for reporting
    #n_original = adata.n_vars
    #original_size = adata.X.nbytes
    
    # Perform in-place filtering
    adata._inplace_subset_var(genes_to_keep)
    
    # Log information about the filtering
    #n_kept = adata.n_vars
    #new_size = adata.X.nbytes
    #print(f"Kept {n_kept:,} genes out of {n_original:,} original genes")
    #print(f"Memory usage reduced from {original_size / 1e9:.2f}GB to {new_size / 1e9:.2f}GB")


    

def run_preprocess_and_harmony(adata, num_pcs=10, use_harmony=False, use_NMF=False, 
                               NMF_kwargs={},hvg_kwargs=None):
    

    adata.layers['raw'] = adata.X.copy()
    sc.pp.normalize_total(adata,target_sum=1e4)
    adata.layers['norm'] = adata.X.copy()
    # Generate log-transformed and store for later:
    sc.pp.log1p(adata,base=10)
    adata.layers['log'] = adata.X.copy()

    
    # Generate z-scaled:
    #adata.X = adata.layers['norm']
    #sc.pp.scale(adata,max_value=10)
    
    # 2. Find highly variable genes (HVGs)
    sc.pp.highly_variable_genes(adata,inplace=True,**hvg_kwargs)
    #print('Number highly variable = ',np.sum(adata.var['highly_variable']))
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Run PCA, projected
    #ref_mask = (adata.obs['timepoint']==0)
    #adata.obs['pca_ref']=False
    #adata.obs.loc[ref_mask,'pca_ref']=True
    #run_PCA_projected(adata, reference_col='pca_ref',reference_val=True, num_pcs=6)
    if use_NMF == False:
        sc.pp.pca(adata,n_comps=num_pcs,layer='log',mask_var="highly_variable")
    else:
        print('Running NMF:')
        NMF_default_params = {'init':'nndsvd', 'random_state':0}
        NMF_params = {**NMF_default_params, **NMF_kwargs}
        X = adata[:, adata.var['highly_variable']].X
        #print(np.shape(X))
        model = NMF(n_components=num_pcs, **NMF_params)
        adata.obsm['X_nmf'] = model.fit_transform(X)
        adata.uns['nmf'] = {
            'params': NMF_params,
            'model' : model,
            'variance_ratio': model.reconstruction_err_,
            'n_components': num_pcs,
            'components': model.components_,  # Store components here
            'highly_variable_genes': adata.var_names[adata.var['highly_variable']].tolist()  # Store which genes were used
        }


    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


    #harmonypy.run_harmony()
    if use_harmony:
        if use_NMF:
            X_key = 'X_nmf'
        else:
            X_key = 'X_pca'
            
        adata.obs['batchkey']= adata.obs['Seeding'].astype(str)+'_'+adata.obs['Library'].astype(str)
        sce.pp.harmony_integrate(
            adata,
            key=['batchkey'],
            basis=X_key,
            theta=4.0,        # Increase from default 2.0 for stronger correction
            lamb=10.0,   # Increase from default 1.0 for stronger correction
            sigma=0.05,        # Decrease from default 0.3 for tighter clustering
            max_iter_harmony = 30
        )
        sc.pp.neighbors(adata, use_rep='X_pca_harmony',n_neighbors=15)
    else:
        if use_NMF:
            sc.pp.neighbors(adata, use_rep='X_nmf',n_neighbors=15)
        else:
            sc.pp.neighbors(adata, use_rep='X_pca',n_neighbors=15)



def rank_genes(adata, logfc_thresh=0.25):

    # Run differential expression analysis
    sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon', layer='log')

    clusters = sorted(adata.obs['leiden'].unique().tolist())  # Sort clusters numerically

    for cluster in clusters:
        print(f"\nCluster {cluster}:")

        # Extract gene names and log fold changes for the cluster
        genes = adata.uns['rank_genes_groups']['names'][cluster]
        logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges'][cluster]

        # Apply log fold change filtering
        filtered_genes = [gene for gene, lfc in zip(genes, logfoldchanges) if lfc > logfc_thresh]

        # Print the top 50 genes after filtering
        print(", ".join(filtered_genes[:50]))


####################################################################################

# CODE BLOCK 3: OLDER/LEGACY PRE-PROCESSING SUBROUTINES

####################################################################################
# def run_PCA_projected(adata, reference_col='pca_ref',reference_val=True, num_pcs=6):
#     """Run dimensionality reduction using reference PCA projection."""
    
#     # Get reference data
#     print('Getting reference data...')
#     ref_mask = (adata.obs[reference_col] == reference_val)
#     ref_adata = adata[ref_mask].copy()
#     gc.collect()  # Cleanup after copy
    
#     # Get PCA on reference
#     print('Computing reference PCA...')
#     sc.tl.pca(ref_adata, n_comps=num_pcs,use_highly_variable=False)
    
#     # Project all samples into reference PCA space
#     print('Projecting into PCA space...')
#     adata.varm['PCs'] = ref_adata.varm['PCs']
#     adata.uns['pca'] = {'variance': ref_adata.uns['pca']['variance']}
#     # Mean-center using the reference data's mean before projection
#     adata.var['means'] = adata.X.mean(axis=0)
#     adata.obsm['X_pca'] = (adata.X - adata.X.mean(axis=0)) @ ref_adata.varm['PCs']
    
#     # Clean up reference data
#     del ref_adata
#     gc.collect()



# def filter_gene_list(genes, exclude_prefixes=['Gm', 'Rpl', 'Rps', 'mt-', 'CT', 'AC']):
#     """
#     Filter a list of genes by excluding those that start with specified prefixes.
    
#     Args:
#         genes (list): List of gene names to filter
#         exclude_prefixes (list): List of prefixes to exclude. 
#                 Defaults to ['Gm', 'Rpl', 'Rps', 'mt-', 'CT', 'AC']
            
#     Returns:
#         list: Filtered list of genes
#     """

#     # Filter genes that don't start with any of the excluded prefixes
#     filtered_genes = [
#         gene for gene in genes 
#         if not any(gene.startswith(prefix) for prefix in exclude_prefixes)
#     ]
    
#     return filtered_genes


####################################################################################

# CODE BLOCK 4: NMF ANALYSIS

####################################################################################
def get_nmf_usage_stats(W, adata, condition=None):
    """
    Compute mean, standard deviation, and coefficient of variation of NMF program usage across timepoints.
    
    Parameters:
    -----------
    W : Union[np.ndarray, pd.DataFrame]
        NMF program usage matrix (samples x programs), either as numpy array or DataFrame
    adata : AnnData
        AnnData object containing metadata
    condition : str, optional
        Condition to filter for (e.g., 'Ctrl'). If None, uses all data
        
    Returns:
    --------
    tuple of pd.DataFrame
        (per_timepoint_mean, per_timepoint_std, per_timepoint_cv)
        Each DataFrame has programs as columns and timepoints as index
    """
    # Convert input to DataFrame if it's an array
    if isinstance(W, np.ndarray):
        nmf_df = pd.DataFrame(W/W.sum(axis=1, keepdims=True),  # Normalize per-cell program usages
                             index=adata.obs.index, 
                             columns=[f'{i}' for i in range(W.shape[1])])
    else:  # Already a DataFrame
        # Ensure the dataframe is normalized
        nmf_df = W.copy()
        nmf_df = nmf_df.div(nmf_df.sum(axis=1), axis=0)
    #print(nmf_df)
    # Add metadata columns
    nmf_df["timepoint"] = adata.obs["timepoint"].astype(int)
    nmf_df["sample_key"] = adata.obs["sample_key"].astype(str)
    nmf_df["condition"] = adata.obs["condition"].astype(str)
    
    # Apply condition filter if specified
    if condition is not None:
        if condition != 'Ctrl':
            mask = (nmf_df["condition"] == condition)
        else:
            mask = nmf_df["condition"].str.contains('Ctrl')
            
        nmf_df = nmf_df[mask]
    
    # Drop condition column as it's no longer needed for grouping
    nmf_df = nmf_df.drop('condition', axis=1)
    
    # Compute per-sample statistics
    per_sample_mean = nmf_df.groupby(["sample_key", "timepoint"]).mean()
    per_sample_median = nmf_df.groupby(["sample_key", "timepoint"]).median()
    per_sample_var = nmf_df.groupby(["sample_key", "timepoint"]).var()

    
    # Compute per-timepoint statistics
    per_timepoint_mean = per_sample_mean.groupby("timepoint").mean()
    per_timepoint_median = per_sample_median.groupby("timepoint").mean()
    per_timepoint_std = per_sample_var.groupby("timepoint").mean().apply(np.sqrt)
    per_timepoint_cv = per_timepoint_std / per_timepoint_mean
    per_timepoint_fano = per_timepoint_std **2 / per_timepoint_mean
    
    return per_timepoint_mean, per_timepoint_std, per_timepoint_cv, per_timepoint_fano, per_timepoint_median


############################################################################################
############################################################################################
############################################################################################

import difflib
import numpy as np
import pandas as pd

import difflib
import pandas as pd
import anndata as ad

def plot_gene_usage(H, gene_names, gene, prog_list=None):
    """
    Plot gene usage across programs.
    
    Parameters:
    -----------
    H : numpy.ndarray
        Matrix of program loadings
    gene_names : numpy.ndarray
        Array of gene names
    gene : str
        Gene name to plot
    prog_list : list, optional
        List of program names
    """
    # Convert gene names to same case as input gene for comparison
    gene_names_normalized = np.array([g.strip() for g in gene_names])
    gene_normalized = gene.strip()
    
    # First check if gene exists (case-insensitive)
    gene_matches = np.where(np.char.lower(gene_names_normalized) == gene_normalized.lower())[0]
    
    if len(gene_matches) == 0:
        raise ValueError(f"Gene '{gene}' not found in gene_names. Please check spelling and case.")
    
    gene_idx = gene_matches[0]  # Take first match if multiple exist
    gene_loadings = H[:, gene_idx]  # Get loadings across programs
    
    if prog_list is None:
        prog_list = [f"Program {i+1}" for i in range(len(gene_loadings))]
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(prog_list, gene_loadings)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Loading')
    plt.title(f'Gene usage: {gene}')
    plt.tight_layout()
    
    return gene_loadings



############################################################################################
############################################################################################
############################################################################################



import numpy as np
from scipy import sparse
import anndata as ad
import pandas as pd

def create_averaged_usage_matrices(ad_sc, timepoint_ns):
    all_adata = []
    
    for timepoint, n in timepoint_ns.items():
        all_averaged_H = []
        all_conditions = []
        
        # Process each condition separately
        for condition in ad_sc.obs['condition'].unique():
            condition_mask = ad_sc.obs['condition'] == condition
            condition_indices = np.where(condition_mask)[0]
            
            H_condition = ad_sc.obsm['X_nmf'][condition_indices]
            N_condition = H_condition.shape[0]
            n_samples = N_condition // n
            
            if n_samples > 0:
                perm_idx = np.random.permutation(N_condition)
                reshaped = H_condition[perm_idx[:n*n_samples]].reshape(n_samples, n, -1)
                averaged_H = reshaped.mean(axis=1)
                
                all_averaged_H.append(averaged_H)
                all_conditions.extend([condition] * n_samples)
        
        combined_H = np.vstack(all_averaged_H)
        
        n_cells = combined_H.shape[0]
        n_genes = ad_sc.n_vars
        
        X_empty = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
        
        # Create obs DataFrame with string index and timepoint
        obs_df = pd.DataFrame({
            'condition': all_conditions,
            'timepoint': [timepoint] * n_cells
        }, index=[f'cell_{timepoint}_{i}' for i in range(n_cells)])

        obs_df['sample_key'] = obs_df['condition']+obs_df['timepoint'].astype(str)
        
        # Create new AnnData
        new_adata = ad.AnnData(
            X=X_empty,
            obs=obs_df,
            var=ad_sc.var.copy(),
            dtype=np.float32
        )
        
        new_adata.obsm['X_nmf'] = combined_H
        all_adata.append(new_adata)
    
    # Concatenate all AnnData objects
    combined_adata = ad.concat(
        all_adata,
        join='outer',
        merge='same'
    )
    
    return combined_adata

# Example usage:
# timepoint_ns = {'0hr': 2, '24hr': 4, '48hr': 6}
# combined_ad = create_averaged_usage_matrices(ad_sc, timepoint_ns)



def create_count_based_clones(ad_sc, ref_df, n_cells_per_timepoint):
    """
    Create mock clones based on a reference dataframe specifying conditions and counts.
    
    Parameters:
    -----------
    ad_sc : AnnData
        Input single-cell data with raw counts in .layers['raw'] and n_counts in obs
    ref_df : pd.DataFrame
        Reference dataframe with columns:
        - 'timepoint': timepoint for the clone
        - 'condition': condition for the clone
        - 'n_counts': target number of counts for the clone
    n_cells_per_timepoint : dict
        Dictionary mapping timepoints to number of cells to combine
            
    Returns:
    --------
    AnnData
        Combined dataset with mock clones
    """
    all_adata = []
    
    # Get NMF components and HVG indices for later projection
    hvg_genes = ad_sc.uns['nmf']['highly_variable_genes']
    hvg_indices = np.where(ad_sc.var_names.isin(hvg_genes))[0]
    nmf_model = ad_sc.uns['nmf']['model']
    
    # Process each timepoint
    for timepoint in n_cells_per_timepoint.keys():
        timepoint_df = ref_df[ref_df['timepoint'] == timepoint]
        min_cells = n_cells_per_timepoint[timepoint]
        min_extra_cells = 2
        
        all_counts = []
        all_conditions = []
        all_other_cols = {col: [] for col in ref_df.columns if col not in ['timepoint', 'condition', 'n_counts']}
        

        # Process each condition
        for condition in timepoint_df['condition'].unique():
            condition_mask = ad_sc.obs['condition'] == condition
            condition_indices = np.where(condition_mask)[0]
            
            if len(condition_indices) == 0:
                continue
                
            # Get the rows for this condition
            clone_specs = timepoint_df[timepoint_df['condition'] == condition]
            
        
            # Create clones according to specifications
            for _, row in clone_specs.iterrows():
                # First determine how many cells we need by checking n_counts
                cell_counts = ad_sc.obs['n_counts'].iloc[condition_indices]
                n_cells = min_cells
                
                # Keep sampling until we have enough total counts
                while True:
                    sampled_indices = np.random.choice(len(condition_indices), size=n_cells, replace=True)
                    total_counts = cell_counts.iloc[sampled_indices].sum()
                    
                    if total_counts >= row['n_counts']:
                        break
                        
                    # Add more cells if needed
                    n_cells += min_extra_cells
                
                # Now get the actual counts from the raw layer
                combined_counts = ad_sc.layers['raw'][condition_indices[sampled_indices]].sum(axis=0).A1
                
                # Convert to frequencies for multinomial sampling
                frequencies = combined_counts / combined_counts.sum()
                #frequencies = (combined_counts / combined_counts.sum()).astype(int)
  
                # Multinomial sampling to target count
                sampled_counts = np.random.multinomial(np.int64(row['n_counts']), frequencies)
                
                # Add to results
                all_counts.append(sampled_counts)
                all_conditions.append(condition)
                for col in all_other_cols:
                    all_other_cols[col].append(row[col])
        
        if len(all_counts) == 0:
            continue
            
        # Convert counts to numpy array
        combined_counts = np.vstack(all_counts)
        
        # CP10K normalization followed by log transform
        cp10k = combined_counts * 10000 / combined_counts.sum(axis=1, keepdims=True)
        log_counts = np.log10(1 + cp10k)
        
        # Project into NMF space using only HVGs and scaled components
        projected_W = nmf_model.transform(log_counts[:, hvg_indices])
        
        n_clones = combined_counts.shape[0]
        n_genes = ad_sc.n_vars
        
        # Create obs DataFrame
        obs_dict = {
            'condition': all_conditions,
            'timepoint': [timepoint] * n_clones
        }
        # Add all additional columns from reference df
        obs_dict.update(all_other_cols)
        
        obs_df = pd.DataFrame(
            obs_dict,
            index=[f'cell_{timepoint}_{i}' for i in range(n_clones)]
        )
        obs_df['sample_key'] = obs_df['condition'] + obs_df['timepoint'].astype(str)
        
        # Create new AnnData
        new_adata = ad.AnnData(
            X=sparse.csr_matrix(log_counts),
            obs=obs_df,
            var=ad_sc.var.copy(),
            dtype=np.float32
        )
        
        # Store different representations as layers
        new_adata.layers['raw'] = sparse.csr_matrix(combined_counts)
        new_adata.layers['norm'] = sparse.csr_matrix(cp10k)
        
        # Store NMF projection
        new_adata.obsm['X_nmf'] = projected_W
        
        all_adata.append(new_adata)
    
    # Concatenate all AnnData objects
    combined_adata = ad.concat(
        all_adata,
        join='outer',
        merge='same'
    )
    
    return combined_adata



def get_gene_cv_mean_dataframes(adata):
    # Get the conditions:
    cond = adata.obs['condition'].unique()
    # Get the timepoints
    timepoints = adata.obs['timepoint'].unique()
    # Get the genes
    genes = adata.var_names
    cv_mean_dict = {}
    
    # Loop over conditions:
    for c in cond:
        cv_mean_dict[c] = {}
        # Loop over timepoints:
        for t in timepoints:
            # Get the samples for this condition and timepoint
            samples = adata.obs[(adata.obs['condition']==c) & (adata.obs['timepoint']==t)]['sample_key'].unique()
            # Loop over samples:
            df_list = []
            n_cells_list = []
            for s in samples:
                # Get the data for this sample
                subsample = adata[adata.obs['sample_key']==s].copy()
                sc.pp.highly_variable_genes(subsample, flavor='seurat_v3', n_top_genes=100, layer='raw')
                
                # Convert sparse matrices to dense for calculations
                norm_layer = subsample.layers['norm'].toarray()
                
                # Get the mean, var, cv, and norm_var for each gene
                mean = np.mean(norm_layer, axis=0)  # Changed to axis=0 for gene-wise calculations
                var = np.var(norm_layer, axis=0)
                mean_sc = subsample.var['means'].values
                var_sc = subsample.var['variances'].values
                cv = np.sqrt(var)/(1e-10+mean)
                norm_var = subsample.var['variances_norm'].values
                hvg = subsample.var['highly_variable'].values

                # Prepare a dataframe for this sample
                df = pd.DataFrame({
                    'mean': mean, 
                    'var': var, 
                    'mean_sc': mean_sc,
                    'var_sc': var_sc,
                    'cv': cv, 
                    'norm_var': norm_var, 
                    'hvg': hvg
                }, index=subsample.var_names)
                
                df_list.append(df)
                n_cells_list.append(len(subsample))
                del subsample, norm_layer  # Clear memory
                
            # Weighted average over samples per condition and per timepoint 
            # Identify genes with zero means in any sample
            zero_mean_genes = set()
            for df in df_list:
                zero_mean_genes.update(df.index[df['mean'] == 0])
            
            weights = np.array(n_cells_list) / np.sum(n_cells_list)
            df_avg = pd.concat([df * w for df, w in zip(df_list, weights)]).groupby(level=0).sum()
            
            # Filter df_avg to keep only non-zero genes
            #df_avg = df_avg[~df_avg.index.isin(zero_mean_genes)]
            # Store in the dictionary
            cv_mean_dict[c][t] = df_avg
            
    return cv_mean_dict


def fit_cv_mean_relationship(mean_dict, cv_dict, exclude_keys=None, fixed_exponent=None):
    """
    Fit power law relationship CV = a * mean^b using log-transformed linear regression.
    Optionally fix the exponent b and only fit the multiplicative constant a.
    
    Parameters:
    -----------
    mean_dict : dict
        Dictionary of dataframes containing mean values
    cv_dict : dict
        Dictionary of dataframes containing CV values
    exclude_keys : list or None
        Keys to exclude from the analysis (e.g., ['Ctrl']). If None, use all data.
    fixed_exponent : float or None
        If provided, fixes the power law exponent to this value (e.g., -0.5)
        and only fits the multiplicative constant
        
    Returns:
    --------
    dict
        Contains fitted parameters and statistics:
        - 'a': multiplicative constant
        - 'b': power law exponent (if not fixed)
        - 'r2': R-squared value
        - 'log_params': Parameters in log space
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Collect all data points
    mean_data = []
    cv_data = []
    
    exclude_keys = [] if exclude_keys is None else exclude_keys
    
    for condition in mean_dict.keys():
        if condition in exclude_keys:
            continue
            
        for col in mean_dict[condition].columns:
            mean_data.extend(mean_dict[condition][col].values)
            cv_data.extend(cv_dict[condition][col].values)
    
    # Convert to numpy arrays and log-transform
    X = np.log(np.array(mean_data))
    y = np.log(np.array(cv_data))
    
    if fixed_exponent is not None:
        # If exponent is fixed, just fit the multiplicative constant
        # In log space: log(y) = log(a) + b*log(x)
        # With b fixed, we just need to find log(a)
        log_a = np.mean(y - fixed_exponent * X)
        a = np.exp(log_a)
        b = fixed_exponent
        
        # Calculate R² manually
        y_pred = log_a + fixed_exponent * X
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        log_params = {'slope': fixed_exponent, 'intercept': log_a}
    
    else:
        # Regular linear regression if exponent is not fixed
        X = X.reshape(-1, 1)  # reshape for sklearn
        reg = LinearRegression()
        reg.fit(X, y)
        
        b = reg.coef_[0]  # power law exponent
        a = np.exp(reg.intercept_)  # multiplicative constant
        r2 = reg.score(X, y)
        log_params = {'slope': b, 'intercept': reg.intercept_}
    
    return {
        'a': a,
        'b': b,
        'r2': r2,
        'log_params': log_params
    }




def create_regressed_randomized_cv(mean_dict, rnd_mean_dict, rnd_cv_dict):
    """
    Create regulated CV dictionary based on the formula:
    rndCV²_reg = rndCV² + a²(1/mean - 1/rnd_mean)

    Following the derivation:
    Assume:
        CV^2 = a^2 * (1 + xi_bio ) / m 
        Note that this form isn't canonical. 
        The overall form is Poisson noise. 
        But we assume that the multiplier has some average a^2 that we learn from regression, and then a correction

    And we want to find:
        CV_rr^2 = a^2 * (1 + xi_bio ) / m0 

    So
        CV_rr^2 = CV^2  * (m/m0)
    
    
    Parameters:
    -----------
    mean_dict : dict
        Dictionary of dataframes with real mean values
    rnd_mean_dict : dict
        Dictionary of dataframes with randomized mean values
    rnd_cv_dict : dict
        Dictionary of dataframes with randomized CV values

            
    Returns:
    --------
    tuple
        (rnd_reg_cv_dict, rnd_reg_mean_dict) where:
        - rnd_reg_cv_dict: Dictionary of dataframes with regulated CV values
        - rnd_reg_mean_dict: Dictionary of dataframes with filtered mean values 
          matching the rows in rnd_reg_cv_dict
    """
    import numpy as np
    
    
    # Create output dictionaries with same structure
    rnd_reg_cv_dict = {}
    rnd_reg_mean_dict = {}
    
    for c in rnd_mean_dict.keys():
        # Initialize dataframes with same structure
        rnd_reg_cv_dict[c] = rnd_cv_dict[c].copy()
        
        # Filter mean_dict to only include indices present in rnd_mean_dict
        if c in mean_dict:
            rnd_reg_mean_dict[c] = mean_dict[c].loc[rnd_mean_dict[c].index].copy()

            for col in rnd_mean_dict[c].columns:
                # Get values
                mean = rnd_reg_mean_dict[c][col]
                rnd_mean = rnd_mean_dict[c][col]
                rnd_cv = rnd_cv_dict[c][col]
                
                # Calculate regulated CV according to formula
                # rndCV²_reg = rndCV² + a²(mean - rnd_mean)
                rnd_reg_cv = rnd_cv * np.sqrt(rnd_mean/mean)
                
                # Store in output dictionary
                rnd_reg_cv_dict[c][col] = rnd_reg_cv



    return rnd_reg_cv_dict, rnd_reg_mean_dict


 

 ###########################################################################################
 ###########################################################################################
 ###########################################################################################
from sklearn.decomposition import TruncatedSVD
import numpy as np

def compare_variance_explained(adata, n_components=20, sample_frac=0.1, n_permutations=10):
    # Get subset and convert to dense
    X = adata[:, adata.var.highly_variable].X
    X_subset = X[np.random.choice(X.shape[0], int(X.shape[0] * sample_frac), replace=False)].toarray()
    
    # Process real data
    X_norm = X_subset - X_subset.mean(axis=0)
    X_norm = X_norm / np.sqrt((X_norm**2).sum())
    real_var = TruncatedSVD(n_components).fit(X_norm).explained_variance_ratio_
    
    # Process permuted data
    perm_vars = []
    for _ in range(n_permutations):
        X_perm = X_norm.copy()
        row_idx = np.random.rand(*X_perm.shape).argsort(axis=0)
        col_idx = np.tile(np.arange(X_perm.shape[1]), (X_perm.shape[0], 1))
        X_perm = X_perm[row_idx, col_idx]
        perm_vars.append(TruncatedSVD(n_components).fit(X_perm).explained_variance_ratio_)
    
    return real_var, np.array(perm_vars)


def evaluate_factor_quality(adata, H):
    X_hvg = adata[:, adata.var.highly_variable].X.toarray()
    nonzero_means = X_hvg.mean(axis=0)/(1-np.sum(X_hvg == 0, axis=0)/X_hvg.shape[0])
    
    H_weighted = H * nonzero_means  # H is (15,4000), nonzero_means is (4000,)
    H_norm = H_weighted / (H_weighted.sum(axis=1, keepdims=True) + 1e-10)
    entropy = -np.sum(H_norm * np.log2(H_norm + 1e-10), axis=1)
    
    top_genes = np.argsort(-H_norm, axis=1)[:,:10]
    other_factors = np.array([np.delete(H_norm.T, i, axis=1).max(axis=1)[top_genes[i]].mean() 
                            for i in range(H.shape[0])])
    uniqueness = 1 - other_factors
    
    return entropy, uniqueness




def vectorized_ranksums(x_groups, y_groups):
    """
    Performs ranksum tests between corresponding groups in x_groups and y_groups
    
    Parameters:
    x_groups: array of shape (n_tests, n_samples_x)
    y_groups: array of shape (n_tests, n_samples_y)
    
    Returns:
    statistic: array of shape (n_tests,)
    pvalue: array of shape (n_tests,)
    """

    import numpy as np
    #from scipy.stats import ranksums
    from scipy.stats import rankdata, norm
    #import numpy.ma as ma

    # Combine all samples for ranking
    n_x = x_groups.shape[1]
    combined = np.hstack((x_groups, y_groups))
    
    # Calculate ranks for all values at once
    ranks = rankdata(combined, axis=1)
    
    # Split ranks back into x and y groups
    x_ranks = ranks[:, :n_x]
    y_ranks = ranks[:, n_x:]
    
    # Calculate rank sums and sample sizes
    n1 = x_groups.shape[1]
    n2 = y_groups.shape[1]
    r1 = np.sum(x_ranks, axis=1)
    
    # Calculate test statistics
    u1 = r1 - (n1 * (n1 + 1)) / 2
    u2 = n1 * n2 - u1
    
    # Use smaller U as test statistic
    statistic = np.where(u1 < u2, u1, u2)
    
    # Calculate z-scores
    mean_rank = n1 * n2 / 2
    sd_rank = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (statistic - mean_rank) / sd_rank
    
    # Calculate p-values
    pvalue = 2 * norm.sf(np.abs(z))
    
    return statistic, pvalue




def differential_expression_analysis(adata, control_dict, timepoints=[4,6], drugs=['Vor','Dec','Aza']):
    """
    Performs differential expression analysis between drug treatments and matched controls.
    
    For each drug condition, compares gene expression to matched control samples at specified timepoints.
    Uses vectorized rank-sum test on highly variable genes. Combines results across timepoints using:
    - For fold change: Weighted average of log10 expression values per condition, then difference
    - For p-values: Fisher's method to combine across timepoints
    
    Note: Assumes adata.X contains log10(1+CP10K) transformed data
    
    Args:
        adata: AnnData object containing log10(1+CP10K) expression data
        control_dict: Dictionary mapping drug conditions to their matched controls
        timepoints: List of timepoints to analyze (default [4,6])
        drugs: List of drug conditions to analyze (default ['Vor','Dec','Aza'])
    
    Returns:
        results_dict: Dictionary with drug keys containing DataFrames of:
            - mean_logFC: Log10 fold-change between weighted means
            - combined_fdr: FDR-corrected combined p-values across timepoints
    """
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import combine_pvalues
    import numpy as np
    
    # Get highly variable genes
    hvg = adata.var_names[adata.var['highly_variable']]
    
    results_dict = {}
    
    for drug in drugs:
        # Initialize storage for timepoint results
        timepoint_pvals = []
        drug_means = []
        ctrl_means = []
        drug_weights = []
        ctrl_weights = []
        total_drug_cells = 0
        total_ctrl_cells = 0
        
        control = control_dict[drug]
        
        for tp in timepoints:
            print(f'Analyzing {drug}, time point {tp} days:')
            # Get drug and control cells at this timepoint
            drug_cells = ((adata.obs['condition']==drug) | 
                         (adata.obs['condition_orig']==drug)) & (adata.obs['timepoint']==tp)
            ctrl_cells = ((adata.obs['condition']==control) | 
                         (adata.obs['condition_orig']==control)) & (adata.obs['timepoint']==tp)
            
            # Skip if no cells in either condition
            if sum(drug_cells)==0 or sum(ctrl_cells)==0:
                continue
            
            # Update total cell counts
            n_drug = sum(drug_cells)
            n_ctrl = sum(ctrl_cells)
            total_drug_cells += n_drug
            total_ctrl_cells += n_ctrl
                
            # Calculate mean expression
            print('Calculating mean expression...')
            drug_mean = adata[drug_cells,hvg].X.toarray().mean(axis=0)
            ctrl_mean = adata[ctrl_cells,hvg].X.toarray().mean(axis=0)
            
            # Store means and cell counts for later weight calculation
            drug_means.append(drug_mean)
            ctrl_means.append(ctrl_mean)
            drug_weights.append(n_drug)
            ctrl_weights.append(n_ctrl)
            
            # Calculate rank-sum p-values (vectorized)
            print('Calculating rank-sum p-values...')
            drug_expr = adata[drug_cells,hvg].X.toarray()  # (n_drug_cells, n_genes)
            ctrl_expr = adata[ctrl_cells,hvg].X.toarray()  # (n_ctrl_cells, n_genes)
            
            # Transpose to match vectorized_ranksums input shape (n_genes, n_cells)
            _, pvals = vectorized_ranksums(drug_expr.T, ctrl_expr.T)
            timepoint_pvals.append(pvals)
        
        print('Calculating weighted means and log fold change...')
        # Convert counts to weights
        drug_weights = np.array(drug_weights) / total_drug_cells
        ctrl_weights = np.array(ctrl_weights) / total_ctrl_cells
        
        # Calculate weighted means for each condition
        weighted_drug_mean = sum(m * w for m, w in zip(drug_means, drug_weights))
        weighted_ctrl_mean = sum(m * w for m, w in zip(ctrl_means, ctrl_weights))
        
        # Calculate log fold change as difference of weighted means
        logFC = weighted_drug_mean - weighted_ctrl_mean
        
        # Combine p-values using Fisher's method
        print('Combining p-values using Fisher\'s method...')
        timepoint_pvals = np.array(timepoint_pvals).T
        combined_pvals = np.array([
            combine_pvalues(gene_pvals, method='fisher')[1]
            for gene_pvals in timepoint_pvals
        ])
            
        # FDR correction
        _, fdr, _, _ = multipletests(combined_pvals, method='fdr_bh')
        
        # Store results
        results_dict[drug] = pd.DataFrame({
            'mean_logFC': logFC,
            'combined_fdr': fdr
        }, index=hvg)
        
    return results_dict




