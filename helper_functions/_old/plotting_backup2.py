# Functions used in processing memory-seq data sets from Ignas Mazelis' work
# 
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42  # Ensures consistency for PostScript files

# Functions in this file:
# save_umaps - for publication-quality
# plot_cluster_markers_heatmap
# 

##################################################

def save_umaps(
   adata,
   color_list,
   prefix="",
   save_folder=None,
   umap_kwargs={},
   rasterize=True,
   dpi=300,
   size=10,
   figsize=(5,5),
):
   """
   Save individual UMAP plots for each color variable with consistent formatting.
   
   Parameters
   ----------
   adata : AnnData
       Annotated data matrix with UMAP coordinates
   color_list : list
       List of variables to color UMAP by (must be in adata.obs)
   prefix : str, optional
       Prefix for saved filenames
   save_folder : str or Path, optional
       Directory to save plots in. Will be created if it doesn't exist.
   umap_kwargs : dict, optional
       Additional arguments to pass to sc.pl.umap
   rasterize : bool, optional
       Whether to rasterize the point layer (default: True)
   dpi : int, optional
       DPI for rasterized elements (default: 300)
   size : float, optional
       Point size (default: 10)
   figsize : tuple, optional
       Figure size in inches (width, height) (default: (5,5))
       
   Returns
   -------
   None
       Saves plots to files named {prefix}_umap_{color}.pdf
   
   Examples
   --------
   Basic usage:
   >>> save_umaps(
   ...     adata=adata,
   ...     color_list=['timepoint', 'condition', 'Library', 'Seeding'],
   ...     prefix='experiment1',
   ...     save_folder='umap_plots',
   ...     umap_kwargs={'palette': 'Set2'},
   ...     rasterize=True,
   ...     dpi=300,
   ...     size=8
   ... )
   """
   from pathlib import Path
   import matplotlib.pyplot as plt
   
   # Handle save folder
   if save_folder is not None:
       save_path = Path(save_folder)
       save_path.mkdir(parents=True, exist_ok=True)
   else:
       save_path = Path('.')
       
   # Default UMAP parameters that can be overridden
   default_params = {
       'frameon': True,
       'return_fig': True,
       'size': size
   }
   
   # Update defaults with user-provided parameters
   plot_params = {**default_params, **umap_kwargs}
   
   for color in color_list:
       # Set figure size before creating plot
       plt.figure(figsize=figsize)
       
       # Create the plot
       fig = sc.pl.umap(
           adata,
           color=color,
           **plot_params
       )
       
       # Force square aspect ratio
       ax = fig.axes[0]
       ax.set_aspect('equal', adjustable='box')
       
       # Get current axis limits
       xlim = ax.get_xlim()
       ylim = ax.get_ylim()
       
       # Make the limits square
       max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
       x_center = sum(xlim) / 2
       y_center = sum(ylim) / 2
       ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
       ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
       
       if rasterize:
           # Get the scatter plot artist (points)
           scatter = fig.axes[0].collections[0]
           # Rasterize only the points
           scatter.set_rasterized(True)
           # Set figure DPI for rasterization quality
           fig.set_dpi(dpi)
       
       # Adjust layout
       plt.tight_layout()
       
       # Create filename with path
       filename = f"{prefix}_umap_{color}.pdf" if prefix else f"umap_{color}.pdf"
       save_file = save_path / filename
       
       # Save with PDF to maintain vector properties of non-rasterized elements
       fig.savefig(save_file, dpi=dpi, bbox_inches='tight')
       plt.close(fig)  # Clean up


##################################################

def save_figure_rasterized_data(fig, filename, dpi=300):
    """
    Save a figure with rasterized data elements (heatmaps, scatter points, lines) 
    while keeping axes, text, and other elements as vectors.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Path where to save the figure. Should end in .pdf, .eps, or .svg
    dpi : int, optional
        Resolution for rasterized elements. Default = 300
    
    Example:
    --------
    # Works for heatmaps:
    fig = create_nmf_dynamics_heatmap_grid(mean_W, cv_W)
    save_figure_rasterized_data(fig, 'heatmap.pdf')
    
    # Also works for scatter plots:
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    save_figure_rasterized_data(fig, 'scatter.pdf')
    """
    # Store current fonttype settings
    old_pdf = plt.rcParams['pdf.fonttype']
    old_ps = plt.rcParams['ps.fonttype']
    
    try:
        # Rasterize data elements in each axis
        for ax in fig.axes:
            # Rasterize collections (heatmaps, scatter points)
            for collection in ax.collections:
                collection.set_rasterized(True)
                
            # Rasterize lines (plot lines, etc)
            for line in ax.lines:
                line.set_rasterized(True)
                
            # Rasterize images (imshow)
            for im in ax.images:
                im.set_rasterized(True)

        # Set text to remain as fonts, not outlines
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        # Save figure
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        
    finally:
        # Restore original settings
        plt.rcParams['pdf.fonttype'] = old_pdf
        plt.rcParams['ps.fonttype'] = old_ps


##################################################
##################################################
##################################################


def plot_cluster_markers_heatmap(
    adata,
    n_genes_per_cluster=5,
    min_logfc=1,
    layer='log',
    figsize=(10, 12)
):
    """
    Generate a heatmap of top marker genes for each Leiden cluster.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with leiden clusters in .obs['leiden']
    n_genes_per_cluster : int, optional (default: 5)
        Number of top genes to show per cluster
    min_logfc : float, optional (default: 1)
        Minimum log fold change for marker genes
    layer : str, optional (default: 'log')
        Layer containing normalized counts
    figsize : tuple, optional (default: (10, 12))
        Figure size for the heatmap
        
    Returns
    -------
    matplotlib.figure.Figure
        Heatmap figure showing cluster-enriched genes
    """
    import scanpy as sc
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Check if rank_genes_groups results exist
    if 'cluster_markers' not in adata.uns:
        # Run rank_genes_groups using wilcoxon test
        sc.tl.rank_genes_groups(
            adata,
            groupby='leiden',
            method='wilcoxon',
            key_added='cluster_markers',
            layer=layer
        )
    
    # Get results dataframe
    marker_df = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key='cluster_markers',
        pval_cutoff=0.05
    )
    
    # Filter for significant markers meeting thresholds
    marker_df = marker_df[
        (marker_df['logfoldchanges'] > min_logfc) &
        (marker_df['pvals_adj'] < 0.05)  # Add FDR filter instead of pct
    ]
    
    # Get top n genes per cluster
    top_genes = []
    for cluster in adata.obs['leiden'].unique():
        cluster_genes = marker_df[marker_df['group'] == cluster]
        top_cluster_genes = cluster_genes.nlargest(n_genes_per_cluster, 'logfoldchanges')
        top_genes.extend(top_cluster_genes['names'].tolist())
    
    # Get expression matrix for top genes
    expr_matrix = adata[:, top_genes].layers[layer].toarray()
    
    # Calculate mean expression per cluster
    cluster_means = []
    for cluster in sorted(adata.obs['leiden'].unique()):
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_mean = expr_matrix[cluster_mask].mean(axis=0)
        cluster_means.append(cluster_mean)
    
    cluster_means = np.array(cluster_means)
    
    # Scale the mean expression values
    scaled_means = (cluster_means - cluster_means.mean(axis=0)) / cluster_means.std(axis=0)
    
    # Order genes by cluster of maximum expression and then by maximum value
    max_cluster = np.argmax(scaled_means, axis=0)
    max_values = np.max(scaled_means, axis=0)
    
    # Create ordering array with cluster as primary key and -max_value as secondary key
    ordering = max_cluster * 1e6 - max_values  # This ensures cluster ordering takes precedence
    gene_order = np.argsort(ordering)
    
    # Reorder the scaled means and gene labels
    scaled_means = scaled_means[:, gene_order]
    ordered_genes = [top_genes[i] for i in gene_order]
    
    # Create heatmap with ordered rows and columns
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        scaled_means,
        xticklabels=ordered_genes,
        yticklabels=np.arange(len(cluster_means)),  # 0,1,2,... clusters
        cmap='RdBu_r',
        center=0,
        ax=ax
    )
    
    plt.xlabel('Genes')
    plt.ylabel('Leiden Clusters')
    plt.title(f'Top {n_genes_per_cluster} marker genes per cluster')
    plt.xticks(rotation=90, ha='right',fontsize=6)
    
    return fig


#########################################################################

def plot_nmf_programs_heatmap(H, gene_names, prog_list, n_genes=20, figsize=(12, 8), kwargs_heatmap={}):
    """
    Create a heatmap showing gene loadings across specified programs, selecting the top n_genes per program.
    
    Parameters:
    -----------
    H : numpy.ndarray
        The NMF H matrix (programs × genes)
    gene_names : list or numpy.ndarray
        List of gene names corresponding to columns in H
    prog_list : list
        List of program indices to visualize
    n_genes : int
        Number of top genes to show per program
    figsize : tuple
        Figure size (width, height)
    """
    # Identify the program where each gene has the highest loading
    gene_top_program = np.argmax(H, axis=0)  # Index of max loading program per gene
    
    # Create a DataFrame of loadings
    heatmap_df = pd.DataFrame(H[prog_list, :].T, 
                              index=gene_names, 
                              columns=[f'Program {i}' for i in prog_list])
    
    # Select top genes per program (without duplication)
    selected_genes = []
    for prog in prog_list:
        genes_in_prog = np.where(gene_top_program == prog)[0]  # Get genes assigned to this program
        sorted_genes = genes_in_prog[np.argsort(-H[prog, genes_in_prog])]  # Sort by descending loading
        selected_genes.extend(sorted_genes[:n_genes])  # Take top n_genes
    
    # Remove duplicates while preserving order
    selected_genes = list(dict.fromkeys(selected_genes))  # Ordered unique genes
    # Reorder heatmap data
    heatmap_df = heatmap_df.loc[np.array(gene_names)[selected_genes]]
    
    # Create figure and axes objects explicitly
    fig, ax = plt.subplots(figsize=figsize)

    
    # Plot heatmap with customized colorbar
    sns.heatmap(heatmap_df, 
                cmap='coolwarm', #'YlOrRd', 
                xticklabels=True, 
                yticklabels=True,
                ax=ax,
                cbar_kws={'shrink': 0.25,  # Make colorbar 50% of the height
                         'aspect': 40,     # Make colorbar thinner
                         'label': 'Loading'},
                **kwargs_heatmap
               )
    
    plt.yticks(fontsize=6)  # Adjust the font size of gene names (yticklabels)
    plt.title('Gene Loading Heatmap Across Programs')
    plt.tight_layout()
    
    return fig



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_nmf_programs_heatmap_horizontal(H, gene_names, prog_list, n_genes=20, min_val=0.3, figsize=(12, 8), kwargs_heatmap={}):
    gene_top_program = np.argmax(H, axis=0)
    
    heatmap_df = pd.DataFrame(H[prog_list, :],
                             index=[f'Program {i}' for i in prog_list],
                             columns=gene_names)
    
    selected_genes = []
    for prog in prog_list:
        genes_in_prog = np.where(gene_top_program == prog)[0]
        sorted_genes = genes_in_prog[np.argsort(-H[prog, genes_in_prog])]
        # Filter by min_val first
        filtered_genes = [gene for gene in sorted_genes if H[prog, gene] >= min_val]
        # Take top n_genes AFTER filtering
        top_genes = filtered_genes[:n_genes]
        selected_genes.extend(top_genes)
    
    # Remove duplicates while preserving order
    selected_genes = list(dict.fromkeys(selected_genes))
    selected_gene_names = [gene_names[i] for i in selected_genes]
    
    heatmap_df = heatmap_df[selected_gene_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_df,
                cmap='coolwarm',
                xticklabels=True,
                yticklabels=True,
                ax=ax,
                cbar_kws={'shrink': 0.7,
                         'aspect': 30,
                         'label': 'Loading'},
                **kwargs_heatmap)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    plt.title('Gene Loading Heatmap Across Programs', fontsize=14)
    plt.tight_layout()
    return fig



#########################

def create_nmf_dynamics_heatmap_grid(mean_W, cv_W, fano_W, sorted_programs=None, figsize=(6,12),conditions=None):
    # If no sorting provided, sort by dynamic change from time 0
    if sorted_programs is None:
        log_fc = np.abs(np.log(mean_W['Ctrl']/mean_W['Ctrl'].iloc[0]))
        max_deviation = log_fc.max()
        sorted_programs = max_deviation.sort_values(ascending=True).index
        #print(max_deviation)
        #print(sorted_programs)
    
    # Create figure with GridSpec to allow for colorbar space
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 0.1])

    if conditions==None:
        conditions = ['Ctrl', 'Vor', 'Aza', 'Dec']

    metrics = ['mean', 'cv', 'fano']

    n_conditions = len(conditions)

    # # Assuming your original dictionary is called 'dict_of_dfs'
    # mean_ctrl_dict = {key: df for key, df in mean_W.items() if 'Ctrl' in key}
    # cv_ctrl_dict = {key: df for key, df in cv_W.items() if 'Ctrl' in key}
    # fano_ctrl_dict = {key: df for key, df in fano_W.items() if 'Ctrl' in key}

    # mean_W_avg_ctrl = sum(mean_ctrl_dict.values()) / len(mean_ctrl_dict)
    # cv_W_avg_ctrl = sum(cv_ctrl_dict.values()) / len(cv_ctrl_dict)
    # fano_W_avg_ctrl = sum(fano_ctrl_dict.values()) / len(fano_ctrl_dict)
       
    data_dict = {
        'mean': mean_W, 
        'cv': cv_W,
        'fano': fano_W
    }
    
    titles = {
        'mean': r"$\log2(\mu/\mu_0^{Ctrl})$",
        'cv': r"$\log2(CV/CV_0^{Ctrl})$",
        'fano': r"$\log2(F/F_0^{Ctrl})$"
    }
    vmins = {'mean': -4, 'cv': -2, 'fano': -2}
    vmaxs = {'mean': 4, 'cv': 2, 'fano': 2}
    
    # Create heatmaps
    for i, metric in enumerate(metrics):
        # Create colorbar axis
        cbar_ax = fig.add_subplot(gs[i, n_conditions])
        
        # Get control t0 values for normalization
        ctrl_t0 = data_dict[metric]['Ctrl'].iloc[0]
        
        for j, cond in enumerate(conditions):
            ax = fig.add_subplot(gs[i, j])
            
            # Get data and normalize by control t0
            data = data_dict[metric][cond]
            data_norm = data.divide(ctrl_t0, axis=1)  # Normalize first
            data_sorted = data_norm[list(sorted_programs)].copy()   # Sort second
            #print(data.columns)
            #print(data[sorted_programs].columns)
            #print(data_sorted.columns)
            
            # Create heatmap
            sns.heatmap(np.log2(data_sorted.T), 
                       cmap="coolwarm", 
                       ax=ax, 
                       cbar=(j == 3),
                       cbar_ax=cbar_ax if j == 3 else None,
                       vmin=vmins[metric],
                       vmax=vmaxs[metric],
                       xticklabels=True,
                       yticklabels=True)
            
            # Set titles
            if i == 0:
                ax.set_title(f"{cond}")
            else:
                ax.set_title(f"{cond}")
                
            # Set x labels
            ax.set_xlabel("Timepoint")
            
            # Only keep y labels for leftmost plots
            if j > 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.tick_params(axis='y', labelsize=8)
            
            # Set colorbar label
            if j == 3:
                cbar_ax.set_ylabel(f'{titles[metric]}')
    
    plt.tight_layout()
    return fig



def create_nmf_dynamics_control_normalized_heatmap_grid(mean_W, cv_W, fano_W, sort_by='mean', figsize=(6,12), conditions=None):
    if conditions is None:
        conditions = ['Vor', 'Aza', 'Dec']
    
    # Calculate sorting metrics
    def get_sorted_programs(metric_W, conditions):
        avg_fold_changes = []
        ctrl_data = metric_W['Ctrl']
        
        for cond in conditions:
            treatment_data = metric_W[cond]
            fold_change = treatment_data.divide(ctrl_data)
            log2_fc = np.log2(fold_change)
            avg_fold_changes.append(log2_fc)
            
        mean_log2_fc = sum(avg_fold_changes) / len(avg_fold_changes)
        avg_magnitude = mean_log2_fc.mean()
        return avg_magnitude.sort_values(ascending=False).index
    
    # Get sorted programs based on selected metric
    if sort_by == 'mean':
        sorted_programs = get_sorted_programs(mean_W, conditions)
    elif sort_by == 'fano':
        sorted_programs = get_sorted_programs(fano_W, conditions)
    elif sort_by == 'cv':
        sorted_programs = get_sorted_programs(cv_W, conditions)
    else:
        raise ValueError("sort_by must be one of: 'mean', 'fano', 'cv'")

    # Create figure with GridSpec to allow for colorbar space
    fig = plt.figure(figsize=figsize)
    metrics = ['mean', 'cv', 'fano']
    n_conditions = len(conditions)
    gs = gridspec.GridSpec(3, n_conditions + 1, width_ratios=[1]*n_conditions + [0.1])

    data_dict = {
        'mean': mean_W,
        'cv': cv_W,
        'fano': fano_W
    }
    
    titles = {
        'mean': r"$\log2(\mu/\mu_{Ctrl})$",
        'cv': r"$\log2(CV/CV_{Ctrl})$",
        'fano': r"$\log2(F/F_{Ctrl})$"
    }
    
    vmins = {'mean': -4, 'cv': -2, 'fano': -2}
    vmaxs = {'mean': 4, 'cv': 2, 'fano': 2}

    # Create heatmaps
    for i, metric in enumerate(metrics):
        cbar_ax = fig.add_subplot(gs[i, -1])
        ctrl_data = data_dict[metric]['Ctrl']
        
        for j, cond in enumerate(conditions):
            ax = fig.add_subplot(gs[i, j])
            
            treatment_data = data_dict[metric][cond]
            data_norm = treatment_data.divide(ctrl_data)
            data_sorted = data_norm[list(sorted_programs)].copy()
            
            sns.heatmap(np.log2(data_sorted.T),
                       cmap="coolwarm",
                       ax=ax,
                       cbar=(j == len(conditions)-1),
                       cbar_ax=cbar_ax if j == len(conditions)-1 else None,
                       vmin=vmins[metric],
                       vmax=vmaxs[metric],
                       xticklabels=True,
                       yticklabels=True)
            
            if i == 0:
                ax.set_title(f"{cond}")
                
            ax.set_xlabel("Timepoint")
            
            if j > 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.tick_params(axis='y', labelsize=8)
            
            if j == len(conditions)-1:
                cbar_ax.set_ylabel(f'{titles[metric]}')
    
    plt.tight_layout()
    return fig


####################################################################################
####################################################################################


# def create_trajectory_plots(mean_W, cv_W, sorted_programs=None):
#     """
#     Create trajectory plots showing mean vs CV relationships across conditions and timepoints.
    
#     Parameters:
#     -----------
#     mean_W : dict
#         Dictionary of DataFrames containing mean values for each condition
#     cv_W : dict
#         Dictionary of DataFrames containing coefficient of variation values for each condition
#     sorted_programs : list-like, optional
#         Order of programs to plot. If None, will use the order from mean_W

        
#     Returns:
#     --------
#     matplotlib.figure.Figure
#         The created figure object
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Define plot settings
#     conditions = ['Ctrl', 'Vor', 'Aza', 'Dec']
#     condition_colors = {'Ctrl': 'black', 'Vor': 'red', 'Aza': 'blue', 'Dec': 'green'}
#     timepoints = mean_W['Ctrl'].index.values
#     alpha_values = np.linspace(0.1, 1.0, len(timepoints))
    
#     # Use provided program order or get from data
#     if sorted_programs is None:
#         sorted_programs = mean_W['Ctrl'].columns
    
#     # Define grid layout
#     num_programs = len(sorted_programs)
#     n_cols = 4
#     n_rows = int(np.ceil(num_programs / n_cols))
    
#     # Create figure with subplots
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
#     axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
#     for idx, prog in enumerate(sorted_programs):
#         ax = axes[idx]
        
#         # Plot each condition
#         for cond in conditions:
#             # Get data normalized to Ctrl t=0
#             x = np.log10(mean_W[cond][prog].values / mean_W['Ctrl'].loc[0, prog])
#             y = cv_W[cond][prog].values
            
#             # Plot line with full alpha for legend
#             ax.plot([], [], color=condition_colors[cond], label=cond, alpha=1.0)
            
#             # Scatter plot with increasing alpha values
#             for i in range(len(x)):
#                 ax.scatter(x[i], y[i], color=condition_colors[cond], 
#                           alpha=alpha_values[i], edgecolors=None)
            
#             # Add arrows for time progression
#             for i in range(len(x) - 1):
#                 ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
#                     arrowprops=dict(arrowstyle='->', color=condition_colors[cond], 
#                                    alpha=1.0, lw=1.0, mutation_scale=10))
        
#         # Set subplot properties
#         ax.set_title(prog, fontsize=10)
#         ax.set_xlabel(r"$log_{10}(\mu/\mu_0^{Ctrl})$", fontsize=8)
#         ax.set_ylabel("CV", fontsize=8)
#         ax.tick_params(axis='both', labelsize=8)
#         ax.set_xlim(-1.5, 1.25)
        
#         # Add legend only to first subplot
#         if idx == 0:
#             ax.legend(fontsize=8)
    
#     # Hide any empty subplots
#     for i in range(num_programs, len(axes)):
#         fig.delaxes(axes[i])
    
#     plt.tight_layout()
    

#     return fig


####################################################################################
####################################################################################


def create_single_trajectory_plot(mean_W, cv_W, programs, conditions, 
                                  figsize=(4,4), ylabel=None):
    """
    Create a single trajectory plot showing mean vs CV relationships.
    Either overlay multiple programs for one condition, or multiple conditions for one program.
    
    Parameters:
    -----------
    mean_W : dict
        Dictionary of DataFrames containing mean values for each condition
    cv_W : dict
        Dictionary of DataFrames containing coefficient of variation values for each condition
    programs : str or list
        Either a single program name (str) or list of programs to overlay
    conditions : str or list
        Either a single condition name (str) or list of conditions to overlay
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
        
    Notes:
    ------
    One argument must be a string and the other a list to determine plotting mode

    Example usage:
    --------------
    # Plot multiple programs for one condition
    fig = create_single_trajectory_plot(mean_W, cv_W,
                                      programs=['Program1', 'Program2', 'Program3'],
                                      conditions='Ctrl')
    
    # Plot multiple conditions for one program
    fig = create_single_trajectory_plot(mean_W, cv_W,
                                      programs='Program1',
                                      conditions=['Ctrl', 'Vor', 'Aza'])
    
    plt.show()

    
    """

    # Input validation
    if isinstance(programs, list) and isinstance(conditions, list):
        raise ValueError("One of programs or conditions must be a single string")
    if isinstance(programs, str) and isinstance(conditions, str):
        raise ValueError("One of programs or conditions must be a list")
        
    # Set up plotting parameters
    timepoints = mean_W['Ctrl'].index.values
    alpha_values = np.linspace(0.2, 1.0, len(timepoints))
    #colors = plt.cm.tab10(np.linspace(0, 1, 10))
    # Create black + tab10 colormap
    colors = ['k'] + [plt.cm.tab10(i) for i in np.linspace(0, 1, 10)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(programs, list):
        # Multiple programs, single condition mode
        for idx, prog in enumerate(programs):
            # Get data normalized to Ctrl t=0
            x = np.log10(mean_W[conditions][prog].values / mean_W['Ctrl'].loc[0, prog])
            y = cv_W[conditions][prog].values
            color = colors[idx % 10]
            
            ax.plot([], [], color=color, label=prog, alpha=1.0)
            for i in range(len(x)):
                ax.scatter(x[i], y[i], color=color, alpha=alpha_values[i], 
                           edgecolors=None, s=50)
            for i in range(len(x) - 1):
                ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle='->', color=color, 
                                   alpha=1.0, lw=1.0, mutation_scale=10))
        
        title = f'Multiple Programs - {conditions} Condition'
        
    else:
        # Multiple conditions, single program mode
        for idx, cond in enumerate(conditions):
            # Get data normalized to Ctrl t=0
            x = np.log10(mean_W[cond][programs].values / mean_W['Ctrl'].loc[0, programs])
            y = cv_W[cond][programs].values
            color = colors[idx % 10]
            
            ax.plot([], [], color=color, label=cond, alpha=1.0)
            for i in range(len(x)):
                ax.scatter(x[i], y[i], color=color, alpha=alpha_values[i], 
                           edgecolors=None, s=50)
            for i in range(len(x) - 1):
                ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle='->', color=color, 
                                   alpha=1.0, lw=1.0, mutation_scale=10))
        
        title = f'Multiple Conditions - Program {programs}'
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$log_{10}(\mu/\mu_0^{Ctrl})$", fontsize=10)
    if ylabel == None:
        ax.set_ylabel("CV", fontsize=10)
    else:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    #ax.set_xlim(-1.5, 1.25)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

################################################################################
################################################################################
################################################################################

def plot_program_cv_histograms(cv_W, conditions, figsize=(12, 3),xlabel='CV'):
    # Create figure with subplots for each timepoint
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    times = [0, 2, 4, 6]
    
    # Loop through timepoints
    for i, time in enumerate(times):
        ax = axes[i]
        
        # Plot histogram for each condition at this timepoint
        for cond in conditions:
            # Get CV values for all programs at this timepoint
            cv_values = cv_W[cond].iloc[i]
            
            # Plot histogram
            ax.hist(cv_values, alpha=0.5, label=cond, bins=15)
            
        # Customize plot
        ax.set_title(f'{time} days')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count' if i == 0 else '')
        
        # Only show legend in first plot
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    return fig

################################################################################


def plot_program_cv_boxplots(cv_W, conditions, ylabel, figsize=(8, 3)):
    # Create figure with subplots for each timepoint
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    times = [0, 2, 4, 6]
    
    # Find global min and max for consistent y-axis
    all_values = []
    for cond in conditions:
        all_values.extend(cv_W[cond].values.flatten())
    ymin, ymax = min(all_values), max(all_values)
    
    # Loop through timepoints
    for i, time in enumerate(times):
        ax = axes[i]
        
        # Collect data for boxplot
        data_to_plot = [cv_W[cond].iloc[i] for cond in conditions]
        
        # Create boxplot
        ax.boxplot(data_to_plot, labels=conditions)
        
        # Customize plot
        if time==0:
            ax.set_title(f'Single cells (day 0)')
        else:
            ax.set_title(f'Clones (day {time})')
        ax.set_ylabel(ylabel if i == 0 else '')
        ax.set_ylim(0, ymax)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

# Usage example:
# plot_program_cv_boxplots(cv_W, conditions=['Ctrl', 'Aza'], ylabel='Coefficient of Variation')



################################################################################

def plot_program_boxplot_time_series(cv_W_dict, ylabel='CV', figsize=(3, 3), colors=None, 
                                   box_width=0.15, group_spacing=1.5):
    """
    Plot boxplots of time series data with customizable spacing
    
    Parameters:
    -----------
    cv_W_dict : dict
        Dictionary of DataFrames to plot
    ylabel : str, default='CV'
        Y-axis label
    figsize : tuple, default=(3, 3)
        Figure size in inches
    colors : list, optional 
        Colors for each series. If None, uses default colormap
    box_width : float, default=0.15
        Width of each individual box
    group_spacing : float, default=1.5
        Multiplier for spacing between time point groups. 
        Higher values = more space between groups
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    times = [0, 2, 4, 6]
    
    # Set default colors if none provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(cv_W_dict)))
    
    # Calculate spacing parameters
    n_groups = len(times)
    n_items = len(cv_W_dict)
    group_width = box_width * n_items * group_spacing
    
    # Calculate offsets for items within each group
    offsets = np.linspace(-group_width/2 + box_width/2, 
                         group_width/2 - box_width/2, 
                         n_items)
    
    # Plot each series
    bp_objects = []
    for (label, data), offset, color in zip(cv_W_dict.items(), offsets, colors):
        # Create positions and data for this series
        valid_positions = []
        plot_data = []
        
        # Get the actual time points from the data's index
        data_times = data.index.tolist()
        
        # Create data arrays for boxplot
        for time in times:
            if time in data_times:
                valid_positions.append(time + offset)
                idx = data_times.index(time)
                plot_data.append(data.iloc[idx])
        
        # Create boxplot
        bp = ax.boxplot(plot_data,
                       positions=valid_positions,
                       widths=box_width,
                       whis=(5, 95),
                       patch_artist=True,
                       medianprops=dict(color="black"),
                       flierprops={'marker': 'none'})
        
        # Color the boxes
        for box in bp['boxes']:
            box.set_facecolor(color)
            box.set_alpha(0.7)
            
        # Store boxplot objects for legend
        bp_objects.append(bp)
    
    # Customize plot
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, 7)
    
    # Set x-axis ticks and labels
    ax.set_xticks(times)
    ax.set_xticklabels([str(t) for t in times])
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add legend outside the plot
    ax.legend([bp['boxes'][0] for bp in bp_objects], 
             list(cv_W_dict.keys()),
             bbox_to_anchor=(1.05, 1), 
             loc='upper left')
    
    plt.tight_layout()
    return fig

################################################################################
################################################################################


import numpy as np
import matplotlib.pyplot as plt

def plot_gene_usage(H, gene_names, gene, prog_list=None):
    """
    Plot the usage of a given gene across programs, normalized to its maximum loading.
    Parameters:
    -----------
    H : numpy.ndarray
        The NMF H matrix (programs × genes)
    gene_names : list or numpy.ndarray
        List of gene names corresponding to columns in H
    gene : str
        The gene to visualize
    prog_list : list, optional
        List of program indices to visualize (default is all programs)
    """
    # Convert gene_names to numpy array if it isn't already
    gene_names = np.array(gene_names)
    
    # Case-insensitive search
    gene_idx = np.where(np.char.lower(gene_names) == gene.lower())[0]
    
    if len(gene_idx) == 0:
        raise ValueError(f"Gene '{gene}' not found in gene_names.")
    
    gene_idx = gene_idx[0]  # Take the first match
    gene_loadings = H[:, gene_idx]  # Get loadings across programs
    
    if prog_list is None:
        prog_list = np.arange(H.shape[0])  # Use all programs if not specified
    
    gene_loadings = gene_loadings[prog_list]  # Filter by selected programs
    gene_loadings /= gene_loadings.max()  # Normalize to max
    
    plt.figure(figsize=(10, 4))
    plt.bar([f'Program {i}' for i in prog_list], gene_loadings, color='dodgerblue', alpha=0.7)
    plt.xlabel("Programs")
    plt.ylabel("Normalized Usage")
    plt.title(f"Usage of {gene} Across Programs")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_gene_cv_vs_mean(ad_all, gene_cv_mean_dict, rnd_gene_cv_mean_dict, condition):
    """
    Create CV vs mean plots for each condition and timepoint.
    
    Parameters:
    -----------
    ad_all : AnnData
        Annotated data object containing condition information
    gene_cv_mean_dict : dict
        Dictionary containing CV and mean values for genes per condition and timepoint
    rnd_gene_cv_mean_dict : dict
        Dictionary containing CV and mean values for random genes per condition and timepoint
    cell_type : str
        Cell type identifier for saving the figure
    save_path : str
        Base path for saving the figures
        
    Returns:
    --------
    None
    """
    
    import matplotlib.pyplot as plt
    c = condition    

    # Create subplot
    fig, ax = plt.subplots(1, 4, figsize=(13, 4))
    
    # Loop through timepoints for this condition
    for i, t in enumerate(gene_cv_mean_dict[c].keys()):
        # Plot all genes
        ax[i].scatter(
            gene_cv_mean_dict[c][t]['mean'],
            gene_cv_mean_dict[c][t]['cv'],
            c=gene_cv_mean_dict[c][t]['norm_var'],
            s=20,
            marker='o',
            alpha=0.3,
            cmap='coolwarm',
            label='Observed clones'
        )
        
        # Set scales
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        
        # Plot random genes if timepoint > 0
        if t > 0:
            ax[i].loglog(
                rnd_gene_cv_mean_dict[c][t]['mean'],
                rnd_gene_cv_mean_dict[c][t]['cv'],
                marker='.',
                color='grey',
                alpha=0.2,
                markersize=1,
                label='Mock clones',
                linestyle='None'
            )
        
        # Set labels and title
        ax[i].set_ylabel('CV')
        ax[i].set_xlabel('Mean')
        ax[i].set_title(f'{c} t={t}')
        ax[i].legend()
        
        # Adjust layout
        plt.tight_layout()
        
    return fig



def plot_weighted_cv_vs_mean(gene_cv_mean_dict, rnd_gene_cv_mean_dict, ad_all, conditions=['Ctrl-2','Ctrl-3'], timepoints=[4,6]):
    """
    Create a single CV vs mean plot with cell-count weighted averaging across specified conditions and timepoints.
    Overlays both real and randomized data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Calculate cell counts for each condition and timepoint
    cell_counts = pd.DataFrame({
        'condition': ad_all.obs['condition'],
        'timepoint': ad_all.obs['timepoint']
    }).groupby(['condition', 'timepoint']).size()
    
    def compute_weighted_stats(data_dict, include_norm_vars=False):
        # Initialize arrays
        n_genes = len(ad_all.var_names)
        all_means = np.zeros((n_genes, len(conditions) * len(timepoints)))
        all_vars = np.zeros((n_genes, len(conditions) * len(timepoints)))
        if include_norm_vars:
            all_norm_vars = np.zeros((n_genes, len(conditions) * len(timepoints)))
        weights = np.zeros(len(conditions) * len(timepoints))
        
        # Collect values and weights across conditions and timepoints
        idx = 0
        for cond in conditions:
            for tp in timepoints:
                if tp in data_dict[cond]:
                    try:
                        weight = cell_counts.loc[(cond, tp)]
                    except KeyError:
                        continue
                    
                    all_means[:, idx] = data_dict[cond][tp]['mean']
                    all_vars[:, idx] = data_dict[cond][tp]['var']
                    if include_norm_vars:
                        all_norm_vars[:, idx] = data_dict[cond][tp]['norm_var']
                    weights[idx] = weight
                    idx += 1
        
        # Normalize weights and compute averages
        valid_weights = weights[:idx]
        normalized_weights = valid_weights / valid_weights.sum()
        
        mean_means = np.average(all_means[:, :idx], axis=1, weights=normalized_weights)
        mean_vars = np.average(all_vars[:, :idx], axis=1, weights=normalized_weights)
        mean_cvs = np.sqrt(mean_vars) / mean_means
        
        if include_norm_vars:
            mean_norm_vars = np.average(all_norm_vars[:, :idx], axis=1, weights=normalized_weights)
            return mean_means, mean_cvs, mean_norm_vars
        return mean_means, mean_cvs
    
    # Compute stats for both real and randomized data
    real_means, real_cvs, real_norm_vars = compute_weighted_stats(gene_cv_mean_dict, include_norm_vars=True)
    rnd_means, rnd_cvs = compute_weighted_stats(rnd_gene_cv_mean_dict)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(3.8, 3))
    

    
    # Plot real data with color mapping
    # Sort indices by normalized variance to plot higher values on top
    sort_idx = np.argsort(real_norm_vars)
    
    # Plot real data with color mapping using sorted indices
    scatter = ax.scatter(
        real_means[sort_idx],
        real_cvs[sort_idx],
        c=real_norm_vars[sort_idx],
        s=20,
        marker='o',
        alpha=0.9,
        cmap='coolwarm',
        label='Observed'
    )

    # Plot randomized data in grey
    ax.scatter(
        rnd_means,
        rnd_cvs,
        color='grey',
        s=10,
        marker='o',
        alpha=0.5,
        label='Randomized'
    )
    # Add colorbar
    plt.colorbar(scatter, label='Normalized variance')
    
    # Set scales and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('CV (from weighted mean/var)')
    ax.set_xlabel('Mean (cell-count weighted)')
    
    # Set title
    #conditions_str = ', '.join(conditions)
    #timepoints_str = ', '.join(map(str, timepoints))
    #ax.set_title(f'CV vs Mean\nCell-count weighted average across\nconditions: {conditions_str}\ntimepoints: {timepoints_str}')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    return fig, ax



def plot_weighted_gene_correlations(gene_cv_mean_dict, ad_all, conditions=['Ctrl-2','Ctrl-1','Ctrl-3'], 
                                  timepoints=[4,6], n_top_genes=500, vmin=-0.2, vmax=0.2,
                                  exclude_prefix_file=None, exclude_suffix_file=None,
                                  clustering_method='ward',  # Try 'ward', 'complete', or 'average'
                                  min_corr_threshold=None,  # e.g., 0.3 to focus on stronger correlations
                                  use_absolute_corr=False): # Use absolute values for clustering
    """
    Calculate and plot correlation matrix for top variable genes, using cell-count weighted averaging
    across multiple timepoints, with optional gene filtering.
    
    Parameters:
    -----------
    gene_cv_mean_dict : dict
        Dictionary containing mean, variance, and norm_var values for genes per condition and timepoint
    ad_all : AnnData
        Annotated data object containing condition and timepoint information
    conditions : list
        List of conditions to analyze
    timepoints : list
        List of timepoints to analyze
    n_top_genes : int
        Number of top variable genes to include
    vmin, vmax : float
        Color scale limits for correlation matrix
    exclude_prefix_file : str, optional
        Path to file containing gene prefixes to exclude (one per line)
    exclude_suffix_file : str, optional
        Path to file containing gene suffixes to exclude (one per line)
    
    Returns:
    --------
    fig : matplotlib figure
    ordered_corr : numpy array
        Ordered correlation matrix
    top_gene_names : list
        Names of selected top variable genes in order
    """
    import numpy as np
    from scipy import sparse
    import matplotlib.pyplot as plt
    from scipy import cluster
    from scipy.spatial import distance  
    import pandas as pd
    
    # Load exclusion patterns if provided
    exclude_patterns = []
    if exclude_prefix_file:
        with open(exclude_prefix_file, 'r') as f:
            prefixes = [line.strip() for line in f if line.strip()]
            exclude_patterns.extend([f"^{p}" for p in prefixes])
    
    if exclude_suffix_file:
        with open(exclude_suffix_file, 'r') as f:
            suffixes = [line.strip() for line in f if line.strip()]
            exclude_patterns.extend([f"{s}$" for s in suffixes])
    
    # Create gene mask if exclusion patterns exist
    import re
    if exclude_patterns:
        combined_pattern = '|'.join(exclude_patterns)
        gene_mask = ~np.array([bool(re.search(combined_pattern, gene)) 
                              for gene in ad_all.var_names])
        print(f"Filtered out {np.sum(~gene_mask)} genes based on patterns")
    else:
        gene_mask = np.ones(len(ad_all.var_names), dtype=bool)
    
    # Calculate cell counts for weighting
    cell_counts = pd.DataFrame({
        'condition': ad_all.obs['condition'],
        'timepoint': ad_all.obs['timepoint']
    }).groupby(['condition', 'timepoint']).size()
    
    # Get weights for each condition-timepoint combination
    weights = []
    for cond in conditions:
        for tp in timepoints:
            weights.append(cell_counts.loc[(cond, tp)])
    normalized_weights = np.array(weights) / np.sum(weights)
    
    # Select top variable genes based on weighted average of norm_var
    all_norm_vars = []
    for cond in conditions:
        for tp in timepoints:
            if tp in gene_cv_mean_dict[cond]:
                all_norm_vars.append(gene_cv_mean_dict[cond][tp]['norm_var'])
    
    mean_norm_vars = np.average(np.array(all_norm_vars), weights=normalized_weights, axis=0)
    
    # Apply gene filtering before selecting top genes
    filtered_norm_vars = mean_norm_vars[gene_mask]
    filtered_gene_names = ad_all.var_names[gene_mask]
    
    top_var_indices = np.argsort(filtered_norm_vars)[-n_top_genes:]
    top_gene_names = filtered_gene_names[top_var_indices]
    
    # 
    ad_t = ad_all[ad_all.obs['timepoint'].isin(timepoints), top_gene_names].copy()
    
    n_genes = len(top_gene_names)
    weighted_covs = np.zeros((n_genes, n_genes))
    weighted_stds = np.zeros(n_genes)
    
    weight_idx = 0
    for cond in conditions:
        for tp in timepoints:
            mask = (ad_t.obs['condition'] == cond) & (ad_t.obs['timepoint'] == tp)
            if np.sum(mask) == 0:
                weight_idx += 1
                continue
                
            expr_matrix = ad_t[mask].X
            if sparse.issparse(expr_matrix):
                expr_matrix = expr_matrix.toarray()
                
            means = np.mean(expr_matrix, axis=0)
            centered = expr_matrix - means
            
            covs = np.dot(centered.T, centered) / (len(centered) - 1)
            stds = np.std(expr_matrix, axis=0)
            
            weight = normalized_weights[weight_idx]
            weighted_covs += weight * covs
            weighted_stds += weight * stds
            weight_idx += 1
    
    # Calculate correlation matrix from weighted covariances and standard deviations
    corr_matrix = weighted_covs / np.outer(weighted_stds, weighted_stds)
    
    # Prepare distance matrix for clustering
    if use_absolute_corr:
        dist_matrix = 1 - np.abs(corr_matrix)
    else:
        dist_matrix = 1 - corr_matrix
    
    # Ensure diagonal is exactly zero
    np.fill_diagonal(dist_matrix, 0)
        
    # Apply correlation threshold if specified
    if min_corr_threshold is not None:
        clustering_matrix = dist_matrix.copy()
        mask = np.abs(corr_matrix) < min_corr_threshold
        clustering_matrix[mask] = 1  # Set as maximum distance
        np.fill_diagonal(clustering_matrix, 0)  # Ensure diagonal remains zero
    else:
        clustering_matrix = dist_matrix
    
    # Perform clustering
    linkage = cluster.hierarchy.linkage(distance.squareform(clustering_matrix), 
                                      method=clustering_method)
    
    # Get cluster ordering
    idx = cluster.hierarchy.leaves_list(linkage)
    ordered_corr = corr_matrix[idx, :][:, idx]
    ordered_genes = top_gene_names[idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(ordered_corr, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    timepoints_str = ','.join(map(str, timepoints))
    ax.set_title(f'Gene correlations\n(top {n_top_genes} variable genes, {clustering_method} clustering)\n'
                 f'timepoints {timepoints_str}')
    plt.tight_layout()
    
    return fig, ordered_corr, ordered_genes

# Example usage:
# fig, corr_matrix, genes = plot_weighted_gene_correlations(
#     gene_cv_mean_dict=gene_cv_mean_dict,
#     ad_all=ad_all,
#     conditions=['Ctrl-2','Ctrl-1','Ctrl-3'],
#     timepoints=[4,6],
#     n_top_genes=500,
#     exclude_prefix_file='exclude_prefixes.txt',
#     exclude_suffix_file='exclude_suffixes.txt'
# )
# plt.show()



def plot_NMF_mean_cv_relationship(mean_dict, cv_dict,
                                rnd_mean_dict, rnd_cv_dict,
                                real_fit, rnd_fit, exclude_keys=['Ctrl']):
    """
    Create plot comparing CV vs Mean relationships.
    
    Parameters:
    -----------
    mean_dict, cv_dict : dict
        Dictionaries of dataframes containing real data
    rnd_mean_dict, rnd_cv_dict : dict
        Dictionaries of dataframes containing randomized data
    real_fit, rnd_fit : dict
        Dictionaries containing fit parameters from fit_cv_mean_relationship()
    exclude_keys : list
        Keys to exclude from plotting (e.g., ['Ctrl'])
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The created axes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create figure with same width as one panel from before
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    
    # Loop through all conditions
    first_plot = True  # Flag for first plot to add labels
    for c in mean_dict.keys():
        if c in exclude_keys:
            continue
            
        # Plot CV relationships
        for col in mean_dict[c].columns:
            if first_plot:
                ax.loglog(mean_dict[c][col], cv_dict[c][col],
                         'ob', alpha=0.3, markersize=5, label='Observed')
                ax.loglog(rnd_mean_dict[c][col], rnd_cv_dict[c][col],
                         'o', color='grey', alpha=0.3, markersize=5, label='Mock clones')
                first_plot = False
            else:
                ax.loglog(mean_dict[c][col], cv_dict[c][col],
                         'ob', alpha=0.3, markersize=5)
                ax.loglog(rnd_mean_dict[c][col], rnd_cv_dict[c][col],
                         'o', color='grey', alpha=0.3, markersize=5)
    
    # Add legend after all plotting is done
    ax.legend()
    
    # Add theoretical lines
    xTh = np.logspace(-2.6, -0.3, num=10)
    # Real data fits
    ax.plot(xTh, real_fit['a']*xTh**(-0.5), '--k')
    # Random data fits
    ax.plot(xTh, rnd_fit['a']*xTh**(-0.5), '--k', alpha=0.5)
    
    # Labels
    ax.set_xlabel("Mean NMF usage (total=1)", fontsize=12)
    ax.set_ylabel("CV NMF usage", fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig, ax



def plot_cv_ratio_timecourse(df_num, df_denom, title=None,
                            timepoints=[2, 4, 6], height=3, ylabel=None):
    """
    Plot timecourse of CV ratio between two sets of measurements.
    
    Parameters:
    -----------
    df_num : pandas.DataFrame
        DataFrame containing values for numerator
    df_denom : pandas.DataFrame
        DataFrame containing values for denominator
    title : str or None
        Optional title for the plot
    timepoints : list
        List of timepoints to plot on x-axis
    height : float
        Height of square subplot in inches
    ylabel : str or None
        Optional custom y-axis label. If None, defaults to 'CV ratio'
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The created axes
    """
    from matplotlib.cm import get_cmap
    import matplotlib.pyplot as plt
    
    # Create square figure
    fig, ax = plt.subplots(1, 1, figsize=(height, height))
    
    # Generate a colormap with unique colors for each factor
    cmap = get_cmap('tab20', len(df_num.columns))
    
    # Plot each factor
    for i, f in enumerate(df_num.columns):
        # Calculate ratio for timepoints
        y = list(df_num.loc[1:, f] / df_denom.loc[:, f])
        
        # Plot with lines and markers
        ax.plot(timepoints, y, 'o-', label=f'factor {f}', color=cmap(i))
        
        # Add factor label if ratio > 1 at final timepoint
        if y[-1] > 1.0:
            ax.text(timepoints[-1] + 0.1, y[-1], f, fontsize=12)
    
    # Add horizontal line at ratio=1
    ax.plot([timepoints[0], timepoints[-1]], [1, 1], 'k--')
    
    # Set title and labels with larger fonts
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel(ylabel if ylabel is not None else 'CV ratio', fontsize=14)
    
    # Set consistent limits
    ax.set_xlim(timepoints[0] - 0.3, timepoints[-1] + 0.5)
    
    # Make plotting area square
    ax.set_box_aspect(1)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    return fig, ax


def save_editable_pdf(fig, filename, dpi=300):
   """
   Save matplotlib figure as PDF with editable text.
   
   Args:
       fig: matplotlib figure object
       filename: output PDF filename
       dpi: resolution (default 300)
   """
   from matplotlib.backends.backend_pdf import PdfPages
   
   with PdfPages(filename, metadata={'Creator': 'Matplotlib'}) as pdf:
        matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
        matplotlib.rcParams['ps.fonttype'] = 42  # Ensures consistency for PostScript files
        pdf.savefig(fig, dpi=dpi, transparent=True)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_nmf_factor_histogram(adata, factor, sample_key_column, filter_column=None, filter_value=None, use_log=False, pseudocount=1e-3):
    """
    Plot a histogram of normalized NMF factor usage for a given factor in each condition.

    Parameters:
    adata: AnnData
        Annotated data matrix with `obsm['X_nmf']` containing NMF factors and `obs[sample_key_column]` specifying conditions.
    factor: int
        Index of the NMF factor to plot (0-based).
    sample_key_column: str
        Column in `adata.obs` containing the sample conditions.
    filter_column: str, optional
        Column in `adata.obs` to apply a filter.
    filter_value: Any, optional
        Value to filter `filter_column` by. Only rows matching this value will be included.
    """
    # Apply filter if specified
    if filter_column is not None and filter_value is not None:
        filter_mask = adata.obs[filter_column] == filter_value
        obs_filtered = adata.obs[filter_mask]
        obsm_filtered = adata.obsm['X_nmf'][filter_mask, :]
    else:
        obs_filtered = adata.obs
        obsm_filtered = adata.obsm['X_nmf']

    # Normalize X_nmf such that the sum of all programs equals 1 per cell
    nmf_normalized = obsm_filtered / obsm_filtered.sum(axis=1, keepdims=True)

    if use_log:
        nmf_normalized = np.log10(nmf_normalized + pseudocount)
  
    # Get sample key conditions
    conditions = obs_filtered[sample_key_column].unique()

    # Initialize a figure
    fig = plt.figure(figsize=(5, 5))

    bins=np.linspace(np.min(nmf_normalized[:, factor]), np.max(nmf_normalized[:, factor]), num=30)  
    
    # Plot histograms for each condition
    for condition in conditions:
        # Select cells belonging to the current condition
        condition_mask = obs_filtered[sample_key_column] == condition

        # Extract normalized NMF values for the specified factor
        factor_values = nmf_normalized[condition_mask, factor]
        #print(condition, np.mean(factor_values<0.005))   
        # Plot histogram
        plt.hist(
            factor_values,
            bins=bins,
            alpha=0.5,
            label=f"{condition} (n={np.sum(condition_mask)})",
            density=True
        )

    # Add labels and legend
    plt.xlabel(f"Normalized Usage of Factor {factor}")
    plt.ylabel("Density")
    plt.title(f"Histogram of Normalized Factor {factor} Usage by Condition")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    return fig

def plot_fraction_factor_below_threshold(adata, factor, sample_key_column, threshold, filter_column=None, filter_value=None):
    """
    Plot a bar graph of the fraction of cells with usage below a threshold for a given factor.

    Parameters:
    adata: AnnData
        Annotated data matrix with `obsm['X_nmf']` containing NMF factors and `obs[sample_key_column]` specifying conditions.
    factor: int
        Index of the NMF factor to analyze (0-based).
    sample_key_column: str
        Column in `adata.obs` containing the sample conditions.
    threshold: float
        Threshold value for the factor usage.
    filter_column: str, optional
        Column in `adata.obs` to apply a filter.
    filter_value: Any, optional
        Value to filter `filter_column` by. Only rows matching this value will be included.
    """
    # Apply filter if specified
    if filter_column is not None and filter_value is not None:
        filter_mask = adata.obs[filter_column] == filter_value
        obs_filtered = adata.obs[filter_mask]
        obsm_filtered = adata.obsm['X_nmf'][filter_mask, :]
    else:
        obs_filtered = adata.obs
        obsm_filtered = adata.obsm['X_nmf']

    nmf_normalized = obsm_filtered / obsm_filtered.sum(axis=1, keepdims=True)
    conditions = obs_filtered[sample_key_column].unique()

    fractions = []
    for condition in conditions:
        condition_mask = obs_filtered[sample_key_column] == condition
        factor_values = nmf_normalized[condition_mask, factor]
        fraction_below = np.mean(factor_values < threshold)
        #print(condition, fraction_below)
        fractions.append(fraction_below)

    fig = plt.figure(figsize=(5, 5))
    plt.bar(conditions, fractions, alpha=0.7)
    plt.xlabel("Condition")
    plt.ylabel("Fraction of Clones Below Threshold")
    plt.title(f"Fraction of Clones with Factor {factor} Usage < {threshold}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_mean_factor_expression(adata, factor, sample_key_column, threshold, filter_column=None, filter_value=None):
    """
    Plot bar graphs of mean expression for all cells and cells above a threshold for a given factor.

    Parameters:
    adata: AnnData
        Annotated data matrix with `obsm['X_nmf']` containing NMF factors and `obs[sample_key_column]` specifying conditions.
    factor: int
        Index of the NMF factor to analyze (0-based).
    sample_key_column: str
        Column in `adata.obs` containing the sample conditions.
    threshold: float
        Threshold value for the factor usage.
    filter_column: str, optional
        Column in `adata.obs` to apply a filter.
    filter_value: Any, optional
        Value to filter `filter_column` by. Only rows matching this value will be included.
    """
    # Apply filter if specified
    if filter_column is not None and filter_value is not None:
        filter_mask = adata.obs[filter_column] == filter_value
        obs_filtered = adata.obs[filter_mask]
        obsm_filtered = adata.obsm['X_nmf'][filter_mask, :]
    else:
        obs_filtered = adata.obs
        obsm_filtered = adata.obsm['X_nmf']

    nmf_normalized = obsm_filtered / obsm_filtered.sum(axis=1, keepdims=True)
    conditions = obs_filtered[sample_key_column].unique()

    mean_all = []
    mean_above = []
    for condition in conditions:
        condition_mask = obs_filtered[sample_key_column] == condition
        factor_values = nmf_normalized[condition_mask, factor]
        mean_all.append(np.mean(factor_values))
        mean_above.append(np.mean(factor_values[factor_values > threshold]))

    x = np.arange(len(conditions))
    width = 0.35

    fig = plt.figure(figsize=(5, 5))
    plt.bar(x - width/2, mean_all, width, label="All Cells", alpha=0.7)
    plt.bar(x + width/2, mean_above, width, label=f"Cells > {threshold}", alpha=0.7)

    plt.xticks(x, conditions)
    plt.xlabel("Condition")
    plt.ylabel("Mean Expression")
    plt.title(f"Mean Factor {factor} Usage by Condition")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig




def plot_nmf_factor_2Dhistogram(adata, factor1, factor2, sample_key_column, 
                    filter_column=None, filter_value=None, 
                    use_log=False, pseudocount=1e-3,
                    cap_density_value=None):
    """
    Plot a 2D density histogram of two normalized NMF factors against each other for each condition.

    Parameters:
    adata: AnnData
        Annotated data matrix with `obsm['X_nmf']` containing NMF factors and `obs[sample_key_column]` specifying conditions.
    factor1: int
        Index of the first NMF factor to plot (0-based).
    factor2: int
        Index of the second NMF factor to plot (0-based).
    sample_key_column: str
        Column in `adata.obs` containing the sample conditions.
    filter_column: str, optional
        Column in `adata.obs` to apply a filter.
    filter_value: Any, optional
        Value to filter `filter_column` by. Only rows matching this value will be included.
    use_log: bool, optional
        If True, log-transform the normalized values (default: False).
    pseudocount: float, optional
        Value to add to normalized values before log transformation (default: 1e-3).
    """
    # Apply filter if specified
    if filter_column is not None and filter_value is not None:
        filter_mask = adata.obs[filter_column] == filter_value
        obs_filtered = adata.obs[filter_mask]
        obsm_filtered = adata.obsm['X_nmf'][filter_mask, :]
    else:
        obs_filtered = adata.obs
        obsm_filtered = adata.obsm['X_nmf']

    # Normalize X_nmf such that the sum of all programs equals 1 per cell
    nmf_normalized = obsm_filtered / obsm_filtered.sum(axis=1, keepdims=True)

    if use_log:
        nmf_normalized = np.log10(nmf_normalized + pseudocount)

    # Get sample key conditions
    conditions = obs_filtered[sample_key_column].unique()

    # Create a figure with subplots for each condition
    num_conditions = len(conditions)
    fig, axes = plt.subplots(1, num_conditions, figsize=(6 * num_conditions, 6), constrained_layout=True)
    if num_conditions == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, condition in zip(axes, conditions):
        # Select cells belonging to the current condition
        condition_mask = obs_filtered[sample_key_column] == condition

        # Extract normalized NMF values for the specified factors
        factor1_values = nmf_normalized[condition_mask, factor1]
        factor2_values = nmf_normalized[condition_mask, factor2]

        # Plot 2D histogram
        if cap_density_value is None:
            h = ax.hist2d(
                factor1_values,
                factor2_values,
                bins=50,
                density=True,
                cmap='Blues'
            )
        else:
            h = ax.hist2d(
                factor1_values,
                factor2_values,
                bins=50,
                density=True,
                cmap='Blues',
                vmax=cap_density_value
            )

        # Add colorbar
        cb = fig.colorbar(h[3], ax=ax, label="Density")

        # Add labels and title
        ax.set_xlabel(f"Normalized Usage of Factor {factor1}")
        ax.set_ylabel(f"Normalized Usage of Factor {factor2}")
        ax.set_title(f"Condition: {condition}")
        ax.grid(alpha=0.5, linestyle='--')

    return fig

def plot_obsCol_by_factor(adata, factor, sample_key_column, obs_column, 
                         use_threshold=True, percentile=0.01,
                         filter_column=None, filter_value=None, figsize=(8, 5)):
    """
    Plot boxplots showing the distribution of an observation column value across bins
    of normalized NMF factor usage. Binning can be done either by threshold or percentiles.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with `obsm['X_nmf']` containing NMF factors
    factor : int
        Index of the NMF factor to analyze (0-based)
    sample_key_column : str
        Column in `adata.obs` containing the sample conditions
    obs_column : str
        Name of column in adata.obs containing quantitative data to plot
    use_threshold : bool
        If True, split data into above/below percentile as a threshold. If False, split into percentile bins
    percentile : float
        If use_threshold=True: percentile to calculate threshold (0-1)
        If use_threshold=False: size of percentile bins (e.g., 0.25 for quartiles)
    filter_column : str, optional
        Column in `adata.obs` to apply a filter
    filter_value : Any, optional
        Value to filter `filter_column` by. Only rows matching this value will be included
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    import seaborn as sns
    import numpy as np
    
    # Validate percentile input
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")
    
    # Apply filter if specified
    if filter_column is not None and filter_value is not None:
        filter_mask = adata.obs[filter_column] == filter_value
        obs_filtered = adata.obs[filter_mask]
        obsm_filtered = adata.obsm['X_nmf'][filter_mask, :]
    else:
        obs_filtered = adata.obs
        obsm_filtered = adata.obsm['X_nmf']

    # Normalize X_nmf such that sum of all programs equals 1 per cell
    nmf_normalized = obsm_filtered / obsm_filtered.sum(axis=1, keepdims=True)
    conditions = obs_filtered[sample_key_column].unique()
    
    # Prepare data by condition
    data_by_condition = {}
    for condition in conditions:
        # Get data for this condition
        condition_mask = obs_filtered[sample_key_column] == condition
        factor_values = nmf_normalized[condition_mask, factor]
        obs_values = obs_filtered.loc[condition_mask, obs_column]
        
        if len(factor_values) == 0:  # Skip if no data for this condition
            continue
            
        if use_threshold:
            # Calculate threshold for this condition
            threshold = np.percentile(factor_values, percentile * 100)
            #print(threshold)
            #print((factor_values<threshold).sum())
            #print((factor_values==threshold).sum())
            #print((factor_values>threshold).sum())
            # Create strict split at threshold
            below_mask = factor_values <= threshold
            above_mask = ~below_mask  # Everything else goes in high bin
            #print()
            #print(below_mask.sum(), above_mask.sum())
            # Create DataFrame with the split data
            data_by_condition[condition] = pd.DataFrame({
                'Value': np.concatenate([
                    obs_values[below_mask].values,
                    obs_values[above_mask].values
                ]),
                'Bin': ['Low'] * sum(below_mask) + ['High'] * sum(above_mask)
            })
            
            # Check for missing bins and add placeholder rows
            existing_bins = set(data_by_condition[condition]['Bin'].unique())
            missing_bins = set(bin_labels) - existing_bins
            
            if missing_bins:
                # Create placeholder rows for missing bins
                placeholder_rows = pd.DataFrame({
                    'Value': [bin_edges[bin_labels.index(bin_label)] for bin_label in missing_bins],
                    'Bin': list(missing_bins)
                })
                
                # Append placeholder rows to the condition's DataFrame
                data_by_condition[condition] = pd.concat([
                    data_by_condition[condition],
                    placeholder_rows
                ], ignore_index=True)
            
        else:
            # Calculate percentile bins
            n_bins = int(1 / percentile)
            print(n_bins)
            pct_edges = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(factor_values, 
                                    np.linspace(0, 100, n_bins + 1))
            print(bin_edges)
            bin_indices = np.digitize(factor_values, bin_edges,right=True) - 1
            bin_indices = np.minimum(bin_indices, n_bins - 1)  # Ensure no overflow
            
            # Create bin labels
            bin_labels = [f'{pct_edges[i]:.0f}-{pct_edges[i+1]:.0f}' 
                         for i in range(len(pct_edges)-1)]
            
            # Create DataFrame with binned data
            data_by_condition[condition] = pd.DataFrame({
                'Value': obs_values.values,
                'Bin': [bin_labels[idx] for idx in bin_indices]
            })
            
            # Check for missing bins and add placeholder rows
            existing_bins = set(data_by_condition[condition]['Bin'].unique())
            missing_bins = set(bin_labels) - existing_bins
            
            if missing_bins:
                # Create placeholder rows for missing bins
                placeholder_rows = pd.DataFrame({
                    'Value': [bin_edges[bin_labels.index(bin_label)] for bin_label in missing_bins],
                    'Bin': list(missing_bins)
                })
                
                # Append placeholder rows to the condition's DataFrame
                data_by_condition[condition] = pd.concat([
                    data_by_condition[condition],
                    placeholder_rows
                ], ignore_index=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Combine all data for plotting
    plot_data = []
    for condition in data_by_condition:
        df = data_by_condition[condition].copy()
        df['Condition'] = condition
        plot_data.append(df)
    
    if not plot_data:  # Check if we have any data to plot
        raise ValueError("No data to plot after filtering")
        
    plot_df = pd.concat(plot_data, ignore_index=True)
    
    # Create boxplot
    sns.boxplot(data=plot_df, x='Condition', y='Value', hue='Bin', ax=ax)
    
    # Customize plot
    ax.set_xlabel("Condition")
    ax.set_ylabel(obs_column)
    if use_threshold:
        title = f"Distribution of {obs_column} by Factor {factor} Usage\n(threshold at {percentile:.1%} percentile)"
    else:
        title = f"Distribution of {obs_column} by Factor {factor} Usage\n({int(1/percentile)} percentile bins)"
    ax.set_title(title)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig
