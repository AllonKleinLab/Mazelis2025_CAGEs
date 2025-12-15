# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fitting a model of stochastic state transitions to NMF usage dynamics in clones

# %% [markdown]
# **This script carries out the following tasks:**
#
# 1. Runs stochastic simulations with different transition rate parameters, and save the simlation time series results to file. The model is a 2-state "on/off" model per cell, with clonal growth.
#
# 2. Train a neural network to predict stochastic simulation results, and in doing so generate a model that predicts the distribution of cells in on/off states over time for any parameter (interpolating). This uses pytorch.
#
# 3. Run MCMC using pyMC to fit parameters to the data for every program across clones

# %%

# Using Allon's conda environment pymc_arm, python 3.10
# PyMC on my mac requires an arm64 environment, so I'm using the pymc_arm environment
# How I built this:
# # 1. Create a new conda environment for `arm64`
# CONDA_SUBDIR=osx-arm64 conda create -n pymc_arm python=3.10
# conda activate pymc_arm
# # 2. Install PyMC with `arm64` support
# CONDA_SUBDIR=osx-arm64 conda install -c conda-forge pymc
# # 3. Verify architecture (should return 'arm64')
# python -c "import platform; print(platform.platform())"


# Configure PyTensor first, before any other imports
import os
import numpy as np
import pandas as pd
import pytensor
pytensor.config.floatX = "float64"
import arviz as az


import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from typing import List, Tuple, Dict, Any  # Make sure typing import is complete
import matplotlib.pyplot as plt
import seaborn as sns

# Add helper_functions to path
from pathlib import Path
import sys
import importlib
helper_path = str(Path().absolute() / 'helper_functions')
if helper_path not in sys.path:
    sys.path.append(helper_path)

# Import my functions specific to this project:
import state_simulations_1D_v1 as ss1D
from helper_functions import MCMC_1D_training_data_tasks as mcmc1D_training_data
from helper_functions import MCMC_1D_surrogate_model_tasks as mcmc1D_surrogate
#import MCMC_1D_pyMC_tasks as mcmc1D_pyMC
from helper_functions import MCMC_1D_pyMC_cont_plots as mcmc_plots
from helper_functions import plotting as hp


cache_file_path = './MCMC_data/1D_model_param_grid_cache.pkl'
surrogate_model_save_path = './MCMC_data/saved_1D_surrogate_models_d2_4_6'
#results_file = './MCMC_data/mcmc_resultsOLD.csv' # not used
cell_type = 'k562'
cont_results_file = f'./MCMC_data/mcmc_results_{cell_type}.csv'





# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Generate training data for the MCMC 1D model, using simulations

# %%
# Adding this to allow constant reloads after making changes:
mcmc1D_training_data = importlib.reload(mcmc1D_training_data)



def main():
    divisions = [0, 2, 4, 6]
    n_sims = 4000 # Eventual number of simulations should be higher than this
    q_min = 1e-4
    q_max = 0.95
    n_grid = 40

    init0_dfs, init1_dfs = mcmc1D_training_data.get_parameter_grid_simulations(
        divisions, n_sims, q_min, q_max, n_grid, cache_file=cache_file_path)




    # %% [markdown]
    # ### Visualize the mean and std of the model over the parameters from the simulations

    # %%
    mcmc1D_training_data = importlib.reload(mcmc1D_training_data)
    t=4
    p0 = 0.5
    mcmc1D_training_data.mean_and_std_over_params((1-p0)*init1_dfs[t]+p0*init0_dfs[t], show_heatmaps=True)

    # %% [markdown]
    # ## Train a smooth model on the probability distribution from the simulations

    # %% [markdown]
    # ### Train and test the model on a single division number
    #

    # %%
    from helper_functions import MCMC_1D_surrogate_model_tasks as mcmc1D_surrogate
    mcmc1D_surrogate = importlib.reload(mcmc1D_surrogate)

    # Load data from cache
    with open(cache_file_path, 'rb') as f:
        cache = pickle.load(f)
        divisions = cache['divisions']
        init0_dfs = cache['init0_dfs']
        init1_dfs = cache['init1_dfs']  


    # Create and train one model: one division number, one initial condition
    div_number = 4  # example
    model = mcmc1D_surrogate.SurrogateModel(
        hidden_layers=[64, 32],
        learning_rate=1e-3,
        batch_size=32,
        n_epochs=100,
        n_folds=5,
        patience=10
    )
    # Train the model
    history = model.train(init0_dfs[div_number], init1_dfs[div_number])

    # Examine the training history
    model.plot_history()


    # %% [markdown]
    # #### Spot-check the model output by comparing to the simulations:
    #
    # Test 1: Plot a random prediction with its neighbors
    # Test 2: Plot an interpolation of the model predictions over the parameter space between two training points
    #

    # %%
    mcmc1D_surrogate = importlib.reload(mcmc1D_surrogate)
    model.plot_random_prediction_with_neighbors(init0_dfs[div_number])
    plt.show()

    model.plot_model_predictions(init0_dfs[div_number],figsize= (10, 6))
    plt.show()


    # %% [markdown]
    # #### Make predictions over a grid of parameters that are offset from the training grid, and generate meand and std heatmaps. 
    #
    # Compare to plots above

    # %%
    # Make predictions
    new_parameters = np.array([[q0_1, q1_0] for q0_1 in np.logspace(-3.8, 0.4, 30) 
                                           for q1_0 in np.logspace(-3.8, 0.4, 30)])
    #predictions = model.predict(new_parameters)
    df_pred = model.predict_df(parameters=new_parameters, q0=0.0)
    mcmc1D_training_data.mean_and_std_over_params(df_pred, show_heatmaps=True)
    df_pred = model.predict_df(parameters=new_parameters, q0=1.0)
    mcmc1D_training_data.mean_and_std_over_params(df_pred, show_heatmaps=True)
    df_pred = model.predict_df(parameters=new_parameters, q0=0.5)
    mcmc1D_training_data.mean_and_std_over_params(df_pred, show_heatmaps=True)


    # %% [markdown]
    # ### Generate trained models for all divisions, and save them for use in the PyMC code

    # %%
    from helper_functions import MCMC_1D_surrogate_model_tasks as mcmc1D_surrogate
    mcmc1D_surrogate = importlib.reload(mcmc1D_surrogate)


    # Load data from cache
    with open(cache_file_path, 'rb') as f:
        cache = pickle.load(f)
        divisions = cache['divisions']
        init0_dfs = cache['init0_dfs']
        init1_dfs = cache['init1_dfs']  


    # Create and train all models
    model_dict = {}
    for d in [2,4,6]:

        model_dict[d] = mcmc1D_surrogate.SurrogateModel(
            hidden_layers=[64, 32],
            learning_rate=1e-3,
            batch_size=32,
            n_epochs=100,
            n_folds=5,
            patience=10
        )
        # Train the model
        history = model_dict[d].train(
            init0_dfs[d], 
            init1_dfs[d],
            verbose=False
        )

        # Examine the training history
        model_dict[d].plot_history()

    # Save the models:
    mcmc1D_surrogate.save_model_dict(model_dict, surrogate_model_save_path)


    # %%

    # Load all models later
    model_dict = mcmc1D_surrogate.load_model_dict(surrogate_model_save_path)


    # %% [markdown]
    # ## PyMC fitting with complete NMF score data

    # %% [markdown]
    # ### Test the continuous model with mock data

    # %%
    import MCMC_1D_pyMC_cont_tasks as mcmc1Dcont_pyMC
    mcmc1Dcont_pyMC = importlib.reload(mcmc1Dcont_pyMC)

    surrogate_model_path = './MCMC_data/saved_1D_surrogate_models_d2_4_6'

    # Initialize the inference object with surrogate models
    mcmc = mcmc1Dcont_pyMC.MCMCInference(surrogate_model_path=surrogate_model_path)

    # Generate mock data with defualt parameters
    true_params = {'r01': 0.1,
                    'r10': 0.1,
                    'p0': 0.5,
                    'threshold': 1e-3,
                    'a': -1.5,
                    'sigma': 0.5}
    mock_data = mcmc.create_mock_data(
        r01=true_params['r01'],
        r10=true_params['r10'],
        p0=true_params['p0'],
        threshold=true_params['threshold'],
        a=true_params['a'],
        sigma=true_params['sigma'],
        n_colonies=1000 # per timepoint
    )

    # Load the mock data
    mcmc.load_data(mock_data)

    # Setup and run MCMC
    mcmc.setup_model()
    mcmc.run_inference(
        draws=800,
        tune=1000,
        chains=6,
        cores=6
    )

    # Plot diagnostics
    plt.figure(figsize=(12, 12))
    mcmc.plot_diagnostics()
    plt.show()

    # Plot predictions vs data
    plt.figure(figsize=(12, 8))
    mcmc.plot_predictions()
    plt.show()


    print("\nTrue vs Inferred Parameters:")
    print("-" * 50)
    summary = az.summary(mcmc.trace)
    for param in ['p0', 'S', 'r', 'threshold', 'a', 'sigma']:
        if param == 'S':
            true_val = true_params['r01'] + true_params['r10']
        elif param == 'r':
            true_val = true_params['r10'] / true_params['r01']
        else:
            true_val = true_params[param]
    
        est_val = float(summary.loc[param, 'mean'])
        print(f"{param}:")
        print(f"  True: {true_val:.3e}")
        print(f"  Est:  {est_val:.3e}")

    # %%

    # mcmc1Dcont_pyMC = importlib.reload(mcmc1Dcont_pyMC)
    # mcmc.plot_predictions = mcmc1Dcont_pyMC.MCMCInference.plot_predictions.__get__(mcmc)
    # mcmc.plot_predictions()
    # plt.show()


    # %% [markdown]
    # ### Load some data and fit the model

    # %%
    mcmc_dict = {}

    # Initialize or load results_df:
    if os.path.exists(cont_results_file):
        cont_results_df = pd.read_csv(cont_results_file)
        ## Remove existing row with same program_id and condition if it exists
        #cont_results_df = cont_results_df.loc[~((cont_results_df['program_id'] == f1[0]) & 
        #                             (cont_results_df['condition'] == cond))]
    else:
        cont_results_df = pd.DataFrame()

    # %%
    import MCMC_1D_pyMC_cont_tasks as mcmc1Dcont_pyMC
    mcmc1Dcont_pyMC = importlib.reload(mcmc1Dcont_pyMC)
    surrogate_model_path = './MCMC_data/saved_1D_surrogate_models_d2_4_6'

    show_plots = False

    # Get some data

    conditions = ['Ctrl','Aza','Dec','Vor']
    programs = range(15)
    data_dir = f'./MCMC_data/'
    nmf_usage_file = f'./MCMC_data/data_NMF_usage_{cell_type}.csv'
    nmf_data = pd.read_csv(nmf_usage_file)
    print(f'Loaded NMF data for {cell_type}...')
    cont_mcmc_dict = {}

    for program_id in programs:
        for cond in conditions:
        
            k=f'{cond}-NMF{program_id}'
            prog_col = f'NMF_{program_id}'

            print(f'Processing {k}...')

            # Filter for condition and get relevant program column
            data = nmf_data[nmf_data['condition_orig'] == cond][['timepoint', prog_col]].copy()
            # Rename program column to 'score'
        
            data = data.rename(columns={prog_col: 'score'})
            print(f'Isolated NMF data for {cell_type}, {cond}, {prog_col}...')
        
            # Initialize the inference object with surrogate models
            cont_mcmc_dict[k] = mcmc1Dcont_pyMC.MCMCInference(surrogate_model_path=surrogate_model_path)

            # Load the mock data
            cont_mcmc_dict[k].load_data(data)

            # Setup and run MCMC
            cont_mcmc_dict[k].setup_model()
            cont_mcmc_dict[k].run_inference(
                draws=2000,
                tune=2000,
                chains=8,
                cores=8,
                show_progressbar=False
            )

            # Plot diagnostics
            cont_mcmc_dict[k].plot_diagnostics()
            hp.save_editable_pdf(plt.gcf(), f'./MCMC_data/plots/MCMC_1Dcont_diagnostics_{cell_type}_{cond}_factor{program_id}.pdf')
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot predictions vs data
            cont_mcmc_dict[k].plot_predictions()
            hp.save_editable_pdf(plt.gcf(), f'./MCMC_data/plots/MCMC_1Dcont_predictions_{cell_type}_{cond}_factor{program_id}.pdf')
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Save parameters to file
            row_data = mcmc1Dcont_pyMC.create_mcmc_summary_row(cont_mcmc_dict[k], program_id, cond)

            # Add new row
            cont_results_df = pd.concat([cont_results_df, pd.DataFrame([row_data])], ignore_index=True)
            # Save results
            cont_results_df.to_csv(cont_results_file, index=False)


    cont_results_df


    # %%

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_fraction_nonzero(cont_mcmc_dict, program_id=2, figsize=(3, 3))
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_fraction_nonzero_{cell_type}_factor3_ALL.pdf')


    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_mean_usage(cont_mcmc_dict, program_id=2, figsize=(3, 3))
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_mean_usage_{cell_type}_factor3_ALL.pdf')


    # %%
    cont_results_df.columns

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    f = 2
    cont_results_df = pd.read_csv(cont_results_file)
    fig = mcmc_plots.plot_transition_rates(cont_results_df, f, figsize=(3, 3))
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_transition_rates_{cell_type}_factor{f}_ALL.pdf')

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    f = 2
    for cond in conditions:
        fig=mcmc_plots.plot_histograms(cont_mcmc_dict, cond, f, figsize=(1.0, 3))
        hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_histograms_{cell_type}_{cond}factor{f}_ALL.pdf')


    # %%
    cont_results_df

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    fig = mcmc_plots.plot_relative_transition_rates(cont_results_df, param='S',figsize=(3,3.5),xscale='log',xlim=(10**(-1),10**1))
    #fig.axes[0].set_xlim(0.1,10)  # Set y-axis limits from 0 to 1.2
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_transition_rates_{cell_type}_S_ALL.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='S',figsize=(3,3.5),clim=None)
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_transition_rates_{cell_type}_S_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates(cont_results_df, param='r',figsize=(3,3.5), xscale='log',xlim=(0.1,10))
    #fig.axes[0].set_xlim(0.1,10)  # Set y-axis limits from 0 to 1.2

    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_transition_rates_{cell_type}_r_ALL.pdf')
    plt.show()

    # Example usage:
    # fig = plot_relative_transition_rates(cont_results_df, param='S')
    # fig = plot_relative_transition_rates(cont_results_df, param='r01')
    # fig = plot_relative_transition_rates(cont_results_df, param='r10')
    # fig = plot_relative_transition_rates(cont_results_df, param='r')

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='r',figsize=(3,3.5),clim=None)
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_rate_ratio_{cell_type}_r_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    fig = mcmc_plots.plot_relative_transition_rates(cont_results_df, param='r01',figsize=(3,3.5), xscale='log',xlim=(0.01,100))
    #fig.axes[0].set_xlim(0.8, 200)  # or whatever limits you want
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_transition_rates_{cell_type}_r01_ALL.pdf')
    plt.show()

    # Example usage:
    # fig = plot_relative_transition_rates(cont_results_df, param='S')
    # fig = plot_relative_transition_rates(cont_results_df, param='r01')
    # fig = plot_relative_transition_rates(cont_results_df, param='r10')
    # fig = plot_relative_transition_rates(cont_results_df, param='r')

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='r01',figsize=(3,3.5),clim=None)
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_on_rate_{cell_type}_r01_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates(cont_results_df, param='r10',figsize=(3,3.5), xscale='log',xlim=(0.01,100))
    #fig.axes[0].set_xlim(0.8, 10)  # or whatever limits you want
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_transition_rates_{cell_type}_r10_ALL.pdf')
    plt.show()

    # Example usage:
    # fig = plot_relative_transition_rates(cont_results_df, param='S')
    # fig = plot_relative_transition_rates(cont_results_df, param='r01')
    # fig = plot_relative_transition_rates(cont_results_df, param='r10')
    # fig = plot_relative_transition_rates(cont_results_df, param='r')

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='r10',figsize=(3,3.5),clim=[-2,2])
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_off_rate_{cell_type}_r10_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    fig = mcmc_plots.plot_relative_transition_rates(cont_results_df, param='1/r01',figsize=(3,3.5), xscale='log',xlim=(0.01,100))
    #fig.axes[0].set_xlim(0.8, 200)  # or whatever limits you want
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_persistence_time_{cell_type}_t0_ALL.pdf')
    plt.show()



    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='1/r10',figsize=(3,3.5),clim=[-2,2])
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_on_time_{cell_type}_t1_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    fig = mcmc_plots.plot_relative_transition_rates(cont_results_df, param='1/r10',figsize=(3,3.5), xscale='log',xlim=(0.01,100))
    #fig.axes[0].set_xlim(0.8, 200)  # or whatever limits you want
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_persistence_time_{cell_type}_t1_ALL.pdf')
    plt.show()



    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='1/r01',figsize=(3,3.5),clim=None)
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_off_time_{cell_type}_t0_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='min_t',figsize=(3,3.5),clim=None)
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_max_persistence_time_{cell_type}_min_t_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='1/S',figsize=(3,3.5),clim=[-3,3])
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_persistence_time_{cell_type}_1_S_heatmap.pdf')
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    fig = mcmc_plots.plot_relative_transition_rates_heatmap(cont_results_df, param='t1_frac',figsize=(3,3.5),clim=[-1,1])
    hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1Dcont_relative_bias_{cell_type}_t1_frac_heatmap.pdf')
    plt.show()

    # %% [markdown]
    # # Final plots

    # %%
    components_file_name = f'./Table S2 sheets/NMF_factors_raw_{cell_type}.csv'
    deg_results_file_name = f'./Table S2 sheets/{cell_type}_deg_results_dict.pkl'

    cont_results_df = pd.read_csv(cont_results_file)
    H = pd.read_csv(components_file_name)
    # Set 'Unnamed: 0' column as index and rename it to 'Gene name'
    H = H.set_index('Unnamed: 0')
    H.index.name = 'Gene name'
    print(H)

    deg_results_dict = pickle.load(open(deg_results_file_name, 'rb'))




    # %%
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Plot for each drug
    for i, (drug, df) in enumerate(deg_results_dict.items()):
        # Check for NaN values
        if df['combined_fdr'].isna().any():
            print(f"Warning: {drug} has {df['combined_fdr'].isna().sum()} NaN FDR values")
    
        # Convert FDR to -log10, saturating at 30
        log_fdr = -np.log10(df['combined_fdr'].clip(lower=1e-30))
    
        # Create volcano plot
        axes[i].scatter(df['mean_logFC'], log_fdr, alpha=0.5, s=5)
        axes[i].set_xlabel('Log Fold Change')
        axes[i].set_ylabel('-log10(FDR)')
        axes[i].set_title(drug)
    
        # Add horizontal line at FDR=0.05
        axes[i].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    
        # Add vertical lines at log2FC = ±1
        axes[i].axvline(-np.log10(2), color='red', linestyle='--', alpha=0.5)
        axes[i].axvline(np.log10(2), color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)
    gene_persistence_times_dict = mcmc_plots.calculate_gene_persistence_times(cont_results_df, H, conditions=['Ctrl', 'Aza', 'Dec', 'Vor'], debug=True )
    gene_persistence_times_dict['1/S']


    # fig = mcmc_plots.plot_persistence_vs_expression(deg_results_dict,
    #                                                 gene_persistence_times_df,
    #                                                 H=H, top_n_genes=100,
    #                                                 annotate_genes=['VIM_hg','KRT8_hg',
    #                                                                 'HBZ_hg','ALAS2_hg',
    #                                                                 'SLC25A37_hg',
    #                                                                 'FOS_hg',
    #                                                                 'TOP2A_hg'])
    # hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1D_del_on_time_vs_log2FC_{cell_type}.pdf')
    # plt.show()






    # %%
    #plt.scatter(deg_results_dict['Aza']['mean_logFC'], -np.log10(deg_results_dict['Aza']['combined_fdr']))
    np.sum(np.abs(deg_results_dict['Aza']['mean_logFC'])>=0.1)

    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    cond = ['Aza','Dec','Vor']

    gene_list = {}
    gene_list['k562'] = ['VIM_hg','KRT8_hg',
                    'HBE1_hg','HBG2_hg',
                    'HBZ_hg','ALAS2_hg',
                    'SLC25A37_hg',
                    'MKI67_hg',
                    'TOP2A_hg',
                    'S100A11_hg',
                    'SCN9A_hg']
    gene_list['l1210'] = ['Sirt5_mm']


    for c in cond:
        fig=mcmc_plots.plot_persistence_vs_expression_density(deg_results_dict, gene_persistence_times_dict, 
                                                transition_type='1/S',
                                                drug=c, figsize=(5,4),
                                                H=H, top_n_genes=20, 
                                                annotate_genes=gene_list[cell_type], 
                                                bins=100, levels=10, smooth_sigma=30,
                                                annotate_outliers=False)
        hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1D_persistence_vs_expression_density_{c}{cell_type}.pdf')
        plt.show()


    # %%
    mcmc_plots = importlib.reload(mcmc_plots)

    cond = ['Aza','Dec','Vor']

    epi_gene_list = {}
    epi_gene_list['k562'] = [
    'BPTF_hg',
    'BRD4_hg',
    'CREBBP_hg',
    'FTO_hg',
    'JMJD1C_hg',
    'KDM4A_hg',
    'KDM6B_hg',
    'PRMT8_hg',
    'PSIP1_hg',
    'RAG2_hg',
    'SETD5_hg',
    'SMYD3_hg',
    'TULP4_hg'
    ]


    for c in cond:
        fig=mcmc_plots.plot_persistence_vs_expression_density(deg_results_dict, gene_persistence_times_dict, 
                                                transition_type='1/S',
                                                drug=c, figsize=(5,4),
                                                H=H, top_n_genes=20, 
                                                annotate_genes=epi_gene_list[cell_type], 
                                                bins=100, levels=10, smooth_sigma=30,
                                                annotate_outliers=False)
        #hp.save_editable_pdf(fig, f'./MCMC_data/plots/MCMC_1D_persistence_vs_expression_density_{c}{cell_type}.pdf')
        plt.show()


    # %%


if __name__ == '__main__':
    main()
