## Dependencies for Capsules_MCMC scripts

This file summarizes the dependencies of the two main scripts so we can refer to it later when updating the GitHub repo and documentation.

---

### script1_Capsule_processing.py (Figure 4)

**Python packages (core):**
- `scanpy`, `scanpy.external`, `anndata`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `psutil`, `gc`, `os`, `sys`, `pathlib`, `importlib`, `pickle`
- `sklearn.decomposition` (NMF, TruncatedSVD)
- `statsmodels`, `scipy` (via helper functions for DE)

**Local helper modules:**
- From `helper_functions/adata_processing.py`:
  - `load_h5ad_files`, `create_sample_info`, `read_batch_data`, `integrate_batch_data`
  - `unify_adata`, `run_preprocess_and_harmony`
  - `compare_variance_explained`
  - `create_count_based_clones`
  - `get_gene_cv_mean_dataframes`
  - `get_nmf_usage_stats`, `fit_cv_mean_relationship`, `create_regressed_randomized_cv`
  - `differential_expression_analysis`
- From `helper_functions/plotting.py`:
  - `plot_gene_cv_vs_mean`, `plot_weighted_cv_vs_mean`, `plot_weighted_gene_correlations`
  - `plot_nmf_programs_heatmap`, `plot_nmf_programs_heatmap_horizontal`
  - `plot_NMF_mean_cv_relationship`
  - `plot_program_boxplot_time_series`
  - `create_nmf_dynamics_control_normalized_heatmap_grid`
  - `create_nmf_dynamics_heatmap_grid`
  - `save_umaps`, `save_figure_rasterized_data`, `save_editable_pdf`

**Input data / config:**
- `./Not_normalized/`:
  - `k562_Ctrl1.h5ad`, `k562_Ctrl2.h5ad`, `k562_Ctrl3.h5ad`
  - `k562_Aza.h5ad`, `k562_Dec.h5ad`, `k562_Vor.h5ad`
  - `l1210_Ctrl1.h5ad`, `l1210_Ctrl2.h5ad`, `l1210_Ctrl3.h5ad`
  - `l1210_Aza.h5ad`, `l1210_Dec.h5ad`, `l1210_Vor.h5ad`
- `./Not_normalized/Table_LibraryBatches.xlsx`
- `./gene_lists/`:
  - `{cell_type}_gene_list_to_plot.txt`
  - `{cell_type}_NMFs_to_plot.txt`
  - `{cell_type}_exclude_prefix_list.txt`
  - `{cell_type}_exclude_suffix_list.txt`
- Environment/requirements files:
  - `environment_script1.yml`
  - `requirements_script1_all.txt`

**Outputs:**
- Figures under `./figures/` (CV vs mean, gene–gene correlations, UMAPs, NMF programs, NMF CV/Fano).
- Workspace and NMF usage:
  - `./saved_workspaces/workspace_{cell_type}.pkl`
  - `./MCMC_data/data_NMF_usage_{cell_type}.csv`
- NMF factor tables (Table S2 inputs):
  - `NMF_factors_{cell_type}.xlsx`
  - `NMF_factors_raw_{cell_type}.xlsx`
  - `NMF_factors_raw_{cell_type}.csv`
- DEG results:
  - `./Table S2 sheets/{cell_type}_deg_results_dict.pkl`

---

### script2_MCMC.py (Figure 5)

**Python packages (core):**
- `numpy`, `pandas`
- `pytensor`, `pytensor.tensor`
- `pymc`
- `arviz`
- `matplotlib`, `pickle`, `os`, `sys`
- (plus `torch`, `sklearn`, etc., via surrogate/plot helpers; captured in `environment_script2_pymc_arm.yml`)

**Local helper modules:**
- `helper_functions/MCMC_1D_training_data_tasks.py`:
  - `get_parameter_grid_simulations`, `mean_and_std_over_params`
  - depends on `helper_functions/state_simulations_1D_v1.py`
- `helper_functions/MCMC_1D_surrogate_model_tasks.py`:
  - `SurrogateModel` (and its methods), `save_model_dict`, `load_model_dict`
- `MCMC_1D_pyMC_cont_tasks.py` (top-level module):
  - `MCMCInference` class with:
    - `load_data`, `load_surrogate_models`, `setup_model`, `run_inference`
    - `plot_diagnostics`, `plot_predictions`, `create_mock_data`
  - `create_mcmc_summary_row`
- `helper_functions/MCMC_1D_pyMC_cont_plots.py`:
  - `plot_fraction_nonzero`, `plot_mean_usage`, `plot_transition_rates`, `plot_histograms`
  - `plot_relative_transition_rates`, `plot_relative_transition_rates_heatmap`
  - `calculate_gene_persistence_times`, `plot_persistence_vs_expression_density`
- `helper_functions/plotting.py`:
  - `save_editable_pdf`

**Input data / config:**
- Simulation cache:
  - `./MCMC_data/1D_model_param_grid_cache.pkl`
- Surrogate models:
  - `./MCMC_data/saved_1D_surrogate_models_d2_4_6/`
- NMF usage (from script1):
  - `./MCMC_data/data_NMF_usage_{cell_type}.csv`
- MCMC results table:
  - `./MCMC_data/mcmc_results_{cell_type}.csv`
- NMF factors and DEG results (from script1):
  - `./Table S2 sheets/NMF_factors_raw_{cell_type}.csv`
  - `./Table S2 sheets/{cell_type}_deg_results_dict.pkl`
- Environment file:
  - `environment_script2_pymc_arm.yml`

**Outputs:**
- Surrogate model artifacts:
  - `./MCMC_data/saved_1D_surrogate_models_d2_4_6/` (per-division models)
- MCMC summary and plots:
  - `./MCMC_data/mcmc_results_{cell_type}.csv`
  - Multiple PDFs under `./MCMC_data/plots/`:
    - diagnostics, predictions, transition-rate barplots and heatmaps
    - persistence vs expression density plots for different drugs/parameters

