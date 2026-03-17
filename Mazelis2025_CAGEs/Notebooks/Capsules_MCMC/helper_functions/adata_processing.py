"""
Public, trimmed-down version of adata_processing helpers for Capsules_MCMC.

Contains only the functions required by:
- script1_Capsule_processing.py

If additional functions are needed later, they can be copied in explicitly
from the internal helper code.
"""

import glob
from typing import Dict, Tuple, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD

import psutil


# ---------------------------------------------------------------------------
# DATA LOADING AND SAMPLE METADATA
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> Tuple[str, str, int]:
    """Parse celltype, condition and replicate from filename."""
    base = filename.split("_t=")[0]
    parts = base.split("_")
    celltype = parts[0]

    if parts[1].endswith(("1", "2", "3")):
        condition = parts[1][:-1]
        replicate = int(parts[1][-1])
    else:
        condition = parts[1]
        replicate = 1

    return celltype, condition, replicate


def parse_timepoint(timepoint_str: str) -> int:
    """Extract numerical day value from timepoint string '...t=4' -> 4."""
    return int(timepoint_str[-1])


def get_umi_stats(adata: ad.AnnData) -> Tuple[float, float, float, float]:
    """Calculate UMI statistics from matrix."""
    umi_counts = adata.X.sum(axis=1)
    return (
        float(umi_counts.mean()),
        float(np.median(umi_counts, axis=0)[0, 0]),
        float(umi_counts.min()),
        float(umi_counts.max()),
    )


def load_h5ad_files(
    directory: str = ".", cell_type_prefixes: Optional[List[str]] = None
) -> Dict[str, ad.AnnData]:
    """
    Load h5ad files from directory, optionally filtering by cell type prefix.

    Parameters
    ----------
    directory
        Directory containing .h5ad files.
    cell_type_prefixes
        Single prefix or list of prefixes; if None, load all files.
    """
    h5ad_files = glob.glob(f"{directory}/*.h5ad")
    adata_dict: Dict[str, ad.AnnData] = {}

    if isinstance(cell_type_prefixes, str):
        cell_type_prefixes = [cell_type_prefixes]

    for file in h5ad_files:
        key = file.split("/")[-1].replace(".h5ad", "")
        celltype, _, _ = parse_filename(key)

        if cell_type_prefixes and not any(
            celltype.startswith(prefix) for prefix in cell_type_prefixes
        ):
            continue

        adata = sc.read_h5ad(file)
        adata.obs_names_make_unique(join="_")
        celltype, condition, replicate = parse_filename(key)
        adata.obs["condition"] = condition
        adata.obs["replicate"] = replicate
        for tp in adata.obs["timepoint"].unique():
            tp_key = f"{key}_t={tp}"
            adata_dict[tp_key] = adata[adata.obs["timepoint"] == tp].copy()

    return adata_dict


def create_sample_info(adata_dict: Dict[str, ad.AnnData]) -> pd.DataFrame:
    """Create DataFrame with sample information."""
    records = []

    for key, adata in adata_dict.items():
        celltype, condition, replicate = parse_filename(key)
        timepoint = parse_timepoint(key.split("t=")[1])
        mean_umi, median_umi, min_umi, max_umi = get_umi_stats(adata)

        record = {
            "key": key,
            "celltype": key.split("_")[0],
            "condition": condition,
            "replicate": replicate,
            "timepoint": timepoint,
            "n_cells": adata.n_obs,
            "mean_umi": mean_umi,
            "median_umi": median_umi,
            "min_umi": min_umi,
            "max_umi": max_umi,
        }
        records.append(record)

    sample_info = pd.DataFrame(records)
    sample_info.set_index("key", inplace=True)
    return sample_info


def read_batch_data(file_path: str) -> pd.DataFrame:
    """
    Read an Excel file containing experiment batch data.

    Must contain at least:
    - 'Key'
    - 'AnnData file'
    - 'Library prep'
    - 'Seeding date'
    - 'Seeding flask (source cells)'
    """
    expected_columns = [
        "Key",
        "AnnData file",
        "Library prep",
        "Seeding date",
        "Seeding flask (source cells)",
    ]

    df = pd.read_excel(file_path, index_col="Key")

    missing_cols = [
        col for col in expected_columns if col not in df.columns and col != "Key"
    ]
    if missing_cols:
        raise ValueError(f"Missing expected columns in batch file: {missing_cols}")

    return df


def integrate_batch_data(sample_info: pd.DataFrame, batch_info: pd.DataFrame) -> pd.DataFrame:
    """Join batch metadata onto sample_info and sort in a consistent way."""
    sample_info = pd.concat([sample_info, batch_info], axis=1)
    sample_info = sample_info.sort_values(["Library prep", "celltype", "condition"])
    return sample_info


# ---------------------------------------------------------------------------
# INTEGRATION / PREPROCESSING / NMF
# ---------------------------------------------------------------------------

def unify_adata(
    adata_dict: Dict[str, ad.AnnData],
    sample_info: pd.DataFrame,
    filter_genes: Optional[List[str]] = None,
) -> ad.AnnData:
    """
    Integrate all samples into a single AnnData with harmonized obs fields.
    """
    for k in adata_dict.keys():
        adata_dict[k].obs["sample_key"] = k
        adata_dict[k].obs["condition"] = sample_info.loc[k, "condition"]
        adata_dict[k].obs["replicate"] = sample_info.loc[k, "replicate"]
        adata_dict[k].obs["timepoint"] = sample_info.loc[k, "timepoint"]
        adata_dict[k].obs["Library"] = sample_info.loc[k, "Library prep"]
        adata_dict[k].obs["Seeding"] = sample_info.loc[k, "Seeding flask (source cells)"]

    adatas = [adata_dict[k] for k in adata_dict.keys()]
    adata_concat = sc.concat(adatas, join="outer")

    if filter_genes is not None:
        filter_to_gene_set(adata_concat, filter_genes)

    adata_concat.obs_names_make_unique(join="_")
    adata_concat.obs["Library"] = adata_concat.obs["Library"].astype("category")
    adata_concat.obs["Seeding"] = adata_concat.obs["Seeding"].astype("category")
    adata_concat.obs["timepoint"] = adata_concat.obs["timepoint"].astype("category")
    return adata_concat


def filter_to_gene_set(adata: ad.AnnData, gene_set) -> None:
    """
    Filter AnnData in-place to only keep genes present in the provided gene set.
    """
    gene_set = set(gene_set)
    genes_to_keep = [gene for gene in adata.var_names if gene in gene_set]
    if len(genes_to_keep) == 0:
        raise ValueError("No genes from the provided gene_set were found in the data")
    adata._inplace_subset_var(genes_to_keep)


def run_preprocess_and_harmony(
    adata: ad.AnnData,
    num_pcs: int = 10,
    use_harmony: bool = False,
    use_NMF: bool = False,
    NMF_kwargs: Optional[dict] = None,
    hvg_kwargs: Optional[dict] = None,
) -> None:
    """
    Preprocess AnnData, compute HVGs, and run PCA or NMF; optionally run Harmony.

    Populates:
    - layers: 'raw', 'norm', 'log'
    - var['highly_variable']
    - obsm['X_nmf'] and uns['nmf'] if use_NMF=True
    - neighbors graph on 'X_pca', 'X_nmf' or 'X_pca_harmony'
    """
    if NMF_kwargs is None:
        NMF_kwargs = {}

    adata.layers["raw"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.layers["norm"] = adata.X.copy()
    sc.pp.log1p(adata, base=10)
    adata.layers["log"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, inplace=True, **(hvg_kwargs or {}))
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    if not use_NMF:
        sc.pp.pca(adata, n_comps=num_pcs, layer="log", mask_var="highly_variable")
    else:
        print("Running NMF:")
        NMF_default_params = {"init": "nndsvd", "random_state": 0}
        NMF_params = {**NMF_default_params, **NMF_kwargs}
        X = adata[:, adata.var["highly_variable"]].X
        model = NMF(n_components=num_pcs, **NMF_params)
        adata.obsm["X_nmf"] = model.fit_transform(X)
        adata.uns["nmf"] = {
            "params": NMF_params,
            "model": model,
            "variance_ratio": model.reconstruction_err_,
            "n_components": num_pcs,
            "components": model.components_,
            "highly_variable_genes": adata.var_names[
                adata.var["highly_variable"]
            ].tolist(),
        }

    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    if use_harmony:
        if use_NMF:
            X_key = "X_nmf"
        else:
            X_key = "X_pca"

        adata.obs["batchkey"] = (
            adata.obs["Seeding"].astype(str) + "_" + adata.obs["Library"].astype(str)
        )
        sce.pp.harmony_integrate(
            adata,
            key=["batchkey"],
            basis=X_key,
            theta=4.0,
            lamb=10.0,
            sigma=0.05,
            max_iter_harmony=30,
        )
        sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15)
    else:
        if use_NMF:
            sc.pp.neighbors(adata, use_rep="X_nmf", n_neighbors=15)
        else:
            sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)


# ---------------------------------------------------------------------------
# NMF USAGE STATISTICS AND GENE-LEVEL STATS
# ---------------------------------------------------------------------------

def get_nmf_usage_stats(
    W, adata: ad.AnnData, condition: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute mean, std, CV, Fano, and median of NMF program usage across timepoints.

    Returns per-timepoint statistics (timepoint index, programs as columns).
    """
    if isinstance(W, np.ndarray):
        nmf_df = pd.DataFrame(
            W / W.sum(axis=1, keepdims=True),
            index=adata.obs.index,
            columns=[f"{i}" for i in range(W.shape[1])],
        )
    else:
        nmf_df = W.copy()
        nmf_df = nmf_df.div(nmf_df.sum(axis=1), axis=0)

    nmf_df["timepoint"] = adata.obs["timepoint"].astype(int)
    nmf_df["sample_key"] = adata.obs["sample_key"].astype(str)
    nmf_df["condition"] = adata.obs["condition"].astype(str)

    if condition is not None:
        if condition != "Ctrl":
            mask = nmf_df["condition"] == condition
        else:
            mask = nmf_df["condition"].str.contains("Ctrl")
        nmf_df = nmf_df[mask]

    nmf_df = nmf_df.drop("condition", axis=1)

    per_sample_mean = nmf_df.groupby(["sample_key", "timepoint"]).mean()
    per_sample_median = nmf_df.groupby(["sample_key", "timepoint"]).median()
    per_sample_var = nmf_df.groupby(["sample_key", "timepoint"]).var()

    per_timepoint_mean = per_sample_mean.groupby("timepoint").mean()
    per_timepoint_median = per_sample_median.groupby("timepoint").mean()
    per_timepoint_std = per_sample_var.groupby("timepoint").mean().apply(np.sqrt)
    per_timepoint_cv = per_timepoint_std / per_timepoint_mean
    per_timepoint_fano = per_timepoint_std**2 / per_timepoint_mean

    return (
        per_timepoint_mean,
        per_timepoint_std,
        per_timepoint_cv,
        per_timepoint_fano,
        per_timepoint_median,
    )


def get_gene_cv_mean_dataframes(adata: ad.AnnData) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Compute per-condition, per-timepoint gene-level mean/var/CV/normalized var/HVG flags.

    Returns
    -------
    dict[condition][timepoint] -> DataFrame with columns:
    ['mean', 'var', 'mean_sc', 'var_sc', 'cv', 'norm_var', 'hvg']
    """
    cond = adata.obs["condition"].unique()
    timepoints = adata.obs["timepoint"].unique()
    cv_mean_dict: Dict[str, Dict[int, pd.DataFrame]] = {}

    for c in cond:
        cv_mean_dict[c] = {}
        for t in timepoints:
            samples = adata.obs[
                (adata.obs["condition"] == c) & (adata.obs["timepoint"] == t)
            ]["sample_key"].unique()
            df_list = []
            n_cells_list = []
            for s in samples:
                subsample = adata[adata.obs["sample_key"] == s].copy()
                sc.pp.highly_variable_genes(
                    subsample, flavor="seurat_v3", n_top_genes=100, layer="raw"
                )
                norm_layer = subsample.layers["norm"].toarray()

                mean = np.mean(norm_layer, axis=0)
                var = np.var(norm_layer, axis=0)
                mean_sc = subsample.var["means"].values
                var_sc = subsample.var["variances"].values
                cv = np.sqrt(var) / (1e-10 + mean)
                norm_var = subsample.var["variances_norm"].values
                hvg = subsample.var["highly_variable"].values

                df = pd.DataFrame(
                    {
                        "mean": mean,
                        "var": var,
                        "mean_sc": mean_sc,
                        "var_sc": var_sc,
                        "cv": cv,
                        "norm_var": norm_var,
                        "hvg": hvg,
                    },
                    index=subsample.var_names,
                )

                df_list.append(df)
                n_cells_list.append(len(subsample))

            if not df_list:
                continue

            weights = np.array(n_cells_list) / np.sum(n_cells_list)
            df_avg = pd.concat(
                [df * w for df, w in zip(df_list, weights)]
            ).groupby(level=0).sum()
            cv_mean_dict[c][t] = df_avg

    return cv_mean_dict


# ---------------------------------------------------------------------------
# PCA VARIANCE, NMF FACTOR QUALITY, CV-MEAN REGRESSIONS
# ---------------------------------------------------------------------------

def compare_variance_explained(
    adata: ad.AnnData, n_components: int = 20, sample_frac: float = 0.1, n_permutations: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare explained variance of leading PCs on real vs column-permuted data.
    """
    X = adata[:, adata.var.highly_variable].X
    X_subset = X[
        np.random.choice(X.shape[0], int(X.shape[0] * sample_frac), replace=False)
    ].toarray()

    X_norm = X_subset - X_subset.mean(axis=0)
    X_norm = X_norm / np.sqrt((X_norm**2).sum())
    real_var = TruncatedSVD(n_components).fit(X_norm).explained_variance_ratio_

    perm_vars = []
    for _ in range(n_permutations):
        X_perm = X_norm.copy()
        row_idx = np.random.rand(*X_perm.shape).argsort(axis=0)
        col_idx = np.tile(np.arange(X_perm.shape[1]), (X_perm.shape[0], 1))
        X_perm = X_perm[row_idx, col_idx]
        perm_vars.append(TruncatedSVD(n_components).fit(X_perm).explained_variance_ratio_)

    return real_var, np.array(perm_vars)


def fit_cv_mean_relationship(
    mean_dict: Dict[str, pd.DataFrame],
    cv_dict: Dict[str, pd.DataFrame],
    exclude_keys: Optional[List[str]] = None,
    fixed_exponent: Optional[float] = None,
) -> dict:
    """
    Fit power-law relationship CV = a * mean^b in log-space.

    Returns a dict with keys: 'a', 'b', 'r2', 'log_params'.
    """
    from sklearn.linear_model import LinearRegression

    mean_data = []
    cv_data = []
    exclude_keys = [] if exclude_keys is None else exclude_keys

    for condition in mean_dict.keys():
        if condition in exclude_keys:
            continue
        for col in mean_dict[condition].columns:
            mean_data.extend(mean_dict[condition][col].values)
            cv_data.extend(cv_dict[condition][col].values)

    X = np.log(np.array(mean_data))
    y = np.log(np.array(cv_data))

    if fixed_exponent is not None:
        log_a = np.mean(y - fixed_exponent * X)
        a = np.exp(log_a)
        b = fixed_exponent

        y_pred = log_a + fixed_exponent * X
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        log_params = {"slope": fixed_exponent, "intercept": log_a}
    else:
        X_2d = X.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X_2d, y)
        b = reg.coef_[0]
        a = np.exp(reg.intercept_)
        r2 = reg.score(X_2d, y)
        log_params = {"slope": b, "intercept": reg.intercept_}

    return {"a": a, "b": b, "r2": r2, "log_params": log_params}


def create_regressed_randomized_cv(
    mean_dict: Dict[str, pd.DataFrame],
    rnd_mean_dict: Dict[str, pd.DataFrame],
    rnd_cv_dict: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Adjust randomized CVs to match changes in mean relative to real data:
    rndCV_reg = rndCV * sqrt(rnd_mean / mean).
    """
    rnd_reg_cv_dict: Dict[str, pd.DataFrame] = {}
    rnd_reg_mean_dict: Dict[str, pd.DataFrame] = {}

    for c in rnd_mean_dict.keys():
        rnd_reg_cv_dict[c] = rnd_cv_dict[c].copy()
        if c in mean_dict:
            rnd_reg_mean_dict[c] = mean_dict[c].loc[rnd_mean_dict[c].index].copy()
            for col in rnd_mean_dict[c].columns:
                mean = rnd_reg_mean_dict[c][col]
                rnd_mean = rnd_mean_dict[c][col]
                rnd_cv = rnd_cv_dict[c][col]
                rnd_reg_cv = rnd_cv * np.sqrt(rnd_mean / mean)
                rnd_reg_cv_dict[c][col] = rnd_reg_cv

    return rnd_reg_cv_dict, rnd_reg_mean_dict


# ---------------------------------------------------------------------------
# MOCK CLONE GENERATION (used for randomized controls)
# ---------------------------------------------------------------------------

def create_count_based_clones(
    ad_sc: ad.AnnData, ref_df: pd.DataFrame, n_cells_per_timepoint: Dict[int, int]
) -> ad.AnnData:
    """
    Create mock clones based on a reference dataframe specifying conditions and counts.

    Parameters
    ----------
    ad_sc
        Single-cell data with raw counts in .layers['raw'] and NMF model in .uns['nmf'].
    ref_df
        DataFrame with columns including 'timepoint', 'condition', 'n_counts'.
    n_cells_per_timepoint
        Mapping from timepoint -> number of cells to combine into each mock clone.
    """
    all_adata = []

    hvg_genes = ad_sc.uns["nmf"]["highly_variable_genes"]
    hvg_indices = np.where(ad_sc.var_names.isin(hvg_genes))[0]
    nmf_model = ad_sc.uns["nmf"]["model"]

    for timepoint in n_cells_per_timepoint.keys():
        timepoint_df = ref_df[ref_df["timepoint"] == timepoint]
        min_cells = n_cells_per_timepoint[timepoint]
        min_extra_cells = 2

        all_counts = []
        all_conditions = []
        all_other_cols = {
            col: [] for col in ref_df.columns if col not in ["timepoint", "condition", "n_counts"]
        }

        for condition in timepoint_df["condition"].unique():
            condition_mask = ad_sc.obs["condition"] == condition
            condition_indices = np.where(condition_mask)[0]
            if len(condition_indices) == 0:
                continue

            clone_specs = timepoint_df[timepoint_df["condition"] == condition]

            for _, row in clone_specs.iterrows():
                cell_counts = ad_sc.obs["n_counts"].iloc[condition_indices]
                n_cells = min_cells

                while True:
                    sampled_indices = np.random.choice(
                        len(condition_indices), size=n_cells, replace=True
                    )
                    total_counts = cell_counts.iloc[sampled_indices].sum()
                    if total_counts >= row["n_counts"]:
                        break
                    n_cells += min_extra_cells

                combined_counts = ad_sc.layers["raw"][
                    condition_indices[sampled_indices]
                ].sum(axis=0).A1

                frequencies = combined_counts / combined_counts.sum()
                sampled_counts = np.random.multinomial(
                    np.int64(row["n_counts"]), frequencies
                )

                all_counts.append(sampled_counts)
                all_conditions.append(condition)
                for col in all_other_cols:
                    all_other_cols[col].append(row[col])

        if len(all_counts) == 0:
            continue

        combined_counts = np.vstack(all_counts)
        cp10k = combined_counts * 10000 / combined_counts.sum(axis=1, keepdims=True)
        log_counts = np.log10(1 + cp10k)

        projected_W = nmf_model.transform(log_counts[:, hvg_indices])

        n_clones = combined_counts.shape[0]

        obs_dict = {
            "condition": all_conditions,
            "timepoint": [timepoint] * n_clones,
        }
        obs_dict.update(all_other_cols)
        obs_df = pd.DataFrame(
            obs_dict, index=[f"cell_{timepoint}_{i}" for i in range(n_clones)]
        )
        obs_df["sample_key"] = obs_df["condition"] + obs_df["timepoint"].astype(str)

        new_adata = ad.AnnData(
            X=sparse.csr_matrix(log_counts),
            obs=obs_df,
            var=ad_sc.var.copy(),
            dtype=np.float32,
        )
        new_adata.layers["raw"] = sparse.csr_matrix(combined_counts)
        new_adata.layers["norm"] = sparse.csr_matrix(cp10k)
        new_adata.obsm["X_nmf"] = projected_W
        all_adata.append(new_adata)

    combined_adata = ad.concat(all_adata, join="outer", merge="same")
    return combined_adata


# ---------------------------------------------------------------------------
# DIFFERENTIAL EXPRESSION (DE) FOR DRUG VS CONTROL
# ---------------------------------------------------------------------------

def vectorized_ranksums(x_groups: np.ndarray, y_groups: np.ndarray):
    """
    Vectorized Wilcoxon rank-sum (Mann–Whitney) tests across many genes.

    x_groups, y_groups have shape (n_tests, n_samples_x/y).
    """
    from scipy.stats import rankdata, norm

    n_x = x_groups.shape[1]
    combined = np.hstack((x_groups, y_groups))

    ranks = rankdata(combined, axis=1)
    x_ranks = ranks[:, :n_x]

    n1 = x_groups.shape[1]
    n2 = y_groups.shape[1]
    r1 = np.sum(x_ranks, axis=1)

    u1 = r1 - (n1 * (n1 + 1)) / 2
    u2 = n1 * n2 - u1
    statistic = np.where(u1 < u2, u1, u2)

    mean_rank = n1 * n2 / 2
    sd_rank = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (statistic - mean_rank) / sd_rank
    pvalue = 2 * norm.sf(np.abs(z))
    return statistic, pvalue


def differential_expression_analysis(
    adata: ad.AnnData,
    control_dict: Dict[str, str],
    timepoints: List[int] = [4, 6],
    drugs: List[str] = ["Vor", "Dec", "Aza"],
) -> Dict[str, pd.DataFrame]:
    """
    Differential expression between drug treatments and matched controls.

    Uses:
    - log10(1+CP10K) values in adata.X
    - HVGs in adata.var['highly_variable']
    - vectorized rank-sum tests per gene, per timepoint
    - Fisher's method to combine p-values across timepoints
    - weighted mean logFC across timepoints (by cell counts)
    """
    from scipy.stats import combine_pvalues
    from statsmodels.stats.multitest import multipletests

    hvg = adata.var_names[adata.var["highly_variable"]]
    results_dict: Dict[str, pd.DataFrame] = {}

    for drug in drugs:
        timepoint_pvals = []
        drug_means = []
        ctrl_means = []
        drug_weights = []
        ctrl_weights = []
        total_drug_cells = 0
        total_ctrl_cells = 0

        control = control_dict[drug]

        for tp in timepoints:
            print(f"Analyzing {drug}, time point {tp} days:")
            drug_cells = (
                (adata.obs["condition"] == drug)
                | (adata.obs.get("condition_orig", adata.obs["condition"]) == drug)
            ) & (adata.obs["timepoint"] == tp)
            ctrl_cells = (
                (adata.obs["condition"] == control)
                | (adata.obs.get("condition_orig", adata.obs["condition"]) == control)
            ) & (adata.obs["timepoint"] == tp)

            if sum(drug_cells) == 0 or sum(ctrl_cells) == 0:
                continue

            n_drug = int(sum(drug_cells))
            n_ctrl = int(sum(ctrl_cells))
            total_drug_cells += n_drug
            total_ctrl_cells += n_ctrl

            print("Calculating mean expression...")
            drug_mean = adata[drug_cells, hvg].X.toarray().mean(axis=0)
            ctrl_mean = adata[ctrl_cells, hvg].X.toarray().mean(axis=0)

            drug_means.append(drug_mean)
            ctrl_means.append(ctrl_mean)
            drug_weights.append(n_drug)
            ctrl_weights.append(n_ctrl)

            print("Calculating rank-sum p-values...")
            drug_expr = adata[drug_cells, hvg].X.toarray()
            ctrl_expr = adata[ctrl_cells, hvg].X.toarray()

            _, pvals = vectorized_ranksums(drug_expr.T, ctrl_expr.T)
            timepoint_pvals.append(pvals)

        if not timepoint_pvals:
            continue

        print("Calculating weighted means and log fold change...")
        drug_weights = np.array(drug_weights) / total_drug_cells
        ctrl_weights = np.array(ctrl_weights) / total_ctrl_cells

        weighted_drug_mean = sum(m * w for m, w in zip(drug_means, drug_weights))
        weighted_ctrl_mean = sum(m * w for m, w in zip(ctrl_means, ctrl_weights))
        logFC = weighted_drug_mean - weighted_ctrl_mean

        print("Combining p-values using Fisher's method...")
        timepoint_pvals = np.array(timepoint_pvals).T
        combined_pvals = np.array(
            [combine_pvalues(gene_pvals, method="fisher")[1] for gene_pvals in timepoint_pvals]
        )

        _, fdr, _, _ = multipletests(combined_pvals, method="fdr_bh")

        results_dict[drug] = pd.DataFrame(
            {"mean_logFC": logFC, "combined_fdr": fdr}, index=hvg
        )

    return results_dict

