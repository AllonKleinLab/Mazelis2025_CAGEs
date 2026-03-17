"""
Public, trimmed-down plotting helpers for Capsules_MCMC.

Contains only the functions required by:
- script1_Capsule_processing.py
- script2_MCMC.py (via generic save_* utilities)
"""

from typing import Dict, List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import gridspec

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ---------------------------------------------------------------------------
# GENERIC FIGURE / UMAP SAVERS
# ---------------------------------------------------------------------------

def save_umaps(
    adata,
    color_list: List[str],
    prefix: str = "",
    save_folder: Optional[str] = None,
    umap_kwargs: Optional[dict] = None,
    rasterize: bool = True,
    dpi: int = 300,
    size: float = 10,
    figsize: Tuple[float, float] = (5, 5),
):
    """
    Save individual UMAP plots for each color variable with consistent formatting.
    """
    from pathlib import Path

    if umap_kwargs is None:
        umap_kwargs = {}

    if save_folder is not None:
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(".")

    default_params = {"frameon": True, "return_fig": True, "size": size}
    plot_params = {**default_params, **umap_kwargs}

    for color in color_list:
        plt.figure(figsize=figsize)
        fig = sc.pl.umap(adata, color=color, **plot_params)

        ax = fig.axes[0]
        ax.set_aspect("equal", adjustable="box")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        x_center = sum(xlim) / 2
        y_center = sum(ylim) / 2
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

        if rasterize:
            scatter = fig.axes[0].collections[0]
            scatter.set_rasterized(True)
            fig.set_dpi(dpi)

        plt.tight_layout()
        filename = f"{prefix}_umap_{color}.pdf" if prefix else f"umap_{color}.pdf"
        save_file = save_path / filename
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_figure_rasterized_data(fig, filename: str, dpi: int = 300):
    """
    Save figure with rasterized data elements while keeping text/axes as vectors.
    """
    old_pdf = plt.rcParams["pdf.fonttype"]
    old_ps = plt.rcParams["ps.fonttype"]

    try:
        for ax in fig.axes:
            for collection in ax.collections:
                collection.set_rasterized(True)
            for line in ax.lines:
                line.set_rasterized(True)
            for im in ax.images:
                im.set_rasterized(True)

        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    finally:
        plt.rcParams["pdf.fonttype"] = old_pdf
        plt.rcParams["ps.fonttype"] = old_ps


def save_editable_pdf(fig, filename: str, dpi: int = 300):
    """
    Save matplotlib figure as PDF with editable text.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename, metadata={"Creator": "Matplotlib"}) as pdf:
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        pdf.savefig(fig, dpi=dpi, transparent=True)


# ---------------------------------------------------------------------------
# GENE-LEVEL CV / MEAN AND CORRELATIONS
# ---------------------------------------------------------------------------

def plot_gene_cv_vs_mean(ad_all, gene_cv_mean_dict, rnd_gene_cv_mean_dict, condition: str):
    """
    Create per-timepoint CV vs mean plots for a given condition.
    """
    c = condition
    fig, ax = plt.subplots(1, 4, figsize=(13, 4))

    for i, t in enumerate(gene_cv_mean_dict[c].keys()):
        ax[i].scatter(
            gene_cv_mean_dict[c][t]["mean"],
            gene_cv_mean_dict[c][t]["cv"],
            c=gene_cv_mean_dict[c][t]["norm_var"],
            s=20,
            marker="o",
            alpha=0.3,
            cmap="coolwarm",
            label="Observed clones",
        )

        ax[i].set_xscale("log")
        ax[i].set_yscale("log")

        if t > 0:
            ax[i].loglog(
                rnd_gene_cv_mean_dict[c][t]["mean"],
                rnd_gene_cv_mean_dict[c][t]["cv"],
                marker=".",
                color="grey",
                alpha=0.2,
                markersize=1,
                label="Mock clones",
                linestyle="None",
            )

        ax[i].set_ylabel("CV")
        ax[i].set_xlabel("Mean")
        ax[i].set_title(f"{c} t={t}")
        ax[i].legend()
        plt.tight_layout()

    return fig


def plot_weighted_cv_vs_mean(
    gene_cv_mean_dict,
    rnd_gene_cv_mean_dict,
    ad_all,
    conditions: List[str] = ["Ctrl-2", "Ctrl-3"],
    timepoints: List[int] = [4, 6],
):
    """
    Single CV vs mean plot with cell-count weighted averaging across conditions/timepoints.
    """
    import pandas as pd

    cell_counts = (
        pd.DataFrame(
            {"condition": ad_all.obs["condition"], "timepoint": ad_all.obs["timepoint"]}
        )
        .groupby(["condition", "timepoint"])
        .size()
    )

    def compute_weighted_stats(data_dict, include_norm_vars=False):
        n_genes = len(ad_all.var_names)
        all_means = np.zeros((n_genes, len(conditions) * len(timepoints)))
        all_vars = np.zeros((n_genes, len(conditions) * len(timepoints)))
        if include_norm_vars:
            all_norm_vars = np.zeros((n_genes, len(conditions) * len(timepoints)))
        weights = np.zeros(len(conditions) * len(timepoints))

        idx = 0
        for cond in conditions:
            for tp in timepoints:
                if tp in data_dict[cond]:
                    try:
                        weight = cell_counts.loc[(cond, tp)]
                    except KeyError:
                        continue
                    all_means[:, idx] = data_dict[cond][tp]["mean"]
                    all_vars[:, idx] = data_dict[cond][tp]["var"]
                    if include_norm_vars:
                        all_norm_vars[:, idx] = data_dict[cond][tp]["norm_var"]
                    weights[idx] = weight
                    idx += 1

        valid_weights = weights[:idx]
        normalized_weights = valid_weights / valid_weights.sum()

        mean_means = np.average(all_means[:, :idx], axis=1, weights=normalized_weights)
        mean_vars = np.average(all_vars[:, :idx], axis=1, weights=normalized_weights)
        mean_cvs = np.sqrt(mean_vars) / mean_means

        if include_norm_vars:
            mean_norm_vars = np.average(
                all_norm_vars[:, :idx], axis=1, weights=normalized_weights
            )
            return mean_means, mean_cvs, mean_norm_vars
        return mean_means, mean_cvs

    real_means, real_cvs, real_norm_vars = compute_weighted_stats(
        gene_cv_mean_dict, include_norm_vars=True
    )
    rnd_means, rnd_cvs = compute_weighted_stats(rnd_gene_cv_mean_dict)

    fig, ax = plt.subplots(figsize=(3.8, 3))

    sort_idx = np.argsort(real_norm_vars)
    scatter = ax.scatter(
        real_means[sort_idx],
        real_cvs[sort_idx],
        c=real_norm_vars[sort_idx],
        s=20,
        marker="o",
        alpha=0.9,
        cmap="coolwarm",
        label="Observed",
    )

    ax.scatter(
        rnd_means,
        rnd_cvs,
        color="grey",
        s=10,
        marker="o",
        alpha=0.5,
        label="Randomized",
    )

    plt.colorbar(scatter, label="Normalized variance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("CV (from weighted mean/var)")
    ax.set_xlabel("Mean (cell-count weighted)")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_weighted_gene_correlations(
    gene_cv_mean_dict,
    ad_all,
    conditions: List[str] = ["Ctrl-2", "Ctrl-1", "Ctrl-3"],
    timepoints: List[int] = [4, 6],
    n_top_genes: int = 500,
    vmin: float = -0.2,
    vmax: float = 0.2,
    exclude_prefix_file: Optional[str] = None,
    exclude_suffix_file: Optional[str] = None,
    clustering_method: str = "average",
    min_corr_threshold: Optional[float] = None,
    use_absolute_corr: bool = False,
):
    """
    Weighted gene–gene correlation matrix for top variable genes.
    """
    from scipy import cluster, distance, sparse
    import re

    exclude_patterns = []
    if exclude_prefix_file:
        with open(exclude_prefix_file, "r") as f:
            prefixes = [line.strip() for line in f if line.strip()]
            exclude_patterns.extend([f"^{p}" for p in prefixes])
    if exclude_suffix_file:
        with open(exclude_suffix_file, "r") as f:
            suffixes = [line.strip() for line in f if line.strip()]
            exclude_patterns.extend([f"{s}$" for s in suffixes])

    if exclude_patterns:
        combined_pattern = "|".join(exclude_patterns)
        gene_mask = ~np.array(
            [bool(re.search(combined_pattern, gene)) for gene in ad_all.var_names]
        )
    else:
        gene_mask = np.ones(len(ad_all.var_names), dtype=bool)

    cell_counts = (
        pd.DataFrame(
            {"condition": ad_all.obs["condition"], "timepoint": ad_all.obs["timepoint"]}
        )
        .groupby(["condition", "timepoint"])
        .size()
    )

    weights = []
    for cond in conditions:
        for tp in timepoints:
            weights.append(cell_counts.loc[(cond, tp)])
    normalized_weights = np.array(weights) / np.sum(weights)

    all_norm_vars = []
    for cond in conditions:
        for tp in timepoints:
            if tp in gene_cv_mean_dict[cond]:
                all_norm_vars.append(gene_cv_mean_dict[cond][tp]["norm_var"])

    mean_norm_vars = np.average(np.array(all_norm_vars), weights=normalized_weights, axis=0)
    filtered_norm_vars = mean_norm_vars[gene_mask]
    filtered_gene_names = ad_all.var_names[gene_mask]

    top_var_indices = np.argsort(filtered_norm_vars)[-n_top_genes:]
    top_gene_names = filtered_gene_names[top_var_indices]

    ad_t = ad_all[ad_all.obs["timepoint"].isin(timepoints), top_gene_names].copy()

    n_genes = len(top_gene_names)
    weighted_covs = np.zeros((n_genes, n_genes))
    weighted_stds = np.zeros(n_genes)

    weight_idx = 0
    for cond in conditions:
        for tp in timepoints:
            mask = (ad_t.obs["condition"] == cond) & (ad_t.obs["timepoint"] == tp)
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

    corr_matrix = weighted_covs / np.outer(weighted_stds, weighted_stds)

    if use_absolute_corr:
        dist_matrix = 1 - np.abs(corr_matrix)
    else:
        dist_matrix = 1 - corr_matrix

    np.fill_diagonal(dist_matrix, 0)

    if min_corr_threshold is not None:
        clustering_matrix = dist_matrix.copy()
        mask = np.abs(corr_matrix) < min_corr_threshold
        clustering_matrix[mask] = 1
        np.fill_diagonal(clustering_matrix, 0)
    else:
        clustering_matrix = dist_matrix

    linkage = cluster.hierarchy.linkage(
        distance.squareform(clustering_matrix), method=clustering_method
    )
    idx = cluster.hierarchy.leaves_list(linkage)
    ordered_corr = corr_matrix[idx, :][:, idx]
    ordered_genes = top_gene_names[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(ordered_corr, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    timepoints_str = ",".join(map(str, timepoints))
    ax.set_title(
        f"Gene correlations\n(top {n_top_genes} variable genes, {clustering_method} clustering)\n"
        f"timepoints {timepoints_str}"
    )
    plt.tight_layout()
    return fig, ordered_corr, ordered_genes


# ---------------------------------------------------------------------------
# NMF HEATMAPS AND DYNAMICS
# ---------------------------------------------------------------------------

def plot_nmf_programs_heatmap(
    H: np.ndarray,
    gene_names: List[str],
    prog_list: List[int],
    n_genes: int = 20,
    figsize: Tuple[float, float] = (12, 8),
    kwargs_heatmap: Optional[dict] = None,
):
    """
    Heatmap showing top n_genes per NMF program (vertical program axis).
    """
    if kwargs_heatmap is None:
        kwargs_heatmap = {}

    gene_top_program = np.argmax(H, axis=0)
    heatmap_df = pd.DataFrame(
        H[prog_list, :].T,
        index=gene_names,
        columns=[f"Program {i}" for i in prog_list],
    )

    selected_genes = []
    for prog in prog_list:
        genes_in_prog = np.where(gene_top_program == prog)[0]
        sorted_genes = genes_in_prog[np.argsort(-H[prog, genes_in_prog])]
        selected_genes.extend(sorted_genes[:n_genes])

    selected_genes = list(dict.fromkeys(selected_genes))
    heatmap_df = heatmap_df.loc[np.array(gene_names)[selected_genes]]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cbar_kws={"shrink": 0.25, "aspect": 40, "label": "Loading"},
        **kwargs_heatmap,
    )
    plt.yticks(fontsize=6)
    plt.title("Gene Loading Heatmap Across Programs")
    plt.tight_layout()
    return fig


def plot_nmf_programs_heatmap_horizontal(
    H: np.ndarray,
    gene_names: List[str],
    prog_list: List[int],
    n_genes: int = 20,
    min_val: float = 0.3,
    figsize: Tuple[float, float] = (12, 8),
    kwargs_heatmap: Optional[dict] = None,
):
    """
    Horizontal version of NMF gene-loading heatmap (programs as rows).
    """
    if kwargs_heatmap is None:
        kwargs_heatmap = {}

    gene_top_program = np.argmax(H, axis=0)
    heatmap_df = pd.DataFrame(
        H[prog_list, :], index=[f"Program {i}" for i in prog_list], columns=gene_names
    )

    selected_genes = []
    for prog in prog_list:
        genes_in_prog = np.where(gene_top_program == prog)[0]
        sorted_genes = genes_in_prog[np.argsort(-H[prog, genes_in_prog])]
        filtered_genes = [gene for gene in sorted_genes if H[prog, gene] >= min_val]
        top_genes = filtered_genes[:n_genes]
        selected_genes.extend(top_genes)

    selected_genes = list(dict.fromkeys(selected_genes))
    selected_gene_names = [gene_names[i] for i in selected_genes]
    heatmap_df = heatmap_df[selected_gene_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cbar_kws={"shrink": 0.7, "aspect": 30, "label": "Loading"},
        **kwargs_heatmap,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    plt.title("Gene Loading Heatmap Across Programs", fontsize=14)
    plt.tight_layout()
    return fig


def create_nmf_dynamics_control_normalized_heatmap_grid(
    mean_W: Dict[str, pd.DataFrame],
    cv_W: Dict[str, pd.DataFrame],
    fano_W: Dict[str, pd.DataFrame],
    sort_by: str = "mean",
    figsize: Tuple[float, float] = (3, 6),
    conditions: Optional[List[str]] = None,
):
    """
    Heatmap grid of NMF dynamics normalized to Ctrl over time, for Vor/Aza/Dec.
    """
    if conditions is None:
        conditions = ["Vor", "Aza", "Dec"]

    def get_sorted_programs(metric_W, conditions):
        avg_fold_changes = []
        ctrl_data = metric_W["Ctrl"]
        for cond in conditions:
            treatment_data = metric_W[cond]
            fold_change = treatment_data.divide(ctrl_data)
            log2_fc = np.log2(fold_change)
            avg_fold_changes.append(log2_fc)
        mean_log2_fc = sum(avg_fold_changes) / len(avg_fold_changes)
        avg_magnitude = mean_log2_fc.mean()
        return avg_magnitude.sort_values(ascending=False).index

    if sort_by == "mean":
        sorted_programs = get_sorted_programs(mean_W, conditions)
    elif sort_by == "fano":
        sorted_programs = get_sorted_programs(fano_W, conditions)
    elif sort_by == "cv":
        sorted_programs = get_sorted_programs(cv_W, conditions)
    else:
        raise ValueError("sort_by must be one of: 'mean', 'fano', 'cv'")

    fig = plt.figure(figsize=figsize)
    metrics = ["mean", "cv", "fano"]
    n_conditions = len(conditions)
    gs = gridspec.GridSpec(3, n_conditions + 1, width_ratios=[1] * n_conditions + [0.1])

    data_dict = {"mean": mean_W, "cv": cv_W, "fano": fano_W}
    titles = {
        "mean": r"$\log2(\mu/\mu_{Ctrl})$",
        "cv": r"$\log2(CV/CV_{Ctrl})$",
        "fano": r"$\log2(F/F_{Ctrl})$",
    }
    vmins = {"mean": -4, "cv": -2, "fano": -2}
    vmaxs = {"mean": 4, "cv": 2, "fano": 2}

    for i, metric in enumerate(metrics):
        cbar_ax = fig.add_subplot(gs[i, -1])
        ctrl_data = data_dict[metric]["Ctrl"]

        for j, cond in enumerate(conditions):
            ax = fig.add_subplot(gs[i, j])
            treatment_data = data_dict[metric][cond]
            data_norm = treatment_data.divide(ctrl_data)
            data_sorted = data_norm[list(sorted_programs)].copy()

            sns.heatmap(
                np.log2(data_sorted.T),
                cmap="coolwarm",
                ax=ax,
                cbar=(j == len(conditions) - 1),
                cbar_ax=cbar_ax if j == len(conditions) - 1 else None,
                vmin=vmins[metric],
                vmax=vmaxs[metric],
                xticklabels=True,
                yticklabels=True,
            )

            if i == 0:
                ax.set_title(f"{cond}")
            ax.set_xlabel("Timepoint")

            if j > 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.tick_params(axis="y", labelsize=8)

            if j == len(conditions) - 1:
                cbar_ax.set_ylabel(f"{titles[metric]}")

    plt.tight_layout()
    return fig


def create_nmf_dynamics_heatmap_grid(
    mean_W: Dict[str, pd.DataFrame],
    cv_W: Dict[str, pd.DataFrame],
    fano_W: Dict[str, pd.DataFrame],
    sorted_programs: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 12),
    conditions: Optional[List[str]] = None,
):
    """
    Heatmap grid of NMF mean/CV/Fano over time across conditions.
    """
    if sorted_programs is None:
        log_fc = np.abs(np.log(mean_W["Ctrl"] / mean_W["Ctrl"].iloc[0]))
        max_deviation = log_fc.max()
        sorted_programs = max_deviation.sort_values(ascending=True).index

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 0.1])

    if conditions is None:
        conditions = ["Ctrl", "Vor", "Aza", "Dec"]

    metrics = ["mean", "cv", "fano"]
    data_dict = {"mean": mean_W, "cv": cv_W, "fano": fano_W}
    titles = {
        "mean": r"$\log2(\mu/\mu_0^{Ctrl})$",
        "cv": r"$\log2(CV/CV_0^{Ctrl})$",
        "fano": r"$\log2(F/F_0^{Ctrl})$",
    }
    vmins = {"mean": -4, "cv": -2, "fano": -2}
    vmaxs = {"mean": 4, "cv": 2, "fano": 2}

    for i, metric in enumerate(metrics):
        cbar_ax = fig.add_subplot(gs[i, len(conditions)])
        ctrl_t0 = data_dict[metric]["Ctrl"].iloc[0]

        for j, cond in enumerate(conditions):
            ax = fig.add_subplot(gs[i, j])
            data = data_dict[metric][cond]
            data_norm = data.divide(ctrl_t0, axis=1)
            data_sorted = data_norm[list(sorted_programs)].copy()

            sns.heatmap(
                np.log2(data_sorted.T),
                cmap="coolwarm",
                ax=ax,
                cbar=(j == len(conditions) - 1),
                cbar_ax=cbar_ax if j == len(conditions) - 1 else None,
                vmin=vmins[metric],
                vmax=vmaxs[metric],
                xticklabels=True,
                yticklabels=True,
            )

            if i == 0:
                ax.set_title(f"{cond}")
            ax.set_xlabel("Timepoint")

            if j > 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.tick_params(axis="y", labelsize=8)

            if j == len(conditions) - 1:
                cbar_ax.set_ylabel(f"{titles[metric]}")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NMF MEAN–CV RELATIONSHIPS
# ---------------------------------------------------------------------------

def plot_NMF_mean_cv_relationship(
    mean_dict: Dict[str, pd.DataFrame],
    cv_dict: Dict[str, pd.DataFrame],
    rnd_mean_dict: Dict[str, pd.DataFrame],
    rnd_cv_dict: Dict[str, pd.DataFrame],
    real_fit: dict,
    rnd_fit: dict,
    exclude_keys: List[str] = ["Ctrl"],
):
    """
    Compare CV vs mean relationships for observed vs randomized NMF usage.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    first_plot = True

    for c in mean_dict.keys():
        if c in exclude_keys:
            continue
        for col in mean_dict[c].columns:
            if first_plot:
                ax.loglog(
                    mean_dict[c][col],
                    cv_dict[c][col],
                    "ob",
                    alpha=0.3,
                    markersize=5,
                    label="Observed",
                )
                ax.loglog(
                    rnd_mean_dict[c][col],
                    rnd_cv_dict[c][col],
                    "o",
                    color="grey",
                    alpha=0.3,
                    markersize=5,
                    label="Mock clones",
                )
                first_plot = False
            else:
                ax.loglog(
                    mean_dict[c][col],
                    cv_dict[c][col],
                    "ob",
                    alpha=0.3,
                    markersize=5,
                )
                ax.loglog(
                    rnd_mean_dict[c][col],
                    rnd_cv_dict[c][col],
                    "o",
                    color="grey",
                    alpha=0.3,
                    markersize=5,
                )

    ax.legend()

    xTh = np.logspace(-2.6, -0.3, num=10)
    ax.plot(xTh, real_fit["a"] * xTh ** (-0.5), "--k")
    ax.plot(xTh, rnd_fit["a"] * xTh ** (-0.5), "--k", alpha=0.5)

    ax.set_xlabel("Mean NMF usage (total=1)", fontsize=12)
    ax.set_ylabel("CV NMF usage", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    return fig, ax


def plot_program_boxplot_time_series(
    cv_W_dict: Dict[str, pd.DataFrame],
    ylabel: str = "CV",
    figsize: Tuple[float, float] = (3, 3),
    colors: Optional[List[str]] = None,
    box_width: float = 0.15,
    group_spacing: float = 1.5,
):
    """
    Boxplots of CV over time, comparing multiple conditions or data series.
    """
    fig, ax = plt.subplots(figsize=figsize)
    times = [0, 2, 4, 6]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(cv_W_dict)))

    n_groups = len(times)
    n_items = len(cv_W_dict)
    group_width = box_width * n_items * group_spacing
    offsets = np.linspace(
        -group_width / 2 + box_width / 2, group_width / 2 - box_width / 2, n_items
    )

    bp_objects = []
    for (label, data), offset, color in zip(cv_W_dict.items(), offsets, colors):
        valid_positions = []
        plot_data = []
        data_times = data.index.tolist()

        for time in times:
            if time in data_times:
                valid_positions.append(time + offset)
                idx = data_times.index(time)
                plot_data.append(data.iloc[idx])

        bp = ax.boxplot(
            plot_data,
            positions=valid_positions,
            widths=box_width,
            whis=(5, 95),
            patch_artist=True,
            medianprops=dict(color="black"),
            flierprops={"marker": "none"},
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.7)
        bp_objects.append(bp)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, 7)
    ax.set_xticks(times)
    ax.set_xticklabels([str(t) for t in times])
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")
    ax.legend(
        [bp["boxes"][0] for bp in bp_objects],
        list(cv_W_dict.keys()),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    return fig

