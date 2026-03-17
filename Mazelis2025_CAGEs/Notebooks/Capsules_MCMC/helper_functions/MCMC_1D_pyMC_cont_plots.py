"""
Public plotting helpers for continuous 1D MCMC model outputs.

These wrap ArviZ summary tables and the MCMCInference API used in script2_MCMC.py.
"""

from typing import Dict, List, Tuple, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def plot_fraction_nonzero(mcmc_dict: Dict[str, "MCMCInference"], program_id: int, figsize=(3, 3)):
    """
    Fraction of colonies with NMF usage above threshold over time, for each condition.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])
    colors = {"Ctrl": "black", "Aza": "blue", "Dec": "red", "Vor": "green"}
    conditions = ["Ctrl", "Aza", "Dec", "Vor"]

    for cond in conditions:
        key = f"{cond}-NMF{program_id}"
        if key not in mcmc_dict:
            continue
        mcmc = mcmc_dict[key]
        x = mcmc.data.score.values
        t_vals = mcmc.data.timepoint.values
        summary = az.summary(mcmc.trace)
        threshold = float(summary.loc["threshold", "mean"])

        empirical_nonzero = []
        empirical_sems = []
        for t in mcmc.times:
            x_t = x[t_vals == t]
            nonzero = x_t > threshold
            frac = float(np.mean(nonzero))
            n_samples = len(x_t)
            sem = float(np.sqrt(frac * (1 - frac) / n_samples))
            empirical_nonzero.append(frac)
            empirical_sems.append(sem)

        # model-predicted fraction via surrogate stored on mcmc
        predicted_nonzero = []
        for t in mcmc.times:
            if t == 0:
                px = np.array(
                    [
                        float(summary.loc["p0", "mean"]),
                        1 - float(summary.loc["p0", "mean"]),
                    ]
                )
            else:
                q01 = float(summary.loc["q01", "mean"])
                q10 = float(summary.loc["q10", "mean"])
                params = np.array([[q01, q10]])
                px = mcmc.surrogate_ops[t].torch_model.predict(
                    params, q0=float(summary.loc["p0", "mean"])
                )[0]
            predicted_nonzero.append(float(np.sum(px[1:])))

        ax.errorbar(
            mcmc.times,
            empirical_nonzero,
            yerr=empirical_sems,
            fmt="o",
            color=colors[cond],
            label=f"{cond} data",
            capsize=5,
            capthick=1,
            elinewidth=1,
            markersize=5,
        )
        ax.plot(
            mcmc.times,
            predicted_nonzero,
            "-",
            color=colors[cond],
            label=f"{cond} model",
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"Fraction colonies, NMF$>\epsilon$")
    ax.set_title(f"Program {program_id}")
    ax.set_xlim(-1, 7)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig


def plot_mean_usage(mcmc_dict: Dict[str, "MCMCInference"], program_id: int, figsize=(3, 3)):
    """
    Mean score among non-zero colonies over time, for each condition.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])
    colors = {"Ctrl": "black", "Aza": "blue", "Dec": "red", "Vor": "green"}
    conditions = ["Ctrl", "Aza", "Dec", "Vor"]

    for cond in conditions:
        key = f"{cond}-NMF{program_id}"
        if key not in mcmc_dict:
            continue
        mcmc = mcmc_dict[key]
        x = mcmc.data.score.values
        t_vals = mcmc.data.timepoint.values
        summary = az.summary(mcmc.trace)
        threshold = float(summary.loc["threshold", "mean"])
        a = float(summary.loc["a", "mean"])
        sigma = float(summary.loc["sigma", "mean"])
        bb = float(summary.loc["bb", "mean"])
        s_min = float(summary.loc["s_min", "mean"])
        p0 = float(summary.loc["p0", "mean"])
        q01 = float(summary.loc["q01", "mean"])
        q10 = float(summary.loc["q10", "mean"])

        empirical_means = []
        empirical_sems = []
        predicted_means = []

        for t in mcmc.times:
            x_t = x[t_vals == t]
            nonzero_mask = x_t > threshold
            x_nonzero = x_t[nonzero_mask]
            empirical_means.append(float(np.mean(x_nonzero)))
            empirical_sems.append(
                float(np.std(x_nonzero) / np.sqrt(len(x_nonzero)))
            )

            if t == 0:
                px = np.array([p0, 1 - p0])
                n_values = np.array([0, 1])
            else:
                n_values = np.linspace(0, 1, 2**t + 1)
                params = np.array([[q01, q10]])
                px = mcmc.surrogate_ops[t].torch_model.predict(params, q0=p0)[0]

            pred_mean = 0.0
            for n, p_n in zip(n_values[1:], px[1:]):
                s_nt = np.sqrt(
                    np.log(1 + (np.exp(sigma**2) - 1) / (n * 2**t)) + s_min**2
                )
                mu_nt = (
                    np.log(n * 2**t) * bb
                    - np.log(2**t)
                    + a
                    + (sigma**2 - s_nt**2) / 2
                )
                pred_mean += p_n * np.exp(mu_nt + s_nt**2 / 2) / (1 - px[0])

            predicted_means.append(float(pred_mean))

        ax.errorbar(
            mcmc.times,
            empirical_means,
            yerr=empirical_sems,
            fmt="o",
            color=colors[cond],
            label=f"{cond} data",
            capsize=5,
            capthick=1,
            elinewidth=1,
            markersize=5,
        )
        ax.plot(
            mcmc.times, predicted_means, "-", color=colors[cond], label=f"{cond} model"
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mean program usage\n(non-zero colonies)")
    ax.set_title(f"Program {program_id}")
    ax.set_xlim(-1, 7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig


def plot_transition_rates(data, program_id: int, figsize=(3, 3)):
    """
    Horizontal bar plots of r01 and r10 vs condition for one program.

    Accepts either:
    - cont_results_df DataFrame produced by create_mcmc_summary_row, or
    - mcmc_dict of MCMCInference objects.
    """
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.15, 0.15, 0.65, 0.3])
    ax2 = fig.add_axes([0.15, 0.55, 0.65, 0.3])

    conditions = ["Ctrl", "Aza", "Dec", "Vor"]
    r01_values = []
    r10_values = []
    r01_errors_low = []
    r01_errors_high = []
    r10_errors_low = []
    r10_errors_high = []

    is_dataframe = hasattr(data, "columns")

    for cond in conditions:
        if is_dataframe:
            row = data[
                (data["program_id"] == program_id) & (data["condition"] == cond)
            ]
            if len(row) == 0:
                continue
            r01_mean = float(row["r01_mean"].values[0])
            r10_mean = float(row["r10_mean"].values[0])
            r01_low = float(row["r01_hdi_low"].values[0])
            r01_high = float(row["r01_hdi_high"].values[0])
            r10_low = float(row["r10_hdi_low"].values[0])
            r10_high = float(row["r10_hdi_high"].values[0])
        else:
            key = f"{cond}-NMF{program_id}"
            if key not in data:
                continue
            mcmc = data[key]
            summary = az.summary(mcmc.trace)
            r01_mean = float(summary.loc["r01", "mean"])
            r10_mean = float(summary.loc["r10", "mean"])
            r01_low = float(summary.loc["r01", "hdi_3%"])
            r01_high = float(summary.loc["r01", "hdi_97%"])
            r10_low = float(summary.loc["r10", "hdi_3%"])
            r10_high = float(summary.loc["r10", "hdi_97%"])

        r01_values.append(r01_mean)
        r10_values.append(r10_mean)
        r01_errors_low.append(r01_mean - r01_low)
        r01_errors_high.append(r01_high - r01_mean)
        r10_errors_low.append(r10_mean - r10_low)
        r10_errors_high.append(r10_high - r10_mean)

    y_pos = np.arange(len(conditions))
    ax1.barh(
        y_pos,
        r01_values,
        xerr=[r01_errors_low, r01_errors_high],
        color="0.3",
        height=0.6,
        capsize=3,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(conditions)
    ax1.set_xlabel(r"$r_{01}$ (cell cycle$^{-1}$)")

    ax2.barh(
        y_pos,
        r10_values,
        xerr=[r10_errors_low, r10_errors_high],
        color="0.3",
        height=0.6,
        capsize=3,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(conditions)
    ax2.set_xlabel(r"$r_{10}$ (cell cycle$^{-1}$)")

    max_limit = max(
        max(r01_values) + max(r01_errors_high),
        max(r10_values) + max(r10_errors_high),
    ) * 1.1
    max_tick = np.ceil(max_limit * 10) / 10
    ax1.set_xlim(0, max_tick)
    ax2.set_xlim(0, max_tick)
    ticks = np.arange(0, max_tick + 0.05, 0.1)
    ax1.set_xticks(ticks)
    ax2.set_xticks(ticks)
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    fig.suptitle(f"Program {program_id}", y=0.95)
    return fig


def plot_histograms(
    mcmc_dict: Dict[str, "MCMCInference"], condition: str, program_id: int, figsize=(4, 6)
):
    """
    Histograms of log10(threshold+score) vs model prediction, stacked over time.
    """
    key = f"{condition}-NMF{program_id}"
    mcmc = mcmc_dict[key]
    x = mcmc.data.score.values
    t_vals = mcmc.data.timepoint.values
    summary = az.summary(mcmc.trace)
    threshold = float(summary.loc["threshold", "mean"])
    a = float(summary.loc["a", "mean"])
    sigma = float(summary.loc["sigma", "mean"])
    bb = float(summary.loc["bb", "mean"])
    s_min = float(summary.loc["s_min", "mean"])
    p0 = float(summary.loc["p0", "mean"])
    q01 = float(summary.loc["q01", "mean"])
    q10 = float(summary.loc["q10", "mean"])

    fig, axs = plt.subplots(4, 1, figsize=figsize, height_ratios=[1, 1, 1, 1.2])
    plt.subplots_adjust(hspace=0.3)
    axs[0].set_title(f"{condition} - Program {program_id}")

    all_data = []
    for t in mcmc.times:
        x_t = x[t_vals == t]
        all_data.extend(np.log10(threshold + x_t))
    x_min, x_max = min(all_data), max(all_data)

    for i, t in enumerate(mcmc.times):
        ax = axs[i]
        x_t = x[t_vals == t]

        hist_vals, _, _ = ax.hist(
            np.log10(threshold + x_t),
            bins=30,
            density=True,
            alpha=0.5,
            color="green",
            label="Data",
        )

        if t == 0:
            px = np.array([p0, 1 - p0])
            n_values = np.array([0, 1])
        else:
            n_values = np.linspace(0, 1, 2**t + 1)
            params = np.array([[q01, q10]])
            px = mcmc.surrogate_ops[t].torch_model.predict(params, q0=p0)[0]

        x_plot = np.logspace(
            np.log10(threshold), np.log10(threshold + 1), 100
        )
        y_plot = np.zeros_like(x_plot)

        for n, p_n in zip(n_values[1:], px[1:]):
            s_nt = np.sqrt(
                np.log(1 + (np.exp(sigma**2) - 1) / (n * 2**t)) + s_min**2
            )
            mu_nt = (
                np.log(n * 2**t) * bb
                - np.log(2**t)
                + a
                + (sigma**2 - s_nt**2) / 2
            )
            y_plot += (
                p_n
                * np.exp(-(np.log(x_plot) - mu_nt) ** 2 / (2 * s_nt**2))
                / (s_nt * np.sqrt(2 * np.pi))
            )

        scaling_factor = np.max(hist_vals[5:]) / np.max(
            y_plot[x_plot > threshold]
        )
        y_plot *= scaling_factor
        ax.plot(np.log10(x_plot), y_plot, "k-", label="Model")
        ax.text(0.06, 0.7, f"Day {t}", transform=ax.transAxes, ha="left")
        ax.set_xlim(x_min, x_max)

        if i != 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("log10(threshold + score)")
        if i == 1:
            ax.set_ylabel("Density")

    return fig


def plot_relative_transition_rates(
    df: pd.DataFrame, param: str = "S", figsize=(8, 10), xscale: str = "linear", xlim=None
):
    """
    Horizontal bar chart of relative parameter values (drug / control) for each program.
    """
    valid_params = {
        "S": "total transition rate",
        "r01": "off→on transition rate",
        "r10": "on→off transition rate",
        "r": "net transition rate",
        "1/r01": "off state persistence time",
        "1/r10": "on state persistence time",
    }
    if param not in valid_params:
        raise ValueError(f"param must be one of {list(valid_params.keys())}")

    if param == "1/r01":
        mean_col = "r01_mean"
        std_col = "r01_std"
    elif param == "1/r10":
        mean_col = "r10_mean"
        std_col = "r10_std"
    else:
        mean_col = f"{param}_mean"
        std_col = f"{param}_std"

    colors = {"Aza": "#1f77b4", "Dec": "#2ca02c", "Vor": "#d62728"}
    programs = range(df["program_id"].max() + 1)
    conditions = ["Aza", "Dec", "Vor"]

    data = {cond: [] for cond in conditions}
    errors = {cond: [] for cond in conditions}

    for prog in programs:
        ctrl_data = df[(df["program_id"] == prog) & (df["condition"] == "Ctrl")]
        if len(ctrl_data) == 0:
            continue
        val_ctrl = ctrl_data[mean_col].values[0]
        sem_ctrl = ctrl_data[std_col].values[0]

        for cond in conditions:
            cond_data = df[
                (df["program_id"] == prog) & (df["condition"] == cond)
            ]
            if len(cond_data) == 0:
                continue
            val_cond = cond_data[mean_col].values[0]
            sem_cond = cond_data[std_col].values[0]

            rel_rate = val_cond / val_ctrl
            if param in ["1/r01", "1/r10"]:
                rel_rate = 1 / rel_rate

            rel_error = rel_rate * np.sqrt(
                (sem_cond / val_cond) ** 2 + (sem_ctrl / val_ctrl) ** 2
            )
            data[cond].append(rel_rate)
            errors[cond].append(rel_error)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.25
    y_pos = np.arange(len(programs))

    for i, cond in enumerate(conditions):
        pos = y_pos + i * bar_width
        values = np.array(data[cond])
        if xscale == "log":
            log_vals = np.log10(values)
            left = np.where(log_vals >= 0, 0, log_vals)
            width = np.abs(log_vals)
            ax.barh(
                pos,
                width,
                bar_width,
                left=left,
                label=cond,
                color=colors[cond],
                alpha=0.7,
            )
            err_log = np.array(errors[cond]) / (values * np.log(10))
            ax.errorbar(
                log_vals, pos, xerr=err_log, fmt="none", color="black", capsize=3
            )
        else:
            ax.barh(
                pos,
                values,
                bar_width,
                label=cond,
                color=colors[cond],
                alpha=0.7,
            )
            ax.errorbar(
                values, pos, xerr=errors[cond], fmt="none", color="black", capsize=3
            )

    if xscale == "log":
        ax.axvline(0, color="k", linewidth=1)
        if xlim is None:
            xlim = 10 ** np.array(ax.get_xlim())
        min_log, max_log = np.round(np.log10(xlim[0]), 1), np.round(
            np.log10(xlim[1]), 1
        )
        xtick_positions = np.log10(np.logspace(min_log, max_log, num=2))
        if 0 not in xtick_positions:
            xtick_positions = np.sort(np.concatenate((xtick_positions, [0])))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{tick:.2f}" for tick in xtick_positions])
        ax.set_xlim(min_log, max_log)

    ax.set_yticks(y_pos + bar_width)
    ax.set_yticklabels([f"{i}" for i in programs])
    ax.invert_yaxis()
    ax.set_ylabel("NMF Program")
    if xscale == "log":
        ax.set_xlabel(
            f"Relative {valid_params[param]} log10({param}/{param}_ctrl)"
        )
    else:
        ax.set_xlabel(
            f"Relative {valid_params[param]} ({param}/{param}_ctrl)"
        )
    ax.set_ylim(len(programs) - 0.2, -0.2)
    if xscale != "log":
        ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_relative_transition_rates_heatmap(
    df: pd.DataFrame,
    param: str = "S",
    figsize=(8, 6),
    clim: Optional[Tuple[float, float]] = None,
    show_annotations: bool = False,
):
    """
    Heatmap of log2(drug/control) for transition parameters across programs.
    """
    valid_params = {
        "S": "total transition rate",
        "r01": "off→on transition rate",
        "r10": "on→off transition rate",
        "r": "net transition rate",
        "min_r": "minimum transition rate",
        "min_t": "maximum persistence time",
        "1/r01": "off state persistence time",
        "1/r10": "on state persistence time",
        "1/S": "inverse total transition rate",
        "t1_frac": "ON state persistence time fraction",
    }
    if param not in valid_params:
        raise ValueError(f"param must be one of {list(valid_params.keys())}")

    if param in ["1/r01", "1/r10"]:
        mean_col = param.replace("1/", "") + "_mean"
    else:
        mean_col = f"{param}_mean"

    programs = range(df["program_id"].max() + 1)
    conditions = ["Aza", "Dec", "Vor"]
    rate_matrix = np.full((len(programs), len(conditions)), np.nan)

    for i, prog in enumerate(programs):
        ctrl_data = df[(df["program_id"] == prog) & (df["condition"] == "Ctrl")]
        if len(ctrl_data) == 0:
            continue

        if param == "t1_frac":
            val_ctrl = ctrl_data["r01_mean"].values[0] / (
                ctrl_data["r01_mean"].values[0] + ctrl_data["r10_mean"].values[0]
            )
        elif param == "1/S":
            val_ctrl = 1 / ctrl_data["S_mean"].values[0]
        elif param in ["min_r", "min_t"]:
            val_ctrl = min(
                ctrl_data["r01_mean"].values[0], ctrl_data["r10_mean"].values[0]
            )
        else:
            val_ctrl = ctrl_data[mean_col].values[0]

        for j, cond in enumerate(conditions):
            cond_data = df[
                (df["program_id"] == prog) & (df["condition"] == cond)
            ]
            if len(cond_data) == 0:
                continue

            if param == "t1_frac":
                val_cond = cond_data["r01_mean"].values[0] / (
                    cond_data["r01_mean"].values[0]
                    + cond_data["r10_mean"].values[0]
                )
                rel_rate = val_cond - val_ctrl
                rate_matrix[i, j] = rel_rate
                continue
            elif param == "1/S":
                val_cond = 1 / cond_data["S_mean"].values[0]
            elif param in ["min_r", "min_t"]:
                val_cond = min(
                    cond_data["r01_mean"].values[0],
                    cond_data["r10_mean"].values[0],
                )
            else:
                val_cond = cond_data[mean_col].values[0]

            rel_rate = val_cond / val_ctrl
            if param in ["1/r01", "1/r10", "min_t"]:
                rel_rate = 1 / rel_rate
            rate_matrix[i, j] = np.log2(rel_rate)

    program_means = np.nanmean(rate_matrix, axis=1)
    sort_indices = np.argsort(program_means)[::-1]
    rate_matrix = rate_matrix[sort_indices]
    program_order = [str(i) for i in sort_indices]

    fig, ax = plt.subplots(figsize=figsize)
    if clim is None:
        max_abs = np.nanmax(np.abs(rate_matrix))
        clim = (-max_abs, max_abs)

    im = ax.imshow(rate_matrix, cmap="coolwarm", aspect="auto", clim=clim)
    cbar = plt.colorbar(im)
    if param == "t1_frac":
        cbar.set_label("Δ(ON state persistence time fraction)")
    else:
        cbar.set_label(f"log2(relative {valid_params[param]})")

    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(program_order)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("NMF Program")
    ax.set_title(f"Relative {valid_params[param]}")

    if show_annotations:
        for i in range(len(programs)):
            for j in range(len(conditions)):
                if not np.isnan(rate_matrix[i, j]):
                    if param == "t1_frac":
                        text = f"{rate_matrix[i, j]:.2f}"
                    else:
                        text = f"{2**rate_matrix[i, j]:.2f}"
                    color = (
                        "white"
                        if abs(rate_matrix[i, j]) > (clim[1] - clim[0]) / 4
                        else "black"
                    )
                    ax.text(j, i, text, ha="center", va="center", color=color)

    plt.tight_layout()
    return fig


def calculate_gene_persistence_times(
    cont_results_df: pd.DataFrame,
    H: pd.DataFrame,
    conditions: List[str] = ["Ctrl", "Aza", "Dec", "Vor"],
    debug: bool = False,
):
    """
    Gene-level persistence metrics by projecting program-level rates using loadings H.
    """
    persistence_times: Dict[str, Dict[str, np.ndarray]] = {"r01": {}, "r10": {}, "1/S": {}}

    if debug:
        print("First 3 genes' loadings into components:")
        print(H.head(3))
        print("\nShape of H matrix:", H.shape)

    for cond in conditions:
        cond_data = cont_results_df[cont_results_df["condition"] == cond]
        r01_values = cond_data["r01_mean"].values
        r10_values = cond_data["r10_mean"].values
        persistence_by_program = {
            "r01": 1 / r01_values,
            "r10": 1 / r10_values,
            "1/S": 1 / (r01_values + r10_values),
        }

        if debug and cond == conditions[0]:
            print(f"\nPersistence times by program for {cond}:")
            for i in range(len(r01_values)):
                print(f"Program {i}:")
                print(f"  off→on time: {persistence_by_program['r01'][i]:.2f}")
                print(f"  on→off time: {persistence_by_program['r10'][i]:.2f}")
                print(f"  1/S time: {persistence_by_program['1/S'][i]:.2f}")

        H_normalized = H.div(H.sum(axis=0), axis=1)
        H_normalized = H_normalized.div(H_normalized.sum(axis=1), axis=0)

        for metric in ["r01", "r10", "1/S"]:
            gene_persistence = np.dot(H_normalized, persistence_by_program[metric])
            if debug and cond == conditions[0] and metric == "r01":
                print(f"\nFirst 3 genes calculation details ({metric}):")
                for i in range(3):
                    gene_name = H.index[i]
                    print(f"\nGene: {gene_name}")
                    print("Normalized loadings:", H_normalized.iloc[i].values)
                    print("Final persistence time:", gene_persistence[i])
            if cond not in persistence_times[metric]:
                persistence_times[metric][cond] = gene_persistence

    result = {
        "r01": pd.DataFrame(persistence_times["r01"], index=H.index),
        "r10": pd.DataFrame(persistence_times["r10"], index=H.index),
        "1/S": pd.DataFrame(persistence_times["1/S"], index=H.index),
    }
    return result


def plot_persistence_vs_expression_density(
    deg_results_dict: Dict[str, pd.DataFrame],
    gene_persistence_times_dict: Dict[str, pd.DataFrame],
    drug: str,
    transition_type: str = "1/S",
    figsize=(4, 4),
    H: Optional[pd.DataFrame] = None,
    top_n_genes: Optional[int] = None,
    annotate_genes: Optional[List[str]] = None,
    bins: int = 50,
    levels: int = 8,
    color: str = "red",
    min_logFC: float = 0.1,
    smooth_sigma: float = 10,
    show_sparse_points: bool = True,
    sparse_threshold: float = 0.1,
    annotate_outliers: bool = False,
):
    """
    Density plot of log2 fold-change in expression vs log2 persistence ratio for one drug.
    """
    if top_n_genes is not None and H is None:
        raise ValueError("H must be provided when top_n_genes is specified")

    fig, ax = plt.subplots(figsize=figsize)
    colors = [(1, 1, 1), {"red": (1, 0, 0), "green": (0, 0.6, 0)}[color]]
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

    included_genes = None
    if top_n_genes is not None:
        included_genes = set()
        for program in H.columns:
            top_genes = H.nlargest(top_n_genes, program).index
            included_genes.update(top_genes)

    if transition_type == "slowest":
        off_to_on_df = gene_persistence_times_dict["r01"]
        on_to_off_df = gene_persistence_times_dict["r10"]
        ctrl_off = off_to_on_df["Ctrl"]
        drug_off = off_to_on_df[drug]
        ctrl_on = on_to_off_df["Ctrl"]
        drug_on = on_to_off_df[drug]
        ratio_off = drug_off / ctrl_off
        ratio_on = drug_on / ctrl_on
        persistence_ratio = np.log2(
            pd.concat([ratio_off, ratio_on], axis=1).max(axis=1)
        )
    else:
        gene_persistence_times_df = gene_persistence_times_dict[transition_type]
        ctrl_persistence = gene_persistence_times_df["Ctrl"]
        drug_persistence = gene_persistence_times_df[drug]
        persistence_ratio = np.log2(drug_persistence / ctrl_persistence)

    plot_df = pd.DataFrame(
        {
            "logFC": deg_results_dict[drug]["mean_logFC"] * np.log2(10),
            "log2_persistence_ratio": persistence_ratio,
            "FDR": deg_results_dict[drug]["combined_fdr"],
        }
    )
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[abs(plot_df["logFC"]) >= min_logFC]

    y_min, y_max = plot_df["log2_persistence_ratio"].min(), plot_df[
        "log2_persistence_ratio"
    ].max()
    y_range = y_max - y_min
    y_limits = [y_min - 0.1 * y_range, y_max + 0.1 * y_range]

    x = plot_df["logFC"]
    y = plot_df["log2_persistence_ratio"]
    h, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[x.min(), x.max()], y_limits]
    )
    h = h.T

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    h_smooth = gaussian_filter(h, sigma=bins / smooth_sigma)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        np.log1p(h_smooth),
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=custom_cmap,
    )

    if h_smooth.max() > 0:
        min_level = h_smooth.max() * 0.05
        contour_levels = np.linspace(min_level, h_smooth.max(), levels)
        ax.contour(
            xcenters,
            ycenters,
            h_smooth,
            levels=contour_levels,
            colors="black",
            alpha=0.3,
            linewidths=0.5,
        )

    if show_sparse_points:
        density_interp = RegularGridInterpolator(
            (ycenters, xcenters), h_smooth, bounds_error=False, fill_value=0
        )
        points = np.column_stack([y, x])
        point_densities = density_interp(points)
        sparse_mask = point_densities < (h_smooth.max() * sparse_threshold)
        if np.any(sparse_mask):
            ax.scatter(
                x[sparse_mask],
                y[sparse_mask],
                c="black",
                s=1,
                alpha=0.2,
                zorder=1,
            )
            if annotate_outliers:
                for gene in plot_df.index[sparse_mask]:
                    gene_x = plot_df.loc[gene, "logFC"]
                    gene_y = plot_df.loc[gene, "log2_persistence_ratio"]
                    ax.annotate(
                        gene,
                        (gene_x, gene_y),
                        fontsize=6,
                        textcoords="offset points",
                        xytext=(10, 10),
                        ha="center",
                        va="center",
                    )

    if annotate_genes is not None:
        for gene in annotate_genes:
            if gene in plot_df.index:
                gx = plot_df.loc[gene, "logFC"]
                gy = plot_df.loc[gene, "log2_persistence_ratio"]
                ax.scatter(gx, gy, c="black", s=50, alpha=1.0)
                display_name = gene.replace("_hg", "").replace("_mm", "")
                ax.annotate(
                    display_name,
                    (gx, gy),
                    xytext=(15, 15),
                    textcoords="offset points",
                    fontsize=12,
                    alpha=1.0,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        alpha=1.0,
                        color="black",
                        connectionstyle="arc3,rad=0.1",
                        shrinkB=5,
                        mutation_scale=8,
                    ),
                )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xlabel("Log2 Fold Change (Expression)")

    if transition_type == "slowest":
        ylabel = "Log2 Ratio of\nMaximum Persistence Time\n(Drug/Control)"
    elif transition_type == "r01":
        ylabel = "Log2 Ratio of\nOFF→ON Persistence Time\n(Drug/Control)"
    elif transition_type == "r10":
        ylabel = "Log2 Ratio of\nON→OFF Persistence Time\n(Drug/Control)"
    elif transition_type == "1/S":
        ylabel = "Log2 Ratio of\nPersistence Time\n(Drug/Control)"
    else:
        ylabel = "Log2 Persistence Ratio"
    ax.set_ylabel(ylabel)

    ax.set_title(f"{drug} ({transition_type})")
    cbar = fig.colorbar(im, ax=ax, label="Density")
    plt.tight_layout()
    return fig

