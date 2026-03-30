import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator

def plot_fraction_nonzero(mcmc_dict, program_id, figsize=(3, 3)):
    """
    Plot fraction of non-zero programs for all conditions.
    
    Args:
        mcmc_dict: Dictionary of MCMCInference objects keyed by 'condition-NMF#'
        program_id: Program number to plot
    """
    # Create figure with square dimensions
    fig = plt.figure(figsize=figsize)
    
    # Create axis with specific dimensions - make width equal to height
    ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])  # [left, bottom, width, height]
    
    # Set style
    colors = {'Ctrl': 'black', 'Aza': 'blue', 'Dec': 'red', 'Vor': 'green'}
    conditions = ['Ctrl', 'Aza', 'Dec', 'Vor']
    
    for cond in conditions:
        key = f'{cond}-NMF{program_id}'
        mcmc = mcmc_dict[key]
        
        # Get data and parameters
        x = mcmc.data.score.values
        t_vals = mcmc.data.timepoint.values
        est_params = {
            'threshold': float(az.summary(mcmc.trace).loc['threshold', 'mean']),
            'p0': float(az.summary(mcmc.trace).loc['p0', 'mean']),
            'q01': float(az.summary(mcmc.trace).loc['q01', 'mean']),
            'q10': float(az.summary(mcmc.trace).loc['q10', 'mean'])
        }
        
        # Calculate empirical fractions and SEMs
        empirical_nonzero = []
        empirical_sems = []
        for t in mcmc.times:
            x_t = x[t_vals == t]
            nonzero = (x_t > est_params['threshold'])
            frac = np.mean(nonzero)
            n_samples = len(x_t)
            sem = np.sqrt(frac * (1-frac) / n_samples)
            empirical_nonzero.append(frac)
            empirical_sems.append(sem)
            #print(f"{cond}, t={t}: frac={frac:.3f}, N={n_samples}, SEM={sem:.3f}")
        
        # Calculate predicted fractions
        predicted_nonzero = []
        for t in mcmc.times:
            if t == 0:
                px = np.array([est_params['p0'], 1-est_params['p0']])
            else:
                params = np.array([[est_params['q01'], est_params['q10']]])
                px = mcmc.surrogate_ops[t].torch_model.predict(params, q0=est_params['p0'])[0]
            predicted_nonzero.append(np.sum(px[1:]))
        
        # Plot with larger error bars
        ax.errorbar(mcmc.times, empirical_nonzero, yerr=empirical_sems, 
                   fmt='o', color=colors[cond], label=f'{cond} data',
                   capsize=5, capthick=1, elinewidth=1, markersize=5)
        ax.plot(mcmc.times, predicted_nonzero, '-', color=colors[cond], 
                label=f'{cond} model')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(r'Fraction colonies, NMF$>\epsilon$')
    ax.set_title(f'Program {program_id}')
    
    # Set axis limits
    ax.set_xlim(-1, 7)
    ax.set_ylim(0, 1)
    
    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig

def plot_mean_usage(mcmc_dict, program_id, figsize=(3, 3)):
    """
    Plot mean program usage for non-zero programs across all conditions.
    
    Args:
        mcmc_dict: Dictionary of MCMCInference objects keyed by 'condition-NMF#'
        program_id: Program number to plot
    """
    # Create figure with square dimensions
    fig = plt.figure(figsize=figsize)
    
    # Create axis with specific dimensions - make width equal to height
    ax = fig.add_axes([0.15, 0.15, 0.65, 0.65])  # [left, bottom, width, height]
    
    colors = {'Ctrl': 'black', 'Aza': 'blue', 'Dec': 'red', 'Vor': 'green'}
    conditions = ['Ctrl', 'Aza', 'Dec', 'Vor']
    
    for cond in conditions:
        key = f'{cond}-NMF{program_id}'
        mcmc = mcmc_dict[key]
        
        # Get data and parameters
        x = mcmc.data.score.values
        t_vals = mcmc.data.timepoint.values
        est_params = {
            'threshold': float(az.summary(mcmc.trace).loc['threshold', 'mean']),
            'p0': float(az.summary(mcmc.trace).loc['p0', 'mean']),
            'q01': float(az.summary(mcmc.trace).loc['q01', 'mean']),
            'q10': float(az.summary(mcmc.trace).loc['q10', 'mean']),
            'a': float(az.summary(mcmc.trace).loc['a', 'mean']),
            'sigma': float(az.summary(mcmc.trace).loc['sigma', 'mean']),
            'bb': float(az.summary(mcmc.trace).loc['bb', 'mean']),
            's_min': float(az.summary(mcmc.trace).loc['s_min', 'mean'])
        }
        
        # Calculate empirical means and SEMs
        empirical_means = []
        empirical_sems = []
        for t in mcmc.times:
            x_t = x[t_vals == t]
            nonzero_mask = (x_t > est_params['threshold'])
            x_nonzero = x_t[nonzero_mask]
            empirical_means.append(np.mean(x_nonzero))
            empirical_sems.append(np.std(x_nonzero)/np.sqrt(len(x_nonzero)))
        
        # Calculate predicted means
        predicted_means = []
        for t in mcmc.times:
            if t == 0:
                px = np.array([est_params['p0'], 1-est_params['p0']])
                n_values = np.array([0, 1])
            else:
                n_values = np.linspace(0, 1, 2**t + 1)
                params = np.array([[est_params['q01'], est_params['q10']]])
                px = mcmc.surrogate_ops[t].torch_model.predict(params, q0=est_params['p0'])[0]
            
            predicted_mean = 0
            for n, p_n in zip(n_values[1:], px[1:]):  # Skip n=0
                s_nt = np.sqrt(np.log(1 + (np.exp(est_params['sigma']**2)-1)/(n*2**t)) + 
                             est_params['s_min']**2)
                mu_nt = (np.log(n*2**t)*est_params['bb']) - np.log(2**t) + \
                       est_params['a'] + (est_params['sigma']**2 - s_nt**2) / 2
                predicted_mean += p_n * np.exp(mu_nt + s_nt**2/2)/(1-px[0])
            predicted_means.append(predicted_mean)
        
        # Plot with larger error bars
        ax.errorbar(mcmc.times, empirical_means, yerr=empirical_sems, 
                   fmt='o', color=colors[cond], label=f'{cond} data',
                   capsize=5, capthick=1, elinewidth=1, markersize=5)
        ax.plot(mcmc.times, predicted_means, '-', color=colors[cond], 
                label=f'{cond} model')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Mean program usage\n(non-zero colonies)')
    ax.set_title(f'Program {program_id}')
    
    # Set axis limits
    ax.set_xlim(-1, 7)
    
    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig

def plot_histograms(mcmc_dict, condition, program_id, figsize=(4, 6)):
    """
    Plot histograms of program usage for one condition, vertically stacked.
    
    Args:
        mcmc_dict: Dictionary of MCMCInference objects keyed by 'condition-NMF#'
        condition: Condition to plot
        program_id: Program number to plot
    """
    key = f'{condition}-NMF{program_id}'
    mcmc = mcmc_dict[key]
    
    # Get data and parameters
    x = mcmc.data.score.values
    t_vals = mcmc.data.timepoint.values
    est_params = {
        'threshold': float(az.summary(mcmc.trace).loc['threshold', 'mean']),
        'p0': float(az.summary(mcmc.trace).loc['p0', 'mean']),
        'q01': float(az.summary(mcmc.trace).loc['q01', 'mean']),
        'q10': float(az.summary(mcmc.trace).loc['q10', 'mean']),
        'a': float(az.summary(mcmc.trace).loc['a', 'mean']),
        'sigma': float(az.summary(mcmc.trace).loc['sigma', 'mean']),
        'bb': float(az.summary(mcmc.trace).loc['bb', 'mean']),
        's_min': float(az.summary(mcmc.trace).loc['s_min', 'mean'])
    }
    
    # Create figure with vertically stacked subplots
    fig, axs = plt.subplots(4, 1, figsize=figsize, height_ratios=[1, 1, 1, 1.2])
    plt.subplots_adjust(hspace=0.3)
    
    # Set title for the whole figure
    axs[0].set_title(f'{condition} - Program {program_id}')
    
    # Find global x limits
    all_data = []
    for t in mcmc.times:
        x_t = x[t_vals == t]
        all_data.extend(np.log10(est_params['threshold'] + x_t))
    x_min, x_max = min(all_data), max(all_data)
    
    for i, t in enumerate(mcmc.times):
        ax = axs[i]
        x_t = x[t_vals == t]
        
        # Plot empirical distribution
        hist_vals, _, _ = ax.hist(np.log10(est_params['threshold'] + x_t), 
                bins=30, density=True, alpha=0.5, color='green', label='Data')
        
        # Generate predicted distribution
        if t == 0:
            px = np.array([est_params['p0'], 1-est_params['p0']])
            n_values = np.array([0, 1])
        else:
            n_values = np.linspace(0, 1, 2**t + 1)
            params = np.array([[est_params['q01'], est_params['q10']]])
            px = mcmc.surrogate_ops[t].torch_model.predict(params, q0=est_params['p0'])[0]
        
        # Generate points for predicted distribution
        x_plot = np.logspace(np.log10(est_params['threshold']), 
                           np.log10(est_params['threshold'] + 1), 100)
        y_plot = np.zeros_like(x_plot)
        
        # Add contribution from each n>0
        for n, p_n in zip(n_values[1:], px[1:]):
            s_nt = np.sqrt(np.log(1 + (np.exp(est_params['sigma']**2)-1)/(n*2**t)) + 
                         est_params['s_min']**2)
            mu_nt = (np.log(n*2**t)*est_params['bb']) - np.log(2**t) + \
                   est_params['a'] + (est_params['sigma']**2 - s_nt**2) / 2
            y_plot += p_n * np.exp(-(np.log(x_plot) - mu_nt)**2 / 
                         (2*s_nt**2)) / (s_nt*np.sqrt(2*np.pi))
        
        scaling_factor = np.max(hist_vals[5:]) / np.max(y_plot[x_plot > est_params['threshold']])
        y_plot *= scaling_factor
        ax.plot(np.log10(x_plot), y_plot, 'k-', label='Model')
        
        # Add "Day X" label
        ax.text(0.06, 0.7, f'Day {t}', transform=ax.transAxes,
                horizontalalignment='left')
        
        # Only add legend to first plot
        #if i == 0:
        #    ax.legend(frameon=False)
        
        # Set consistent x limits
        ax.set_xlim(x_min, x_max)
        
        # Remove x tick labels except for bottom plot
        if i != 3:
            ax.set_xticklabels([])
        
        # Only add x-label to bottom plot
        if i == 3:
            ax.set_xlabel('log10(threshold + score)')
        
        # Only add y-label to middle plot
        if i == 1:
            ax.set_ylabel('Density')
    
    return fig

def plot_transition_rates(data, program_id, figsize=(3, 3)):
    """
    Plot r01 and r10 as horizontal bars for all conditions.
    
    Args:
        data: Either a dictionary of MCMCInference objects keyed by 'condition-NMF#'
              or a DataFrame containing the MCMC results
        program_id: Program number to plot
    """
    # Create figure with square dimensions
    fig = plt.figure(figsize=figsize)
    
    # Create two axes side by side
    ax1 = fig.add_axes([0.15, 0.15, 0.65, 0.3])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.15, 0.55, 0.65, 0.3])  # [left, bottom, width, height]
    
    conditions = ['Ctrl', 'Aza', 'Dec', 'Vor']
    
    r01_values = []
    r10_values = []
    r01_errors_low = []
    r01_errors_high = []
    r10_errors_low = []
    r10_errors_high = []
    
    # Check if input is DataFrame or mcmc_dict
    is_dataframe = hasattr(data, 'columns')
    
    for cond in conditions:
        if is_dataframe:
            # Extract values from DataFrame
            row = data[(data['program_id'] == program_id) & (data['condition'] == cond)]
            if len(row) == 0:
                continue
                
            r01_mean = float(row['r01_mean'].values[0])
            r10_mean = float(row['r10_mean'].values[0])
            r01_low = float(row['r01_hdi_low'].values[0])
            r01_high = float(row['r01_hdi_high'].values[0])
            r10_low = float(row['r10_hdi_low'].values[0])
            r10_high = float(row['r10_hdi_high'].values[0])
        else:
            # Extract values from mcmc_dict
            key = f'{cond}-NMF{program_id}'
            if key not in data:
                continue
                
            mcmc = data[key]
            summary = az.summary(mcmc.trace)
            
            r01_mean = float(summary.loc['r01', 'mean'])
            r10_mean = float(summary.loc['r10', 'mean'])
            r01_low = float(summary.loc['r01', 'hdi_3%'])
            r01_high = float(summary.loc['r01', 'hdi_97%'])
            r10_low = float(summary.loc['r10', 'hdi_3%'])
            r10_high = float(summary.loc['r10', 'hdi_97%'])
        
        r01_values.append(r01_mean)
        r10_values.append(r10_mean)
        r01_errors_low.append(r01_mean - r01_low)
        r01_errors_high.append(r01_high - r01_mean)
        r10_errors_low.append(r10_mean - r10_low)
        r10_errors_high.append(r10_high - r10_mean)
    
    # Plot r01 values
    y_pos = np.arange(len(conditions))
    ax1.barh(y_pos, r01_values, 
            xerr=[r01_errors_low, r01_errors_high], 
            color='0.3',  # dark gray
            height=0.6, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(conditions)
    ax1.set_xlabel('$r_{01}$ (cell cycle$^{-1}$)')
    
    # Plot r10 values
    ax2.barh(y_pos, r10_values, 
            xerr=[r10_errors_low, r10_errors_high],
            color='0.3',  # dark gray
            height=0.6, capsize=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(conditions)
    ax2.set_xlabel('$r_{10}$ (cell cycle$^{-1}$)')
    
    # Set equal x limits for both plots and ensure tick at upper limit
    max_limit = max(
        max(r01_values) + max(r01_errors_high),
        max(r10_values) + max(r10_errors_high)
    ) * 1.1  # Add 10% margin
    
    # Round max_limit to a nice number and adjust ticks
    max_tick = np.ceil(max_limit * 10) / 10  # Round up to nearest 0.1
    ax1.set_xlim(0, max_tick)
    ax2.set_xlim(0, max_tick)
    
    # Set x-ticks at 0.1 intervals
    ticks = np.arange(0, max_tick + 0.05, 0.1)  # Add small offset to include max_tick
    
    ax1.set_xticks(ticks)
    ax2.set_xticks(ticks)
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # Add title
    fig.suptitle(f'Program {program_id}', y=0.95)
    
    return fig

def plot_relative_transition_rates(df, param='S', figsize=(8, 10), xscale='linear',xlim=None):
    """
    Create horizontal bar chart of relative rates compared to control.
    
    Args:
        df: DataFrame containing the MCMC results
        param: str, one of 'S', 'r01', 'r10', 'r' - parameter to plot
        figsize: tuple of figure dimensions (width, height)
        xscale: str, 'linear' or 'log' - scale for x-axis
    """
    # Validate parameter choice
    valid_params = {'S': 'total transition rate', 
                   'r01': 'off→on transition rate',
                   'r10': 'on→off transition rate',
                   'r': 'net transition rate',
                   '1/r01': 'off state persistence time',
                   '1/r10': 'on state persistence time'}
    if param not in valid_params:
        raise ValueError(f"param must be one of {list(valid_params.keys())}")
    
    # Define column names for mean and std based on parameter
    if param == '1/r01':
        mean_col = 'r01_mean'
        std_col = 'r01_std'
    elif param == '1/r10':
        mean_col = 'r10_mean'
        std_col = 'r10_std'
    else:
        mean_col = f'{param}_mean'
        std_col = f'{param}_std'
    
    # Set up colors for conditions
    colors = {
        'Aza': '#1f77b4',  # blue
        'Dec': '#2ca02c',  # green
        'Vor': '#d62728'   # red
    }
    
    # Initialize lists to store data
    programs = range(df['program_id'].max()+1)
    conditions = ['Aza', 'Dec', 'Vor']
    
    # Calculate relative rates and errors for each program and condition
    data = {cond: [] for cond in conditions}
    errors = {cond: [] for cond in conditions}
    
    for prog in programs:
        # Get control values
        ctrl_data = df[(df['program_id'] == prog) & (df['condition'] == 'Ctrl')]
        if len(ctrl_data) == 0:
            continue
        val_ctrl = ctrl_data[mean_col].values[0]
        sem_ctrl = ctrl_data[std_col].values[0]
        
        # Calculate relative rates for each condition
        for cond in conditions:
            cond_data = df[(df['program_id'] == prog) & (df['condition'] == cond)]
            if len(cond_data) == 0:
                continue
            val_cond = cond_data[mean_col].values[0]
            sem_cond = cond_data[std_col].values[0]
            

            # Calculate relative rate
            rel_rate = val_cond / val_ctrl
            if param == '1/r01':
                rel_rate = 1/rel_rate
            elif param == '1/r10':
                rel_rate = 1/rel_rate

            # Error propagation: sem_ratio^2 = (sem_cond/val_cond)^2 + (sem_ctrl/val_ctrl)^2
            rel_error = rel_rate * np.sqrt((sem_cond/val_cond)**2 + (sem_ctrl/val_ctrl)**2)
            
            data[cond].append(rel_rate)
            errors[cond].append(rel_error)
    #print(data)
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the width of each bar and positions of the bars
    bar_width = 0.25
    y_pos = np.arange(len(programs))
    

    for i, cond in enumerate(conditions):
        pos = y_pos + i * bar_width
        values = np.array(data[cond])
        
        if xscale == 'log':
            # Convert values to log10 so that 1 becomes 0
            log_vals = np.log10(values)
            left = np.where(log_vals >= 0, 0, log_vals)
            width = np.abs(log_vals)
            
            ax.barh(pos, width, bar_width, left=left, label=cond,
                    color=colors[cond], alpha=0.7)
            
            # Convert error bars to log-space
            err_log = np.array(errors[cond]) / (values * np.log(10))
            ax.errorbar(log_vals, pos, xerr=err_log, fmt='none',
                        color='black', capsize=3)
        else:
            ax.barh(pos, values, bar_width, label=cond,
                    color=colors[cond], alpha=0.7)
            ax.errorbar(values, pos, xerr=errors[cond], fmt='none',
                        color='black', capsize=3)


    if xscale == 'log':
        # Add a vertical line at 0 (log(1)) as the center reference
        ax.axvline(0, color='k', linewidth=1)

        # Get x-axis limits in linear scale
        if xlim is None:
            xlim = 10**np.array(ax.get_xlim())  # Convert current log-space limits to linear scale

        
        # Define tick positions in log-space, ensuring log(1) = 0 is included
        min_log, max_log = np.round(np.log10(xlim[0]),1), np.round(np.log10(xlim[1]),1)
        xtick_positions = np.log10(np.logspace(min_log, max_log, num=2))  # 3 ticks across range
        
        # Ensure 0 (log10(1)) is explicitly included
        if 0 not in xtick_positions:
            xtick_positions = np.concatenate((xtick_positions, [0]))
            xtick_positions = np.sort(xtick_positions)  # Keep order

        # Apply ticks and labels
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f'{tick:.2f}' for tick in xtick_positions])

        # Ensure correct axis limits in log-space
        ax.set_xlim(min_log, max_log)


    # # Customize the plot
    ax.set_yticks(y_pos + bar_width)
    ax.set_yticklabels([f'{i}' for i in programs])  # Remove reversed()
    ax.invert_yaxis()  # This will put 0 at the top
    ax.set_ylabel('NMF Program')
    if xscale == 'log':
        ax.set_xlabel(f'Relative {valid_params[param]} log10({param}/{param}_ctrl)')
    else:   
        ax.set_xlabel(f'Relative {valid_params[param]} ({param}/{param}_ctrl)')
    ax.set_ylim(len(programs)-0.2, -0.2)  # Keep these limits as is
    
   
    
    # Add a vertical line at x=1 (no change) AFTER setting the scale
    if xscale != 'log':
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_relative_transition_rates_heatmap(df, param='S', figsize=(8, 6), clim=None, show_annotations=False):
    """
    Create heatmap of relative rates compared to control, with programs sorted by mean relative rate.
    
    Args:
        df: DataFrame containing the MCMC results
        param: str, one of 'S', 'r01', 'r10', 'r', 'min_r', 'min_t', '1/S', 't1_frac' - parameter to plot
        figsize: tuple of figure dimensions (width, height)
        clim: tuple of (min, max) values for color scale, in log2 units. If None, will be symmetric around 0.
    """
    # Validate parameter choice
    valid_params = {'S': 'total transition rate', 
                   'r01': 'off→on transition rate',
                   'r10': 'on→off transition rate',
                   'r': 'net transition rate',
                   'min_r': 'minimum transition rate',
                   'min_t': 'maximum persistence time',
                   '1/r01': 'off state persistence time',
                   '1/r10': 'on state persistence time',
                   '1/S': 'inverse total transition rate',
                   't1_frac': 'ON state persistence time fraction'}
    if param not in valid_params:
        raise ValueError(f"param must be one of {list(valid_params.keys())}")
    
    # Define column name for mean based on parameter
    if param in ['1/r01', '1/r10']:
        mean_col = param.replace('1/', '') + '_mean'
    else:
        mean_col = f'{param}_mean'
    
    # Initialize lists to store data
    programs = range(df['program_id'].max()+1)
    conditions = ['Aza', 'Dec', 'Vor']
    
    # Create matrix to store relative rates
    rate_matrix = np.zeros((len(programs), len(conditions)))
    rate_matrix[:] = np.nan  # Fill with NaN initially
    
    # Calculate relative rates for each program and condition
    for i, prog in enumerate(programs):
        # Get control values
        ctrl_data = df[(df['program_id'] == prog) & (df['condition'] == 'Ctrl')]
        if len(ctrl_data) == 0:
            continue
            
        if param == 't1_frac':
            # Calculate t1/(t1+t0) = (1/r10)/(1/r10 + 1/r01) = r01/(r01 + r10)
            val_ctrl = (ctrl_data['r01_mean'].values[0] / 
                       (ctrl_data['r01_mean'].values[0] + ctrl_data['r10_mean'].values[0]))
        elif param == '1/S':
            val_ctrl = 1 / ctrl_data['S_mean'].values[0]
        elif param in ['min_r', 'min_t']:
            val_ctrl = min(ctrl_data['r01_mean'].values[0], ctrl_data['r10_mean'].values[0])
        else:
            val_ctrl = ctrl_data[mean_col].values[0]
        
        # Calculate relative rates for each condition
        for j, cond in enumerate(conditions):
            cond_data = df[(df['program_id'] == prog) & (df['condition'] == cond)]
            if len(cond_data) == 0:
                continue
                
            if param == 't1_frac':
                # Calculate t1/(t1+t0) for condition
                val_cond = (cond_data['r01_mean'].values[0] / 
                           (cond_data['r01_mean'].values[0] + cond_data['r10_mean'].values[0]))
                # For t1_frac, we want the absolute difference rather than the ratio
                rel_rate = val_cond - val_ctrl
                rate_matrix[i, j] = rel_rate  # Store directly, no log2 needed
                continue
            elif param == '1/S':
                val_cond = 1 / cond_data['S_mean'].values[0]
            elif param in ['min_r', 'min_t']:
                val_cond = min(cond_data['r01_mean'].values[0], cond_data['r10_mean'].values[0])
            else:
                val_cond = cond_data[mean_col].values[0]
            
            # Calculate relative rate
            rel_rate = val_cond / val_ctrl
            if param in ['1/r01', '1/r10', 'min_t']:
                rel_rate = 1/rel_rate
            
            # Store log2 of relative rate
            rate_matrix[i, j] = np.log2(rel_rate)
    
    # Sort programs based on mean relative rate
    program_means = np.nanmean(rate_matrix, axis=1)
    sort_indices = np.argsort(program_means)[::-1]  # Sort in descending order
    rate_matrix = rate_matrix[sort_indices]
    program_order = [str(i) for i in sort_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set color limits
    if clim is None:
        max_abs = np.nanmax(np.abs(rate_matrix))
        clim = (-max_abs, max_abs)
    
    # Create heatmap
    im = ax.imshow(rate_matrix, cmap='coolwarm', aspect='auto', clim=clim)
    
    # Add colorbar with appropriate label
    cbar = plt.colorbar(im)
    if param == 't1_frac':
        cbar.set_label('Δ(ON state persistence time fraction)')
    else:
        cbar.set_label(f'log2(relative {valid_params[param]})')
    
    # Customize the plot
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(program_order)  # Use sorted program order
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    
    ax.set_ylabel('NMF Program')
    ax.set_title(f'Relative {valid_params[param]}')
    
    # Add text annotations with actual values
    if show_annotations:
        for i in range(len(programs)):
            for j in range(len(conditions)):
                if not np.isnan(rate_matrix[i, j]):
                    if param == 't1_frac':
                        text = f'{rate_matrix[i, j]:.2f}'
                    else:
                        text = f'{2**rate_matrix[i, j]:.2f}'
                    # Choose text color based on background
                    color = 'white' if abs(rate_matrix[i, j]) > (clim[1] - clim[0])/4 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def calculate_gene_persistence_times(cont_results_df, H, conditions=['Ctrl', 'Aza', 'Dec', 'Vor'], debug=False):
    """
    Calculate persistence times and total transition rates for each gene in each condition.
    
    Args:
        cont_results_df: DataFrame containing MCMC results with r01 and r10 values
        H: DataFrame with gene loadings into programs (rows=genes, columns=programs)
        conditions: List of conditions to analyze
        debug: Boolean to enable debug output
    
    Returns:
        Dictionary containing three DataFrames:
        - 'r01': DataFrame with 1/r01 times (off→on transition)
        - 'r10': DataFrame with 1/r10 times (on→off transition)
        - '1/S': DataFrame with 1/(r01 + r10) values (inverse total transition rate)
    """
    persistence_times = {
        'r01': {},    # Will store 1/r01 times
        'r10': {},    # Will store 1/r10 times
        '1/S': {}     # Will store 1/(r01 + r10) values
    }
    
    # Debug output remains unchanged
    if debug:
        print("First 3 genes' loadings into components:")
        print(H.head(3))
        print("\nShape of H matrix:", H.shape)
    
    for cond in conditions:
        # Get r01 and r10 values for this condition
        cond_data = cont_results_df[cont_results_df['condition'] == cond]
        
        # Calculate persistence times and total transition rate for each program
        r01_values = cond_data['r01_mean'].values
        r10_values = cond_data['r10_mean'].values
        persistence_by_program = {
            'r01': 1 / r01_values,           # 1/r01 times
            'r10': 1 / r10_values,           # 1/r10 times
            '1/S': 1 / (r01_values + r10_values)  # 1/(r01 + r10) values
        }
        
        if debug and cond == conditions[0]:  # Only print for first condition
            print(f"\nPersistence times by program for {cond}:")
            for i in range(len(r01_values)):
                print(f"Program {i}:")
                print(f"  off→on time: {persistence_by_program['r01'][i]:.2f} days")
                print(f"  on→off time: {persistence_by_program['r10'][i]:.2f} days")
                print(f"  1/S time: {persistence_by_program['1/S'][i]:.2f} days")
        
        # Normalize H matrix first by column then by rows to sum to 1
        H_normalized = H.div(H.sum(axis=0), axis=1)
        H_normalized = H_normalized.div(H_normalized.sum(axis=1), axis=0)
        
        # Calculate weighted values for each gene
        for metric in ['r01', 'r10', '1/S']:
            gene_persistence = np.dot(H_normalized, persistence_by_program[metric])
            
            if debug and cond == conditions[0] and metric == 'r01':  # Only print for first condition
                print(f"\nFirst 3 genes calculation details ({metric}):")
                for i in range(3):
                    gene_name = H.index[i]
                    print(f"\nGene: {gene_name}")
                    print("Normalized loadings:", H_normalized.iloc[i].values)
                    print("Final persistence time:", gene_persistence[i])
            
            # Store results
            if cond not in persistence_times[metric]:
                persistence_times[metric][cond] = gene_persistence
    
    # Convert results to DataFrames
    result = {
        'r01': pd.DataFrame(persistence_times['r01'], index=H.index),
        'r10': pd.DataFrame(persistence_times['r10'], index=H.index),
        '1/S': pd.DataFrame(persistence_times['1/S'], index=H.index)
    }
    
    return result


def plot_persistence_vs_expression_density(deg_results_dict, gene_persistence_times_dict, drug,
                                        transition_type='1/S', figsize=(4, 4), H=None, 
                                        top_n_genes=None, annotate_genes=None, bins=50, 
                                        levels=8, color='red', min_logFC=0.1, smooth_sigma=10,
                                        show_sparse_points=True, sparse_threshold=0.1,
                                        annotate_outliers=False):
    """
    Plot changes in gene persistence time vs expression changes for one drug condition,
    using a density plot with isolines. Y-axis shows log2 ratio of maximum persistence times.
    Annotated genes are shown as individual points.
    
    Args:
        deg_results_dict: Dictionary containing differential expression results for each drug
        gene_persistence_times_dict: Dictionary containing DataFrames with persistence times
                                   for each gene and condition, with keys 'off_to_on' and 'on_to_off'
        drug: Drug condition to plot (e.g., 'Aza', 'Dec', or 'Vor')
        transition_type: Which transition times to plot ('off_to_on', 'on_to_off', or 'slowest')
        figsize: Tuple specifying figure size (width, height)
        H: DataFrame containing program loadings for each gene (rows=genes, columns=programs)
        top_n_genes: Number of top genes per program to include, optional
        annotate_genes: List of gene names to annotate in the plot
        bins: Number of bins for density calculation
        levels: Number of isoline levels to show
        color: Color for density plot ('red' or 'green')
        show_sparse_points: If True, show individual points in sparse regions
        sparse_threshold: Threshold for what constitutes a sparse region (fraction of max density)
    """
    if top_n_genes is not None and H is None:
        raise ValueError("H must be provided when top_n_genes is specified")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap
    colors = [(1, 1, 1), {'red': (1, 0, 0), 'green': (0, 0.6, 0)}[color]]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # If using top genes, get the set of genes to include
    included_genes = None
    if top_n_genes is not None:
        included_genes = set()
        for program in H.columns:
            top_genes = H.nlargest(top_n_genes, program).index
            included_genes.update(top_genes)
    
    # Get persistence times based on transition type
    if transition_type == 'slowest':
        # Get both transition DataFrames
        off_to_on_df = gene_persistence_times_dict['off_to_on']
        on_to_off_df = gene_persistence_times_dict['on_to_off']
        
        # Calculate persistence ratios for both transitions
        ctrl_persistence_off_to_on = off_to_on_df['Ctrl']
        drug_persistence_off_to_on = off_to_on_df[drug]
        ctrl_persistence_on_to_off = on_to_off_df['Ctrl']
        drug_persistence_on_to_off = on_to_off_df[drug]
        
        # Calculate log2 ratios for both transitions
        ratio_off_to_on = drug_persistence_off_to_on / ctrl_persistence_off_to_on
        ratio_on_to_off = drug_persistence_on_to_off / ctrl_persistence_on_to_off
        
        # Take the maximum ratio for each gene
        persistence_ratio = np.log2(pd.concat([ratio_off_to_on, ratio_on_to_off], axis=1).max(axis=1))
    else:
        # Use the specified transition type DataFrame
        gene_persistence_times_df = gene_persistence_times_dict[transition_type]
        ctrl_persistence = gene_persistence_times_df['Ctrl']
        drug_persistence = gene_persistence_times_df[drug]
        persistence_ratio = np.log2(drug_persistence / ctrl_persistence)
    
    # Create DataFrame combining changes
    plot_df = pd.DataFrame({
        'logFC': deg_results_dict[drug]['mean_logFC'] * np.log2(10),  # Convert from log10 to log2
        'log2_persistence_ratio': persistence_ratio,
        'FDR': deg_results_dict[drug]['combined_fdr']
    })
    
    # Remove rows with infinite or NaN values and filter for minimum fold change
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[abs(plot_df['logFC']) >= min_logFC]  # Filter for minimum fold change

    
    # Calculate extended y-axis limits with 5% padding
    y_min, y_max = plot_df['log2_persistence_ratio'].min(), plot_df['log2_persistence_ratio'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1
    y_limits = [y_min - y_padding, y_max + y_padding]
    
    # Define significant points
    significant = (plot_df['FDR'] < 0.05) & (abs(plot_df['logFC']) >= 1)
    
    # Create density plot
    x = plot_df['logFC']
    y = plot_df['log2_persistence_ratio']
    
    # Calculate the 2D histogram with extended y range
    h, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                      range=[[x.min(), x.max()], y_limits])
    h = h.T  # Transpose to match imshow convention
    
    # Calculate bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    
    # Create a smoothed version for contours with increased smoothing
    h_smooth = gaussian_filter(h, sigma=bins/smooth_sigma)
    
    # Plot density with white background
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(np.log1p(h_smooth), extent=extent, origin='lower', 
                   aspect='auto', cmap=custom_cmap)
    
    # Add contour lines, starting above zero
    if h_smooth.max() > 0:
        min_level = h_smooth.max() * 0.05
        contour_levels = np.linspace(min_level, h_smooth.max(), levels)
        cs = ax.contour(xcenters, ycenters, h_smooth, levels=contour_levels,
                        colors='black', alpha=0.3, linewidths=0.5)
    
    # Add points in sparse regions
    if show_sparse_points:
        # Create interpolation function for density
        density_interp = RegularGridInterpolator((ycenters, xcenters), h_smooth,
                                               bounds_error=False, fill_value=0)
        
        # Get density at each point
        points = np.column_stack([y, x])
        point_densities = density_interp(points)
        
        # Find points in sparse regions
        sparse_mask = point_densities < (h_smooth.max() * sparse_threshold)
        
        # Plot sparse points
        if np.any(sparse_mask):
            ax.scatter(x[sparse_mask], y[sparse_mask], 
                      c='black', s=1, alpha=0.2, zorder=1)
            if annotate_outliers:
                for gene in plot_df.index[sparse_mask]:
                    gene_x = plot_df.loc[gene, 'logFC']
                    gene_y = plot_df.loc[gene, 'log2_persistence_ratio']
                    ax.annotate(gene, (gene_x, gene_y),
                                fontsize=6,
                                textcoords='offset points',
                                xytext=(10,10),
                                ha='center',
                                va='center')
    
    # Add annotated genes as points
    if annotate_genes is not None:
        for gene in annotate_genes:
            if gene in plot_df.index:
                x = plot_df.loc[gene, 'logFC']
                y = plot_df.loc[gene, 'log2_persistence_ratio']
                
                # Plot point
                point_color = 'black'
                ax.scatter(x, y, c=point_color, s=50, alpha=1.0)
                
                # Add label
                display_name = gene.replace('_hg', '').replace('_mm', '')
                ax.annotate(display_name, 
                           (x, y),
                           xytext=(15, 15),  # Increased from (10, 10) to (20, 20)
                           textcoords='offset points',
                           fontsize=12,
                           alpha=1.0,
                           arrowprops=dict(
                               arrowstyle='-|>',
                               alpha=1.0,
                               color='black',
                               connectionstyle='arc3,rad=0.1',
                               shrinkB=5,  # Added to make arrow head smaller
                               mutation_scale=8  # Added to make arrow head smaller (default is 10)
                           ))
            else:
                print(f"Warning: Gene '{gene}' not found in data for {drug} condition")
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Set axis limits to match the extent of the density plot
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    
    # Update labels to reflect transition type
    ax.set_xlabel('Log2 Fold Change (Expression)')
    if transition_type == 'slowest':
        ylabel = 'Log2 Ratio of\nMaximum Persistence Time\n(Drug/Control)'
    elif transition_type == 'r01':
        ylabel = 'Log2 Ratio of\nOFF→ON Persistence Time\n(Drug/Control)'
    elif transition_type == 'r10':
        ylabel = 'Log2 Ratio of\nON→OFF Persistence Time\n(Drug/Control)'
    elif transition_type == '1/S':
        ylabel = 'Log2 Ratio of\nPersistence Time\n(Drug/Control)'

    ax.set_ylabel(ylabel)
    
    title = f'{drug} ({transition_type})'
    ax.set_title(title)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Density')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
