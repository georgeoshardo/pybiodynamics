"""
Plotting utilities.

This module provides comprehensive visualization tools for biological simulation results,
including time series plots, statistical comparisons, and phase space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings


def plot_simulation_results(results: Dict[str, Any], result_type: str = 'auto',
                          title: Optional[str] = None, 
                          figsize: Tuple[float, float] = (10, 6),
                          style: str = 'seaborn-v0_8-whitegrid',
                          save_path: Optional[str] = None):
    """
    General plotting function that automatically detects and plots simulation results.
    
    Args:
        results (Dict[str, Any]): Results from simulate_ode or run_gillespie_simulation
        result_type (str): Type of results ('ode', 'gillespie', or 'auto' for auto-detection)
        title (str, optional): Plot title. If None, generates automatic title.
        figsize (Tuple[float, float]): Figure size as (width, height)
        style (str): Matplotlib style to use
        save_path (str, optional): Path to save the plot. If None, displays interactively.
    """
    plt.style.use(style)
    
    # Auto-detect result type if needed
    if result_type == 'auto':
        if 'solution' in results:
            result_type = 'ode'
        elif 'simulator' in results:
            result_type = 'gillespie'
        else:
            raise ValueError("Cannot auto-detect result type. Please specify 'ode' or 'gillespie'.")
    
    if result_type == 'ode':
        plot_ode_solution(results, title, figsize, style)
    elif result_type == 'gillespie':
        plot_gillespie_trace(results, title, figsize, style)
    else:
        raise ValueError("result_type must be 'ode', 'gillespie', or 'auto'")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")


def plot_ode_solution(result: Dict[str, Any], title: Optional[str] = None, 
                     figsize: Tuple[float, float] = (10, 6), 
                     style: str = 'seaborn-v0_8-whitegrid',
                     species_subset: Optional[List[str]] = None):
    """
    Plot the results of an ODE simulation.
    
    Args:
        result (Dict[str, Any]): Result dictionary from simulate_ode
        title (str, optional): Plot title. If None, generates automatic title.
        figsize (Tuple[float, float]): Figure size as (width, height)
        style (str): Matplotlib style to use
        species_subset (List[str], optional): Subset of species to plot
    """
    if not result['success']:
        warnings.warn("ODE integration was not successful")
        return
        
    solution = result['solution']
    sim_data = result['sim_data']
    
    # Set plot style
    plt.style.use(style)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Determine which species to plot
    species_names = sim_data['species_names']
    if species_subset:
        species_indices = [i for i, name in enumerate(species_names) if name in species_subset]
        species_names = [species_names[i] for i in species_indices]
    else:
        species_indices = range(len(species_names))

    # Plot each species
    for i, name in zip(species_indices, species_names):
        ax.plot(solution.t, solution.y[i], label=f"{name}", lw=2.5)

    # Create dynamic title with parameter values if not provided
    if title is None:
        param_str = ", ".join(f"{name}={val}" for name, val in 
                             zip(sim_data['param_names'], sim_data['params']))
        title = f"ODE Simulation\nParameters: {param_str}"
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Concentration/Population", fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_gillespie_trace(results: Dict[str, Any], title: Optional[str] = None,
                        figsize: Tuple[float, float] = (12, 7),
                        style: str = 'seaborn-v0_8-whitegrid',
                        species_subset: Optional[List[str]] = None,
                        alpha: float = 0.8):
    """
    Plot the trace of a Gillespie simulation.
    
    Args:
        results (Dict[str, Any]): Results from run_gillespie_simulation
        title (str, optional): Plot title
        figsize (Tuple[float, float]): Figure size
        style (str): Matplotlib style
        species_subset (List[str], optional): Subset of species to plot
        alpha (float): Line transparency
    """
    simulator = results['simulator']
    
    if simulator.T is None:
        raise RuntimeError("Simulation has not been run yet.")

    plt.style.use(style)
    plt.figure(figsize=figsize)
    
    # Determine which species to plot
    if species_subset:
        species_indices = [i for i, label in enumerate(simulator.labels) if label in species_subset]
        labels_to_plot = [simulator.labels[i] for i in species_indices]
    else:
        species_indices = range(len(simulator.labels))
        labels_to_plot = simulator.labels

    for i, label in zip(species_indices, labels_to_plot):
        plt.plot(simulator.T, simulator.X[i, :], label=label, alpha=alpha)

    if title is None:
        title = "Gillespie Simulation Trace"
    
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Number of Molecules", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_phase_space(result: Dict[str, Any], species_x: str, species_y: str,
                    title: Optional[str] = None, figsize: Tuple[float, float] = (8, 8),
                    style: str = 'seaborn-v0_8-whitegrid'):
    """
    Create a phase space plot for two species.
    
    Args:
        result (Dict[str, Any]): Result from simulate_ode or run_gillespie_simulation
        species_x (str): Name of species for x-axis
        species_y (str): Name of species for y-axis
        title (str, optional): Plot title
        figsize (Tuple[float, float]): Figure size
        style (str): Matplotlib style
    """
    plt.style.use(style)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Handle different result types
    if 'solution' in result:  # ODE results
        sim_data = result['sim_data']
        solution = result['solution']
        species_names = sim_data['species_names']
        
        try:
            x_idx = species_names.index(species_x)
            y_idx = species_names.index(species_y)
        except ValueError as e:
            raise ValueError(f"Species not found: {e}")
        
        x_data = solution.y[x_idx]
        y_data = solution.y[y_idx]
        
    elif 'simulator' in result:  # Gillespie results
        simulator = result['simulator']
        labels = simulator.labels
        
        try:
            x_idx = labels.index(species_x)
            y_idx = labels.index(species_y)
        except ValueError as e:
            raise ValueError(f"Species not found: {e}")
        
        x_data = simulator.X[x_idx, :]
        y_data = simulator.X[y_idx, :]
    else:
        raise ValueError("Invalid result format")
    
    # Create phase space plot
    ax.plot(x_data, y_data, 'b-', alpha=0.7, lw=1)
    ax.plot(x_data[0], y_data[0], 'go', markersize=8, label='Start')
    ax.plot(x_data[-1], y_data[-1], 'ro', markersize=8, label='End')
    
    if title is None:
        title = f"Phase Space: {species_x} vs {species_y}"
    
    ax.set_xlabel(f"{species_x}", fontsize=14)
    ax.set_ylabel(f"{species_y}", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_statistical_comparison(ode_result: Dict[str, Any], gillespie_results: Dict[str, Any],
                               figsize: Tuple[float, float] = (12, 8)):
    """
    Create a statistical comparison between ODE and Gillespie results.
    
    Args:
        ode_result (Dict[str, Any]): Results from simulate_ode
        gillespie_results (Dict[str, Any]): Results from run_gillespie_simulation
        figsize (Tuple[float, float]): Figure size
    """
    # Get statistics from Gillespie
    stats_df = gillespie_results['stats_df']
    
    # Get final values from ODE
    ode_solution = ode_result['solution']
    ode_sim_data = ode_result['sim_data']
    ode_final_values = ode_solution.y[:, -1]
    
    # Create comparison DataFrame
    comparison_data = []
    for i, species_name in enumerate(ode_sim_data['species_names']):
        gillespie_row = stats_df[stats_df['Component'] == species_name]
        if not gillespie_row.empty:
            comparison_data.append({
                'Species': species_name,
                'ODE_Final': ode_final_values[i],
                'Gillespie_Mean': gillespie_row['Gillespie Mean'].iloc[0],
                'Gillespie_Std': np.sqrt(gillespie_row['Gillespie Variance'].iloc[0])
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create bar plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Mean comparison
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    ax1.bar(x_pos - width/2, comparison_df['ODE_Final'], width, 
           label='ODE Final', alpha=0.8)
    ax1.bar(x_pos + width/2, comparison_df['Gillespie_Mean'], width,
           yerr=comparison_df['Gillespie_Std'], label='Gillespie Mean ± Std', 
           alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Species', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('ODE vs Gillespie Comparison', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comparison_df['Species'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Coefficient of variation
    cv = comparison_df['Gillespie_Std'] / comparison_df['Gillespie_Mean']
    ax2.bar(x_pos, cv, alpha=0.8, color='orange')
    ax2.set_xlabel('Species', fontsize=12)
    ax2.set_ylabel('Coefficient of Variation', fontsize=12)
    ax2.set_title('Gillespie Noise Level', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(comparison_df['Species'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df


def plot_parameter_sensitivity(results_dict: Dict[float, Dict[str, Any]], 
                              parameter_name: str, species_name: str,
                              metric: str = 'final_value',
                              figsize: Tuple[float, float] = (10, 6)):
    """
    Plot parameter sensitivity analysis results.
    
    Args:
        results_dict (Dict[float, Dict[str, Any]]): Dictionary mapping parameter values to results
        parameter_name (str): Name of the parameter being swept
        species_name (str): Name of the species to analyze
        metric (str): Metric to plot ('final_value', 'max_value', 'mean_value')
        figsize (Tuple[float, float]): Figure size
    """
    param_values = []
    metric_values = []
    
    for param_val, result in results_dict.items():
        if not result['success']:
            continue
            
        solution = result['solution']
        sim_data = result['sim_data']
        
        try:
            species_idx = sim_data['species_names'].index(species_name)
        except ValueError:
            raise ValueError(f"Species '{species_name}' not found")
        
        species_data = solution.y[species_idx]
        
        if metric == 'final_value':
            metric_value = species_data[-1]
        elif metric == 'max_value':
            metric_value = np.max(species_data)
        elif metric == 'mean_value':
            metric_value = np.mean(species_data)
        else:
            raise ValueError("metric must be 'final_value', 'max_value', or 'mean_value'")
        
        param_values.append(param_val)
        metric_values.append(metric_value)
    
    # Sort by parameter value
    sorted_data = sorted(zip(param_values, metric_values))
    param_values, metric_values = zip(*sorted_data)
    
    plt.figure(figsize=figsize)
    plt.plot(param_values, metric_values, 'o-', linewidth=2, markersize=6)
    plt.xlabel(f"{parameter_name}", fontsize=14)
    plt.ylabel(f"{species_name} ({metric.replace('_', ' ').title()})", fontsize=14)
    plt.title(f"Parameter Sensitivity: {parameter_name} vs {species_name}", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
