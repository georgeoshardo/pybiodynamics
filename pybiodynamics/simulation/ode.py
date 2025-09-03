"""
ODE simulation utilities.

This module provides deterministic simulation capabilities using
ordinary differential equation (ODE) solvers from SciPy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple, Any


def simulate_ode(system_model, t_span: Tuple[float, float], t_eval: Optional[np.ndarray] = None, 
                method: str = 'LSODA', **kwargs) -> Dict[str, Any]:
    """
    Simulate a SystemModel using ODE integration.
    
    Args:
        system_model: The SystemModel to simulate
        t_span (Tuple[float, float]): Time span as (t_start, t_end)
        t_eval (np.ndarray, optional): Specific time points to evaluate. 
            If None, uses 1000 evenly spaced points.
        method (str): Integration method for solve_ivp. Default is 'LSODA'.
        **kwargs: Additional keyword arguments passed to solve_ivp
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'solution': The scipy.integrate.OdeResult object
            - 'sim_data': The lambdified ODE data from the system model
            - 'success': Boolean indicating if integration was successful
    """
    # Generate the lambdified ODE system
    sim_data = system_model.lambdify_odes()
    
    # Set default time evaluation points if not provided
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    # Run the ODE simulation
    solution = solve_ivp(
        fun=sim_data['func'],
        t_span=t_span,
        y0=sim_data['y0'],
        args=sim_data['params'],
        t_eval=t_eval,
        method=method,
        dense_output=True,
        **kwargs
    )
    
    return {
        'solution': solution,
        'sim_data': sim_data,
        'success': solution.success
    }


def plot_ode_solution(result: Dict[str, Any], title: Optional[str] = None, 
                     figsize: Tuple[float, float] = (10, 6), style: str = 'seaborn-v0_8-whitegrid'):
    """
    Plot the results of an ODE simulation.
    
    Args:
        result (Dict[str, Any]): Result dictionary from simulate_ode
        title (str, optional): Plot title. If None, generates automatic title.
        figsize (Tuple[float, float]): Figure size as (width, height)
        style (str): Matplotlib style to use
    """
    if not result['success']:
        print("Warning: ODE integration was not successful")
        return
        
    solution = result['solution']
    sim_data = result['sim_data']
    
    # Set plot style
    plt.style.use(style)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot each species
    for i, name in enumerate(sim_data['species_names']):
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


def compare_ode_gillespie(system_model, t_span: Tuple[float, float], 
                         gillespie_results: Dict[str, Any], 
                         t_eval: Optional[np.ndarray] = None,
                         figsize: Tuple[float, float] = (12, 8)):
    """
    Compare ODE and Gillespie simulation results side by side.
    
    Args:
        system_model: The SystemModel that was simulated
        t_span (Tuple[float, float]): Time span for ODE simulation
        gillespie_results (Dict[str, Any]): Results from run_gillespie_simulation
        t_eval (np.ndarray, optional): Time points for ODE evaluation
        figsize (Tuple[float, float]): Figure size
    """
    # Run ODE simulation
    ode_result = simulate_ode(system_model, t_span, t_eval)
    
    if not ode_result['success']:
        print("Error: ODE simulation failed")
        return
    
    # Extract data
    ode_solution = ode_result['solution']
    ode_sim_data = ode_result['sim_data']
    
    gillespie_sim = gillespie_results['simulator']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ODE results
    for i, name in enumerate(ode_sim_data['species_names']):
        ax1.plot(ode_solution.t, ode_solution.y[i], label=f"{name}", lw=2.5)
    
    ax1.set_title("Deterministic (ODE)", fontsize=14)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Concentration/Population", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Plot Gillespie results
    for i, label in enumerate(gillespie_sim.labels):
        ax2.plot(gillespie_sim.T, gillespie_sim.X[i, :], label=label, alpha=0.8)
    
    ax2.set_title("Stochastic (Gillespie)", fontsize=14)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Number of Molecules", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def parameter_sweep(system_model, parameter_name: str, parameter_values: List[float],
                   t_span: Tuple[float, float], t_eval: Optional[np.ndarray] = None,
                   plot_species: Optional[List[str]] = None, 
                   figsize: Tuple[float, float] = (10, 6)):
    """
    Perform a parameter sweep for ODE simulations.
    
    Args:
        system_model: The SystemModel to simulate
        parameter_name (str): Name of the parameter to sweep
        parameter_values (List[float]): Values to test for the parameter
        t_span (Tuple[float, float]): Time span for simulations
        t_eval (np.ndarray, optional): Time points for evaluation
        plot_species (List[str], optional): Species to plot. If None, plots all.
        figsize (Tuple[float, float]): Figure size
        
    Returns:
        Dict[str, Any]: Dictionary containing results for each parameter value
    """
    if parameter_name not in system_model.parameters:
        raise ValueError(f"Parameter '{parameter_name}' not found in model")
    
    # Store original parameter value
    original_param = system_model.parameters[parameter_name]
    original_value = original_param.default_value
    
    results = {}
    
    # Set default time evaluation points if not provided
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    # Run simulations for each parameter value
    for param_val in parameter_values:
        # Temporarily change parameter value
        system_model.parameters[parameter_name].default_value = param_val
        
        # Run simulation
        result = simulate_ode(system_model, t_span, t_eval)
        results[param_val] = result
    
    # Restore original parameter value
    system_model.parameters[parameter_name].default_value = original_value
    
    # Plot results
    plt.figure(figsize=figsize)
    
    # Determine which species to plot
    sim_data = list(results.values())[0]['sim_data']
    species_to_plot = plot_species if plot_species else sim_data['species_names']
    
    for param_val in parameter_values:
        result = results[param_val]
        if result['success']:
            solution = result['solution']
            for i, species_name in enumerate(sim_data['species_names']):
                if species_name in species_to_plot:
                    plt.plot(solution.t, solution.y[i], 
                           label=f"{species_name} ({parameter_name}={param_val})", 
                           lw=2)
    
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Concentration/Population", fontsize=14)
    plt.title(f"Parameter Sweep: {parameter_name}", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    return results
