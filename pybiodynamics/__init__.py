"""
PyBiodynamics: A Python library for modeling and simulating biological dynamic systems.

This library provides tools for:
- Defining biological species and parameters
- Creating reaction networks with various kinetics
- Simulating systems using deterministic (ODE) and stochastic (Gillespie) methods
- Visualizing simulation results

Main classes:
    Species: Represents a biological species with initial conditions
    Parameter: Represents model parameters with default values
    SystemModel: Container for complete biological systems
    
Reaction types:
    MassActionReaction: Standard mass-action kinetics
    LogisticGrowthReaction: Logistic growth dynamics
    
Simulators:
    GillespieSimulator: Stochastic simulation algorithm
    simulate_ode: Deterministic ODE simulation
"""

from .core.models import Species, Parameter, SystemModel
from .core.reactions import Reaction, MassActionReaction, LogisticGrowthReaction
from .simulation.gillespie import GillespieSimulator, run_gillespie_simulation
from .simulation.ode import simulate_ode
from .visualization.plotting import plot_simulation_results, plot_ode_solution, plot_gillespie_trace

__version__ = "0.1.0"
__author__ = "PyBiodynamics Team"
__email__ = "contact@pybiodynamics.org"

__all__ = [
    # Core classes
    "Species",
    "Parameter", 
    "SystemModel",
    
    # Reaction types
    "Reaction",
    "MassActionReaction",
    "LogisticGrowthReaction",
    
    # Simulation
    "GillespieSimulator",
    "run_gillespie_simulation",
    "simulate_ode",
    
    # Visualization
    "plot_simulation_results",
    "plot_ode_solution",
    "plot_gillespie_trace",
]
