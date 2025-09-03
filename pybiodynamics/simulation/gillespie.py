"""
Gillespie simulation algorithm.

This module implements the Gillespie Stochastic Simulation Algorithm (SSA)
for modeling chemical reaction networks with discrete molecule counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from numba import njit
import time
import sympy as sp
from typing import Dict, Tuple, List, Callable


class GillespieSimulator:
    """
    A generic Gillespie simulator for stochastic chemical kinetics.

    This simulator can model any system of chemical reactions, provided a
    stoichiometric matrix and a function to calculate reaction propensities.

    Args:
        stoichiometric_matrix (np.ndarray): A matrix of shape (num_species, num_reactions)
            where S[i, j] is the change in the count of species i due to reaction j.
        propensity_func (callable): A Numba-jitted function that calculates reaction
            rates (propensities). It must have the signature:
            `propensity_func(species_counts: np.ndarray, rate_constants: np.ndarray) -> np.ndarray`.
        initial_state (np.ndarray): A 1D array of initial counts for each species.
        rate_constants (np.ndarray): A 1D array of rate constants passed to the propensity function.
        species_labels (list[str], optional): Labels for each species for plotting and stats.
            Defaults to None.

    Attributes:
        S (np.ndarray): The stoichiometric matrix.
        propensity_func (callable): The jitted propensity function.
        x0 (np.ndarray): The initial state vector.
        k (np.ndarray): The rate constants.
        labels (list[str]): The species labels.
        X (np.ndarray): Trace of component counts for each iteration.
        T (np.ndarray): Trace of the simulation time at each iteration.
        tsteps (np.ndarray): Duration spent in each state (time step `tau`).
    """

    def __init__(
        self,
        stoichiometric_matrix,
        propensity_func,
        initial_state,
        rate_constants,
        species_labels=None,
    ):
        self.S = stoichiometric_matrix
        self.propensity_func = propensity_func
        self.x0 = initial_state
        self.k = rate_constants

        num_species = self.S.shape[0]
        self.labels = (
            species_labels
            if species_labels is not None
            else [f"Species {i+1}" for i in range(num_species)]
        )

        # Results attributes, initialized to None
        self.X = None
        self.T = None
        self.tsteps = None

    @staticmethod
    def _gillespie_engine_py(x0, S, propensity_func, k, max_iter, seed):
        """
        Core Gillespie algorithm engine in pure Python/NumPy.

        Args:
            x0 (np.ndarray): Initial state vector.
            S (np.ndarray): Stoichiometric matrix.
            propensity_func (callable): Propensity function.
            k (np.ndarray): Rate constants.
            max_iter (int): Number of iterations.
            seed (int): RNG seed; use negative to skip seeding.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - X (np.ndarray): State history (num_species, max_iter).
                - T (np.ndarray): Time history (max_iter,).
                - tsteps (np.ndarray): Time step history (max_iter,).
        """
        # Initialization
        if seed >= 0:
            np.random.seed(seed)
        t = 0.0
        x = x0.copy()
        T = np.zeros(max_iter)
        tsteps = np.zeros(max_iter)
        X = np.zeros((S.shape[0], max_iter))

        # Simulation loop
        for i in range(max_iter):
            # 1. Calculate reaction propensities (rates)
            rates = propensity_func(x, k)
            sum_rates = np.sum(rates)

            if sum_rates == 0:  # No more reactions can occur
                # Fill remaining history with the last state and break
                for j in range(i, max_iter):
                    X[:, j] = x
                T[i:] = t
                tsteps[i:] = 0  # Or some indicator like np.inf
                break

            # 2. Determine WHEN the next state change occurs (time step `tau`)
            u = np.random.random()
            tau = -np.log(u) / sum_rates
            t += tau
            T[i] = t
            tsteps[i] = tau

            # 3. Determine WHICH reaction occurs
            # Select reaction j with probability rates[j] / sum_rates
            rand_val = np.random.random() * sum_rates
            reaction_index = 0
            cumulative_rate = rates[0]
            while cumulative_rate < rand_val:
                reaction_index += 1
                cumulative_rate += rates[reaction_index]

            # 4. Update the state
            x += S[:, reaction_index]
            X[:, i] = x

        return X, T, tsteps

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _gillespie_engine_numba(x0, S, propensity_func, k, max_iter, seed):
        """
        Core Gillespie algorithm engine, optimized with Numba.

        Args:
            x0 (np.ndarray): Initial state vector.
            S (np.ndarray): Stoichiometric matrix.
            propensity_func (callable): Jitted propensity function.
            k (np.ndarray): Rate constants.
            max_iter (int): Number of iterations.
            seed (int): RNG seed; use negative to skip seeding.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - X (np.ndarray): State history (num_species, max_iter).
                - T (np.ndarray): Time history (max_iter,).
                - tsteps (np.ndarray): Time step history (max_iter,).
        """
        # Initialization
        if seed >= 0:
            np.random.seed(seed)
        t = 0.0
        x = x0.copy()
        T = np.zeros(max_iter)
        tsteps = np.zeros(max_iter)
        X = np.zeros((S.shape[0], max_iter))

        # Simulation loop
        for i in range(max_iter):
            # 1. Calculate reaction propensities (rates)
            rates = propensity_func(x, k)
            sum_rates = np.sum(rates)

            if sum_rates == 0:  # No more reactions can occur
                # Fill remaining history with the last state and break
                for j in range(i, max_iter):
                    X[:, j] = x
                T[i:] = t
                tsteps[i:] = 0
                break

            # 2. Determine WHEN the next state change occurs (time step `tau`)
            u = np.random.random()
            tau = -np.log(u) / sum_rates
            t += tau
            T[i] = t
            tsteps[i] = tau

            # 3. Determine WHICH reaction occurs
            rand_val = np.random.random() * sum_rates
            reaction_index = 0
            cumulative_rate = rates[0]
            while cumulative_rate < rand_val:
                reaction_index += 1
                cumulative_rate += rates[reaction_index]

            # 4. Update the state
            x += S[:, reaction_index]
            X[:, i] = x

        return X, T, tsteps

    def run(self, max_iter=100_000, use_numba=True, seed=-1):
        """
        Run the Gillespie simulation.

        The results (X, T, tsteps) are stored as attributes of the instance.

        Args:
            max_iter (int): The number of iterations to simulate.
            use_numba (bool): Whether to use the Numba-optimized engine.
            seed (int): RNG seed; pass a non-negative integer for reproducibility.
        """
        if use_numba:
            self.X, self.T, self.tsteps = self._gillespie_engine_numba(
                self.x0, self.S, self.propensity_func, self.k, max_iter, seed
            )
        else:
            self.X, self.T, self.tsteps = self._gillespie_engine_py(
                self.x0, self.S, self.propensity_func, self.k, max_iter, seed
            )

    def get_stats(self):
        """
        Calculate statistics for the most recent Gillespie simulation.

        Returns:
            pd.DataFrame: A DataFrame containing the time-weighted mean and
            variance for each species.
        """
        if self.X is None:
            raise RuntimeError("Simulation has not been run yet. Call .run() first.")

        total_time = self.tsteps.sum()
        # Calculate time-weighted means
        tw_means = (self.X * self.tsteps).sum(axis=1) / total_time

        # Calculate time-weighted variances
        residuals = self.X - tw_means[:, np.newaxis]
        tw_vars = (self.tsteps * residuals**2).sum(axis=1) / total_time

        stats_df = pd.DataFrame(
            {
                "Component": self.labels,
                "Gillespie Mean": tw_means,
                "Gillespie Variance": tw_vars,
            }
        )
        return stats_df

    def plot_trace(self, title="Gillespie Simulation Trace"):
        """Plot the time series trace for each species."""
        if self.T is None:
            raise RuntimeError("Simulation has not been run yet. Call .run() first.")

        plt.figure(figsize=(12, 7))
        for i, label in enumerate(self.labels):
            plt.plot(self.T, self.X[i, :], label=label)

        plt.xlabel("Time")
        plt.ylabel("Number of Molecules")
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()


def _create_propensity_function(symbolic_propensities: sp.Matrix, system_model) -> callable:
    """
    Create a callable propensity function from symbolic expressions that can
    be JIT-compiled by Numba.
    
    This function first uses SymPy's lambdify to convert the symbolic math
    into a single Python function, then defines a simple wrapper to match the
    signature required by the Gillespie simulator (func(x, k)), which is
    then suitable for Numba compilation.
    
    Args:
        symbolic_propensities (sp.Matrix): Vector of symbolic propensity expressions.
        system_model: The system model for context.
        
    Returns:
        callable: A Python function with signature f(x, k) -> np.ndarray, ready for Numba.
    """
    # Ensure a consistent order for species and parameters
    ordered_species = list(system_model.species.values())
    ordered_params = list(system_model.parameters.values())
    
    # Create symbols for the discrete species counts and parameters.
    # These must match the symbols used in the symbolic_propensities expressions.
    species_symbols = [sp.Symbol(s.name, positive=True) for s in ordered_species]
    param_symbols = [p.get_symbol() for p in ordered_params]
    
    # Lambdify requires a flat list of argument symbols
    all_symbols = species_symbols + param_symbols
    propensity_expressions = list(symbolic_propensities)
    
    # 1. Create a single numerical function from all symbolic expressions.
    #    This function will expect all arguments to be passed individually,
    #    e.g., func(species1, species2, ..., param1, param2, ...).
    lambdified_func = sp.lambdify([species_symbols, param_symbols], symbolic_propensities, 'numpy')
    
    # 2. Create a simple wrapper to adapt the signature.
    #    The Gillespie engine provides arguments as two arrays: x (species) and k (params).
    #    This wrapper unpacks these arrays to call the lambdified function.
    #    This type of wrapper IS JIT-compatible with Numba.

    return lambdified_func


def create_gillespie_from_system_model(system_model) -> dict:
    """
    Convert a SystemModel into components ready for Gillespie simulation.
    
    Args:
        system_model: The system model with reactions and species
        
    Returns:
        dict: Dictionary containing all components needed for GillespieSimulator
    """
    # Generate the stoichiometric matrix
    S = system_model.generate_stoichiometric_matrix()
    
    # Generate the symbolic propensity vector
    symbolic_propensities = system_model.generate_propensity_vector()
    
    # Convert symbolic propensities to a callable function
    propensity_func = _create_propensity_function(symbolic_propensities, system_model)
    
    # Get initial state and rate constants
    initial_state = np.array([int(s.initial_condition) for s in system_model.species.values()])
    rate_constants = np.array([p.get_default_value() for p in system_model.parameters.values()])
    
    # Get species labels
    species_labels = [s.name for s in system_model.species.values()]
    
    return {
        'stoichiometric_matrix': S,
        'propensity_func': propensity_func,
        'initial_state': initial_state,
        'rate_constants': rate_constants,
        'species_labels': species_labels,
        'symbolic_propensities': symbolic_propensities  # Keep for inspection
    }


def run_gillespie_simulation(system_model, max_iter: int = 100_000, 
                           use_numba: bool = True, seed: int = -1, plot: bool = True):
    """
    Run a Gillespie simulation using a SystemModel.
    
    Args:
        system_model: The system model to simulate
        max_iter (int): Number of simulation iterations
        use_numba (bool): Whether to use Numba optimization
        seed (int): Random seed for reproducibility
        plot (bool): Whether to plot the results
        
    Returns:
        dict: Results containing the simulator and statistics
    """
    # Convert system model to Gillespie components
    gillespie_components = create_gillespie_from_system_model(system_model)

    if use_numba:
        gillespie_components['propensity_func'] = njit(gillespie_components['propensity_func'])

    # Create the Gillespie simulator
    simulator = GillespieSimulator(
        stoichiometric_matrix=gillespie_components['stoichiometric_matrix'],
        propensity_func=gillespie_components['propensity_func'],
        initial_state=gillespie_components['initial_state'],
        rate_constants=gillespie_components['rate_constants'],
        species_labels=gillespie_components['species_labels']
    )
        
    # Run the simulation
    print(f"Running Gillespie simulation for {max_iter:,} iterations...")
    start_time = time.time()
    simulator.run(max_iter=max_iter, use_numba=use_numba, seed=seed)
    end_time = time.time()
    
    elapsed_ms = (end_time - start_time) * 1000
    print(f"Simulation completed in {elapsed_ms:.2f} ms")
    
    # Get statistics
    stats_df = simulator.get_stats()
    print("\nSimulation Statistics:")
    print(tabulate(stats_df, headers="keys", showindex=False, tablefmt="psql"))
    
    # Plot if requested
    if plot:
        simulator.plot_trace(title=f"Gillespie Simulation: {system_model.name}")
    
    return {
        'simulator': simulator,
        'stats_df': stats_df,
        'elapsed_ms': elapsed_ms,
        'gillespie_components': gillespie_components
    }
