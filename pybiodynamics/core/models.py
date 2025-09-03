"""
Core model classes

This module contains the fundamental building blocks for biological modeling:
- Species: Represents biological entities (molecules, populations, etc.)
- Parameter: Represents model parameters with values and metadata  
- SystemModel: Container class for complete biological systems
"""

import sympy as sp
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Callable
from sympy import lambdify
from sympy.core.symbol import Symbol

# Define global time symbol
t = sp.Symbol('t', positive=True)


class Parameter:
    """
    A parameter class that allows defining a symbol with optional default value
    and optional list of parameter values for parameter sweeps.
    """
    
    def __init__(self, name: str, default_value: float = None, values: List[float] = None, **kwargs):
        """
        Initialize a Parameter.
        
        Args:
            name (str): The parameter name
            default_value (float, optional): Default value for the parameter
            values (List[float], optional): List of values for parameter sweeps
            **kwargs: Additional arguments passed to sympy.Symbol
        """
        self.name = name
        self.symbol = sp.Symbol(name, **kwargs)
        self.default_value = default_value
        self.values = values if values is not None else []
        
    def get_symbol(self) -> Symbol: 
        """Get the symbolic representation of the parameter."""
        return self.symbol

    def get_default_value(self) -> float:
        """Get the default value, raising an error if not set."""
        if self.default_value is None: 
            raise ValueError(f"No default value for '{self.name}'")
        return self.default_value

    def __repr__(self) -> str: 
        return f"Parameter('{self.name}', default={self.default_value})"

    def __mul__(self, other):
        """Multiply parameters or parameter by scalar."""
        if isinstance(other, Parameter):
            name = f"{self.name}*{other.name}"
            symbol = self.symbol * other.symbol
            default_value = self.default_value * other.default_value
            return Parameter(name, default_value=default_value, symbol=symbol)
        elif isinstance(other, (int, float)):
            name = f"{self.name}*{other}"
            symbol = self.symbol
            default_value = self.default_value * other
            return Parameter(name, default_value=default_value, symbol=symbol)
        else:
            raise ValueError(f"Unsupported operand type for *: {type(other)}")

    def __rmul__(self, other):
        """Right multiplication (scalar * parameter)."""
        return self * other

    def __truediv__(self, other):
        """Divide parameters or parameter by scalar."""
        if isinstance(other, Parameter):
            name = f"{self.name}/{other.name}"
            symbol = self.symbol / other.symbol
            default_value = self.default_value / other.default_value
            return Parameter(name, default_value=default_value, symbol=symbol)
        elif isinstance(other, (int, float)):
            name = f"{self.name}/{other}"
            symbol = self.symbol
            default_value = self.default_value / other
            return Parameter(name, default_value=default_value, symbol=symbol)
        else:
            raise ValueError(f"Unsupported operand type for /: {type(other)}")


class Species:
    """
    Represents a species in a model, holding its symbolic representation
    and an initial condition.
    """
    
    def __init__(self, name: str, initial_condition: float = 0.0, **kwargs):
        """
        Initialize a Species.
        
        Args:
            name (str): The name of the species (e.g., 'x', 'glucose')
            initial_condition (float, optional): The starting value for simulations. Defaults to 0.0
            **kwargs: Additional keyword arguments passed to sympy.Function (e.g., positive=True)
        """
        self.name = name
        # Represent the species as a function of time, e.g., x(t), which is ideal for ODEs
        self.symbol = sp.Function(name, **kwargs)(t)
        self.initial_condition = initial_condition

    def __repr__(self) -> str:
        """Provides a clear string representation of the Species object."""
        return f"Species('{self.name}', initial_condition={self.initial_condition})"

    # --- Methods for NetworkX compatibility ---

    def __hash__(self):
        """Allows the object to be used as a key in a dictionary or a node in a graph."""
        return hash(self.name)

    def __eq__(self, other):
        """Defines equality based on the unique species name."""
        if not isinstance(other, Species):
            return NotImplemented
        return self.name == other.name


class SystemModel:
    """
    A class to build and represent a system of interacting species using a network graph.
    """
    
    def __init__(self, name: str):
        """
        Initialize a SystemModel.
        
        Args:
            name (str): Name of the biological system
        """
        self.name = name
        self.graph = nx.DiGraph()
        self.species: Dict[str, Species] = {}
        self.parameters: Dict[str, Parameter] = {}
        self.reactions: List = []  # Will contain Reaction objects

    def add_species(self, species: Species):
        """
        Add a species to the model, creating a node in the graph.
        
        Args:
            species (Species): The species to add
            
        Returns:
            SystemModel: Self for method chaining
        """
        if species.name in self.species:
            raise ValueError(f"Species '{species.name}' already exists in the model.")
        self.species[species.name] = species
        self.graph.add_node(species, label=species.name)
        return self

    def add_parameter(self, parameter: Parameter):
        """
        Add a parameter to the model.
        
        Args:
            parameter (Parameter): The parameter to add
            
        Returns:
            SystemModel: Self for method chaining
        """
        if parameter.name in self.parameters:
            raise ValueError(f"Parameter '{parameter.name}' already exists in the model.")
        self.parameters[parameter.name] = parameter
        return self

    def add_reaction(self, reaction):
        """
        Add a reaction to the model, creating edges in the graph.
        An edge from A to B means that reactant A is involved in a reaction that produces B.
        
        Args:
            reaction: The reaction to add (must have reactants, products, and name attributes)
            
        Returns:
            SystemModel: Self for method chaining
        """
        self.reactions.append(reaction)
        # Add edges to the graph based on the reaction
        # An edge (u, v) means species u influences the abundance of species v
        for reactant in reaction.reactants:
            for product in reaction.products:
                # Add a directed edge from reactant to product
                if self.graph.has_edge(reactant, product):
                    # Append reaction name to existing edge attribute
                    self.graph.edges[reactant, product]['reactions'].append(reaction.name)
                else:
                    self.graph.add_edge(reactant, product, reactions=[reaction.name])
        return self

    def generate_rate_vector(self) -> sp.Matrix:
        """
        Generate the symbolic rate vector where each element is the 
        rate law for a reaction.
        
        Returns:
            sp.Matrix: Vector of rate laws
        """
        rate_laws = [rxn.rate_law for rxn in self.reactions]
        return sp.Matrix(rate_laws)

    def generate_stoichiometric_matrix(self) -> np.ndarray:
        """
        Generate the stoichiometric matrix S where S[i,j] is the change in 
        species i due to reaction j.
        
        Returns:
            np.ndarray: Matrix of shape (num_species, num_reactions)
        """
        num_species = len(self.species)
        num_reactions = len(self.reactions)
        
        # Create ordered lists for consistent indexing
        ordered_species = list(self.species.values())
        
        S = np.zeros((num_species, num_reactions), dtype=int)
        
        for j, reaction in enumerate(self.reactions):
            # Species consumed (negative stoichiometry)
            for species, stoich in reaction.reactants.items():
                species_idx = ordered_species.index(species)
                S[species_idx, j] -= stoich
            
            # Species produced (positive stoichiometry)
            for species, stoich in reaction.products.items():
                species_idx = ordered_species.index(species)
                S[species_idx, j] += stoich
        
        return S

    def generate_odes(self) -> Dict[Symbol, sp.Expr]:
        """
        Generate the system of Ordinary Differential Equations (ODEs) 
        by multiplying the stoichiometric matrix S by the rate vector v.
        
        Returns:
            Dict[Symbol, sp.Expr]: Mapping from species symbols to ODE expressions
        """
        # 1. Get the components
        S = sp.Matrix(self.generate_stoichiometric_matrix()) # Convert numpy array to sympy Matrix
        v = self.generate_rate_vector()
        
        # 2. Perform the core calculation: S * v
        # This results in a column vector where each element is the RHS of an ODE
        dxdt_vector = S * v
        
        # 3. Map the results back to the species symbols
        ordered_species_symbols = [s.symbol for s in self.species.values()]
        odes = {symbol: expr for symbol, expr in zip(ordered_species_symbols, dxdt_vector)}
        
        return odes

    def lambdify_odes(self) -> Dict:
        """
        Convert the symbolic ODE system into a numerical function and return
        all necessary information for simulation in a dictionary.

        Returns:
            Dict: A dictionary containing:
            - 'func' (Callable): The numerical function f(t, y, *params).
            - 'y0' (np.ndarray): The array of initial conditions.
            - 'params' (Tuple): A tuple of the default parameter values.
            - 'species_names' (List[str]): Ordered list of species names.
            - 'param_names' (List[str]): Ordered list of parameter names.
        """
        # Ensure a consistent order for species and parameters
        ordered_species = list(self.species.values())
        ordered_params = list(self.parameters.values())

        # Get symbolic representations in the correct order
        species_symbols = [s.symbol for s in ordered_species]
        param_symbols = [p.symbol for p in ordered_params]
        
        # Generate the symbolic ODEs
        symbolic_odes = self.generate_odes()
        ode_expressions = [symbolic_odes[s] for s in species_symbols]

        # Use lambdify to create a fast numerical function
        lambda_func = lambdify(species_symbols + param_symbols, ode_expressions, 'numpy')

        # Create a wrapper function that matches the standard f(t, y, *p) signature
        def ode_function(t, y, *p):
            return lambda_func(*y, *p)

        # Package all results into a single dictionary
        simulation_data = {
            'func': ode_function,
            'y0': np.array([s.initial_condition for s in ordered_species]),
            'params': tuple(p.get_default_value() for p in ordered_params),
            'species_names': [s.name for s in ordered_species],
            'param_names': [p.name for p in ordered_params]
        }
        
        return simulation_data

    def display_latex_odes(self) -> None:
        """
        Generate and display the system of ODEs in a clean LaTeX format.
        
        Note: This method requires IPython/Jupyter environment for display.
        """
        try:
            from IPython.display import display, Latex
        except ImportError:
            print("IPython not available. LaTeX display requires Jupyter environment.")
            return
            
        # 1. Generate the ODE dictionary {symbol: expression}
        odes = self.generate_odes()
        
        # 2. Convert the dictionary into a list of SymPy equations (dx/dt = expression)
        equations = [sp.Eq(sp.Derivative(s, t), e) for s, e in odes.items()]
        
        # 3. Format the list of equations into a single LaTeX string
        # The .replace("=", "&=", 1) aligns all equations at the equals sign
        latex_str = r"\begin{aligned}" + r" \\ ".join(
            [sp.latex(eq).replace("=", " &= ", 1) for eq in equations]
        ) + r"\end{aligned}"
        
        # 4. Display the formatted string as rendered LaTeX
        display(Latex(f"$${latex_str}$$"))

    def display_latex_odes_with_values(self) -> None:
            """
            Generate and display the system of ODEs in LaTeX format,
            substituting parameter symbols with their default numerical values.
            
            Note: This method requires an IPython/Jupyter environment for display.
            """
            try:
                from IPython.display import display, Latex
            except ImportError:
                print("IPython not available. LaTeX display requires a Jupyter environment.")
                return
                
            # 1. Get the symbolic ODEs
            odes = self.generate_odes()
            
            # 2. Create a dictionary to map parameter symbols to their default values
            param_substitutions = {
                p.get_symbol(): p.get_default_value() 
                for p in self.parameters.values()
            }
            
            # 3. Substitute values and create a list of SymPy equations
            equations = []
            for species_symbol, ode_expr in odes.items():
                # Apply the substitution to the expression
                expr_with_values = ode_expr.subs(param_substitutions)
                # Create the equation object: d(species)/dt = substituted_expression
                eq = sp.Eq(sp.Derivative(species_symbol, t), expr_with_values)
                equations.append(eq)

            # 4. Format the list of equations into a single LaTeX string for display
            # The .replace("=", "&=", 1) aligns all equations at the equals sign
            latex_str = r"\begin{aligned}" + r" \\ ".join(
                [sp.latex(eq).replace("=", " &= ", 1) for eq in equations]
            ) + r"\end{aligned}"
            
            # 5. Display the rendered LaTeX
            display(Latex(f"$${latex_str}$$"))

    def generate_propensity_vector(self) -> sp.Matrix:
        """
        Generate the symbolic propensity vector where each element is the 
        rate law for a reaction, suitable for Gillespie simulation.
        
        Returns:
            sp.Matrix: Vector of symbolic propensity expressions
        """
        propensities = []
        
        for reaction in self.reactions:
            # For Gillespie, we need to convert the rate law to use 
            # discrete counts instead of continuous concentrations
            rate_law = reaction.rate_law
            
            # Replace continuous species symbols with discrete count symbols
            discrete_rate_law = self._convert_to_discrete_propensity(rate_law)
            propensities.append(discrete_rate_law)
        
        return sp.Matrix(propensities)
    
    def _convert_to_discrete_propensity(self, rate_law: sp.Expr) -> sp.Expr:
        """
        Convert a continuous rate law to a discrete propensity function.
        
        Args:
            rate_law (sp.Expr): The symbolic rate law expression
            
        Returns:
            sp.Expr: The discrete propensity expression
        """
        # Create discrete species symbols (without time dependence)
        discrete_symbols = {}
        for species_name in self.species.keys():
            discrete_symbols[species_name] = sp.Symbol(species_name, positive=True)
        
        # Replace continuous species symbols with discrete ones
        discrete_rate_law = rate_law
        for species in self.species.values():
            discrete_symbols[species.name] = discrete_symbols[species.name]
            discrete_rate_law = discrete_rate_law.subs(species.symbol, discrete_symbols[species.name])
        
        return discrete_rate_law
    
    def generate_gillespie_components(self) -> tuple:
        """
        Generate all components needed for Gillespie simulation.
        
        Returns:
            tuple: (stoichiometric_matrix, propensity_vector, initial_state, rate_constants)
        """
        S = self.generate_stoichiometric_matrix()
        propensities = self.generate_propensity_vector()
        
        # Initial state (convert from float to int for discrete counts)
        initial_state = np.array([int(s.initial_condition) for s in self.species.values()])
        
        # Rate constants (extract default values)
        rate_constants = np.array([p.get_default_value() for p in self.parameters.values()])
        
        return S, propensities, initial_state, rate_constants

    def __repr__(self) -> str:
        return (f"SystemModel(name='{self.name}', "
                f"species={len(self.species)}, "
                f"parameters={len(self.parameters)}, "
                f"reactions={len(self.reactions)})")
