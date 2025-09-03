"""
Tests for core model classes.

This module tests the fundamental building blocks:
- Species
- Parameter  
- SystemModel
"""

import pytest
import numpy as np
import sympy as sp
from pybiodynamics.core.models import Species, Parameter, SystemModel
from pybiodynamics.core.reactions import MassActionReaction


class TestSpecies:
    """Test cases for the Species class."""
    
    def test_species_creation(self):
        """Test basic species creation."""
        species = Species('test_species', initial_condition=100.0)
        assert species.name == 'test_species'
        assert species.initial_condition == 100.0
        assert isinstance(species.symbol, sp.core.function.UndefinedFunction)
    
    def test_species_with_kwargs(self):
        """Test species creation with sympy kwargs."""
        species = Species('positive_species', initial_condition=50.0, positive=True)
        assert species.name == 'positive_species'
        assert species.initial_condition == 50.0
    
    def test_species_equality(self):
        """Test species equality comparison."""
        species1 = Species('same_name', initial_condition=10.0)
        species2 = Species('same_name', initial_condition=20.0)
        species3 = Species('different_name', initial_condition=10.0)
        
        assert species1 == species2  # Same name
        assert species1 != species3  # Different name
    
    def test_species_hash(self):
        """Test that species can be used as dictionary keys."""
        species1 = Species('hashable', initial_condition=1.0)
        species2 = Species('hashable', initial_condition=2.0)
        
        # Should be able to use as dict keys
        test_dict = {species1: 'value1'}
        test_dict[species2] = 'value2'  # Should overwrite due to same name
        
        assert len(test_dict) == 1
        assert test_dict[species1] == 'value2'


class TestParameter:
    """Test cases for the Parameter class."""
    
    def test_parameter_creation(self):
        """Test basic parameter creation."""
        param = Parameter('test_param', default_value=5.0)
        assert param.name == 'test_param'
        assert param.default_value == 5.0
        assert param.get_default_value() == 5.0
        assert isinstance(param.symbol, sp.Symbol)
    
    def test_parameter_no_default(self):
        """Test parameter without default value."""
        param = Parameter('no_default')
        assert param.name == 'no_default'
        assert param.default_value is None
        
        with pytest.raises(ValueError, match="No default value"):
            param.get_default_value()
    
    def test_parameter_with_values(self):
        """Test parameter with sweep values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        param = Parameter('sweep_param', default_value=3.0, values=values)
        assert param.values == values
    
    def test_parameter_arithmetic(self):
        """Test parameter arithmetic operations."""
        param1 = Parameter('p1', default_value=2.0)
        param2 = Parameter('p2', default_value=3.0)
        
        # Multiplication
        result = param1 * param2
        assert isinstance(result, Parameter)
        assert result.default_value == 6.0
        
        # Scalar multiplication
        result_scalar = param1 * 5
        assert result_scalar.default_value == 10.0
        
        # Right multiplication
        result_rmul = 5 * param1
        assert result_rmul.default_value == 10.0
        
        # Division
        result_div = param2 / param1
        assert result_div.default_value == 1.5
        
        # Scalar division
        result_scalar_div = param2 / 2
        assert result_scalar_div.default_value == 1.5


class TestSystemModel:
    """Test cases for the SystemModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = SystemModel("Test Model")
        self.species_a = Species('A', initial_condition=100.0)
        self.species_b = Species('B', initial_condition=50.0)
        self.param_k = Parameter('k', default_value=0.1)
    
    def test_model_creation(self):
        """Test basic model creation."""
        assert self.model.name == "Test Model"
        assert len(self.model.species) == 0
        assert len(self.model.parameters) == 0
        assert len(self.model.reactions) == 0
    
    def test_add_species(self):
        """Test adding species to model."""
        self.model.add_species(self.species_a)
        assert 'A' in self.model.species
        assert self.model.species['A'] == self.species_a
        
        # Test method chaining
        result = self.model.add_species(self.species_b)
        assert result is self.model
        assert len(self.model.species) == 2
    
    def test_duplicate_species_error(self):
        """Test error when adding duplicate species."""
        self.model.add_species(self.species_a)
        duplicate_species = Species('A', initial_condition=200.0)
        
        with pytest.raises(ValueError, match="Species 'A' already exists"):
            self.model.add_species(duplicate_species)
    
    def test_add_parameter(self):
        """Test adding parameters to model."""
        self.model.add_parameter(self.param_k)
        assert 'k' in self.model.parameters
        assert self.model.parameters['k'] == self.param_k
    
    def test_duplicate_parameter_error(self):
        """Test error when adding duplicate parameter."""
        self.model.add_parameter(self.param_k)
        duplicate_param = Parameter('k', default_value=0.2)
        
        with pytest.raises(ValueError, match="Parameter 'k' already exists"):
            self.model.add_parameter(duplicate_param)
    
    def test_add_reaction(self):
        """Test adding reactions to model."""
        self.model.add_species(self.species_a).add_species(self.species_b)
        self.model.add_parameter(self.param_k)
        
        reaction = MassActionReaction(
            'test_reaction',
            reactants={self.species_a: 1},
            products={self.species_b: 1},
            rate=self.param_k
        )
        
        self.model.add_reaction(reaction)
        assert len(self.model.reactions) == 1
        assert self.model.reactions[0] == reaction
    
    def test_stoichiometric_matrix(self):
        """Test stoichiometric matrix generation."""
        self.model.add_species(self.species_a).add_species(self.species_b)
        self.model.add_parameter(self.param_k)
        
        # A -> B reaction
        reaction = MassActionReaction(
            'conversion',
            reactants={self.species_a: 1},
            products={self.species_b: 1},
            rate=self.param_k
        )
        self.model.add_reaction(reaction)
        
        S = self.model.generate_stoichiometric_matrix()
        
        # Should be 2x1 matrix (2 species, 1 reaction)
        assert S.shape == (2, 1)
        assert S[0, 0] == -1  # A is consumed
        assert S[1, 0] == 1   # B is produced
    
    def test_ode_generation(self):
        """Test ODE system generation."""
        self.model.add_species(self.species_a).add_species(self.species_b)
        self.model.add_parameter(self.param_k)
        
        reaction = MassActionReaction(
            'conversion',
            reactants={self.species_a: 1},
            products={self.species_b: 1},
            rate=self.param_k
        )
        self.model.add_reaction(reaction)
        
        odes = self.model.generate_odes()
        
        assert len(odes) == 2
        assert self.species_a.symbol in odes
        assert self.species_b.symbol in odes
        
        # Check that ODEs are sympy expressions
        assert isinstance(odes[self.species_a.symbol], sp.Expr)
        assert isinstance(odes[self.species_b.symbol], sp.Expr)
    
    def test_lambdify_odes(self):
        """Test lambdification of ODE system."""
        self.model.add_species(self.species_a).add_species(self.species_b)
        self.model.add_parameter(self.param_k)
        
        reaction = MassActionReaction(
            'conversion',
            reactants={self.species_a: 1},
            products={self.species_b: 1},
            rate=self.param_k
        )
        self.model.add_reaction(reaction)
        
        sim_data = self.model.lambdify_odes()
        
        # Check return structure
        assert 'func' in sim_data
        assert 'y0' in sim_data
        assert 'params' in sim_data
        assert 'species_names' in sim_data
        assert 'param_names' in sim_data
        
        # Check types and values
        assert callable(sim_data['func'])
        assert isinstance(sim_data['y0'], np.ndarray)
        assert len(sim_data['y0']) == 2
        assert sim_data['y0'][0] == 100.0  # A initial condition
        assert sim_data['y0'][1] == 50.0   # B initial condition
        
        # Test function call
        dydt = sim_data['func'](0, sim_data['y0'], *sim_data['params'])
        assert isinstance(dydt, np.ndarray)
        assert len(dydt) == 2
    
    def test_model_repr(self):
        """Test string representation of model."""
        self.model.add_species(self.species_a)
        self.model.add_parameter(self.param_k)
        
        repr_str = repr(self.model)
        assert "Test Model" in repr_str
        assert "species=1" in repr_str
        assert "parameters=1" in repr_str
        assert "reactions=0" in repr_str


class TestModelIntegration:
    """Integration tests for model components working together."""
    
    def test_simple_decay_model(self):
        """Test a simple A -> 0 decay model."""
        model = SystemModel("Decay")
        
        A = Species('A', initial_condition=100.0)
        k_decay = Parameter('k_decay', default_value=0.1)
        
        model.add_species(A).add_parameter(k_decay)
        
        decay_reaction = MassActionReaction(
            'decay',
            reactants={A: 1},
            products={},
            rate=k_decay
        )
        model.add_reaction(decay_reaction)
        
        # Test ODE generation
        odes = model.generate_odes()
        assert len(odes) == 1
        
        # Test lambdification
        sim_data = model.lambdify_odes()
        
        # At t=0, should have decay rate = k * A = 0.1 * 100 = 10
        dydt = sim_data['func'](0, sim_data['y0'], *sim_data['params'])
        assert abs(dydt[0] + 10.0) < 1e-10  # Negative because A is being consumed
    
    def test_enzyme_kinetics_model(self):
        """Test a simple enzyme kinetics model: E + S -> E + P."""
        model = SystemModel("Enzyme Kinetics")
        
        enzyme = Species('E', initial_condition=10.0)
        substrate = Species('S', initial_condition=1000.0)
        product = Species('P', initial_condition=0.0)
        
        k_cat = Parameter('k_cat', default_value=100.0)
        
        model.add_species(enzyme).add_species(substrate).add_species(product)
        model.add_parameter(k_cat)
        
        catalysis = MassActionReaction(
            'catalysis',
            reactants={enzyme: 1, substrate: 1},
            products={enzyme: 1, product: 1},
            rate=k_cat
        )
        model.add_reaction(catalysis)
        
        # Check stoichiometric matrix
        S = model.generate_stoichiometric_matrix()
        assert S.shape == (3, 1)  # 3 species, 1 reaction
        assert S[0, 0] == 0   # E: consumed and produced (net 0)
        assert S[1, 0] == -1  # S: consumed
        assert S[2, 0] == 1   # P: produced
        
        # Test simulation data
        sim_data = model.lambdify_odes()
        dydt = sim_data['func'](0, sim_data['y0'], *sim_data['params'])
        
        # Rate should be k_cat * E * S = 100 * 10 * 1000 = 1,000,000
        expected_rate = 1_000_000
        assert abs(dydt[0]) < 1e-10  # dE/dt = 0
        assert abs(dydt[1] + expected_rate) < 1e-6  # dS/dt = -rate
        assert abs(dydt[2] - expected_rate) < 1e-6  # dP/dt = +rate
