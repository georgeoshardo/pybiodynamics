"""
Tests for simulation modules.

This module tests:
- ODE simulation functionality
- Gillespie simulation functionality  
- Integration between models and simulators
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from pybiodynamics.core.models import Species, Parameter, SystemModel
from pybiodynamics.core.reactions import MassActionReaction
from pybiodynamics.simulation.ode import simulate_ode
from pybiodynamics.simulation.gillespie import GillespieSimulator, create_gillespie_from_system_model


class TestODESimulation:
    """Test cases for ODE simulation."""
    
    def setup_method(self):
        """Set up a simple test model."""
        self.model = SystemModel("Test Model")
        
        # Simple decay: A -> 0
        self.species_a = Species('A', initial_condition=100.0)
        self.param_k = Parameter('k', default_value=0.1)
        
        self.model.add_species(self.species_a).add_parameter(self.param_k)
        
        decay_reaction = MassActionReaction(
            'decay',
            reactants={self.species_a: 1},
            products={},
            rate=self.param_k
        )
        self.model.add_reaction(decay_reaction)
    
    def test_simulate_ode_basic(self):
        """Test basic ODE simulation."""
        t_span = [0, 10]
        result = simulate_ode(self.model, t_span)
        
        assert 'solution' in result
        assert 'sim_data' in result
        assert 'success' in result
        assert result['success'] is True
        
        solution = result['solution']
        assert hasattr(solution, 't')
        assert hasattr(solution, 'y')
        assert len(solution.y) == 1  # One species
        
        # Check that concentration decreases (decay)
        assert solution.y[0, 0] == 100.0  # Initial condition
        assert solution.y[0, -1] < 100.0  # Should decay
    
    def test_simulate_ode_custom_times(self):
        """Test ODE simulation with custom time points."""
        t_span = [0, 5]
        t_eval = np.linspace(0, 5, 21)  # 21 time points
        
        result = simulate_ode(self.model, t_span, t_eval)
        
        assert result['success'] is True
        solution = result['solution']
        assert len(solution.t) == 21
        np.testing.assert_array_almost_equal(solution.t, t_eval)
    
    def test_simulate_ode_different_method(self):
        """Test ODE simulation with different integration method."""
        t_span = [0, 5]
        result = simulate_ode(self.model, t_span, method='RK45')
        
        assert result['success'] is True
    
    def test_exponential_decay_analytical(self):
        """Test that exponential decay matches analytical solution."""
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 101)
        
        result = simulate_ode(self.model, t_span, t_eval)
        
        # Analytical solution: A(t) = A0 * exp(-k*t)
        A0 = 100.0
        k = 0.1
        analytical = A0 * np.exp(-k * t_eval)
        
        # Numerical solution should be close to analytical
        numerical = result['solution'].y[0]
        np.testing.assert_allclose(numerical, analytical, rtol=1e-6)


class TestGillespieComponents:
    """Test Gillespie simulation components."""
    
    def setup_method(self):
        """Set up test model for Gillespie simulation."""
        self.model = SystemModel("Gillespie Test")
        
        # Birth-death process: 0 -> A, A -> 0
        self.species_a = Species('A', initial_condition=50)  # Integer for discrete simulation
        self.birth_rate = Parameter('birth', default_value=10.0)
        self.death_rate = Parameter('death', default_value=0.1)
        
        self.model.add_species(self.species_a)
        self.model.add_parameter(self.birth_rate)
        self.model.add_parameter(self.death_rate)
        
        # Birth: 0 -> A
        birth = MassActionReaction('birth', {}, {self.species_a: 1}, self.birth_rate)
        # Death: A -> 0  
        death = MassActionReaction('death', {self.species_a: 1}, {}, self.death_rate)
        
        self.model.add_reaction(birth).add_reaction(death)
    
    def test_create_gillespie_components(self):
        """Test creation of Gillespie simulation components."""
        components = create_gillespie_from_system_model(self.model)
        
        # Check required components
        required_keys = ['stoichiometric_matrix', 'propensity_func', 'initial_state', 
                        'rate_constants', 'species_labels', 'symbolic_propensities']
        for key in required_keys:
            assert key in components
        
        # Check stoichiometric matrix
        S = components['stoichiometric_matrix']
        assert S.shape == (1, 2)  # 1 species, 2 reactions
        assert S[0, 0] == 1   # Birth increases A
        assert S[0, 1] == -1  # Death decreases A
        
        # Check initial state
        assert components['initial_state'][0] == 50
        
        # Check rate constants
        np.testing.assert_array_equal(components['rate_constants'], [10.0, 0.1])
    
    def test_gillespie_simulator_creation(self):
        """Test GillespieSimulator instantiation."""
        components = create_gillespie_from_system_model(self.model)
        
        simulator = GillespieSimulator(
            stoichiometric_matrix=components['stoichiometric_matrix'],
            propensity_func=components['propensity_func'],
            initial_state=components['initial_state'],
            rate_constants=components['rate_constants'],
            species_labels=components['species_labels']
        )
        
        assert simulator.S.shape == (1, 2)
        assert len(simulator.x0) == 1
        assert simulator.x0[0] == 50
        assert len(simulator.labels) == 1
        assert simulator.labels[0] == 'A'
    
    def test_gillespie_simulation_run(self):
        """Test running a Gillespie simulation."""
        components = create_gillespie_from_system_model(self.model)
        
        simulator = GillespieSimulator(
            stoichiometric_matrix=components['stoichiometric_matrix'],
            propensity_func=components['propensity_func'],
            initial_state=components['initial_state'],
            rate_constants=components['rate_constants'],
            species_labels=components['species_labels']
        )
        
        # Run short simulation
        simulator.run(max_iter=1000, use_numba=False, seed=42)
        
        # Check results exist
        assert simulator.X is not None
        assert simulator.T is not None
        assert simulator.tsteps is not None
        
        # Check dimensions
        assert simulator.X.shape == (1, 1000)  # 1 species, 1000 iterations
        assert len(simulator.T) == 1000
        assert len(simulator.tsteps) == 1000
        
        # Check that simulation progressed in time
        assert simulator.T[-1] > simulator.T[0]
        assert np.all(simulator.T[1:] >= simulator.T[:-1])  # Time should be non-decreasing
    
    def test_gillespie_statistics(self):
        """Test Gillespie simulation statistics."""
        components = create_gillespie_from_system_model(self.model)
        
        simulator = GillespieSimulator(
            stoichiometric_matrix=components['stoichiometric_matrix'],
            propensity_func=components['propensity_func'],
            initial_state=components['initial_state'],
            rate_constants=components['rate_constants'],
            species_labels=components['species_labels']
        )
        
        simulator.run(max_iter=10000, use_numba=False, seed=42)
        stats = simulator.get_stats()
        
        # Check stats structure
        assert 'Component' in stats.columns
        assert 'Gillespie Mean' in stats.columns
        assert 'Gillespie Variance' in stats.columns
        assert len(stats) == 1  # One species
        assert stats.iloc[0]['Component'] == 'A'
        
        # For birth-death process, equilibrium mean should be birth_rate/death_rate = 10/0.1 = 100
        # Allow some tolerance for stochastic simulation
        mean_value = stats.iloc[0]['Gillespie Mean']
        assert 80 < mean_value < 120  # Rough check
    
    def test_gillespie_error_before_run(self):
        """Test that accessing results before running raises error."""
        components = create_gillespie_from_system_model(self.model)
        
        simulator = GillespieSimulator(
            stoichiometric_matrix=components['stoichiometric_matrix'],
            propensity_func=components['propensity_func'],
            initial_state=components['initial_state'],
            rate_constants=components['rate_constants'],
            species_labels=components['species_labels']
        )
        
        with pytest.raises(RuntimeError, match="Simulation has not been run"):
            simulator.get_stats()
        
        with pytest.raises(RuntimeError, match="Simulation has not been run"):
            simulator.plot_trace()


class TestSimulationIntegration:
    """Integration tests between models and simulation methods."""
    
    def test_ode_gillespie_consistency(self):
        """Test that ODE and Gillespie give similar results for large populations."""
        # Create a model with large populations where stochastic effects are small
        model = SystemModel("Consistency Test")
        
        A = Species('A', initial_condition=10000)  # Large population
        k = Parameter('k', default_value=0.1)  # Small rate to keep system stable
        
        model.add_species(A).add_parameter(k)
        
        # Simple decay: A -> 0
        decay = MassActionReaction('decay', {A: 1}, {}, k)
        model.add_reaction(decay)
        
        # ODE simulation
        ode_result = simulate_ode(model, t_span=[0, 100])
        
        # Gillespie simulation
        components = create_gillespie_from_system_model(model)
        simulator = GillespieSimulator(**{k: v for k, v in components.items() 
                                        if k != 'symbolic_propensities'})
        simulator.run(max_iter=50000, use_numba=False, seed=42)
        
        # Compare final values
        ode_final = ode_result['solution'].y[0, -1]
        gillespie_final = simulator.X[0, -1]
        
        # Should be reasonably close (within 10% for large populations)
        relative_error = abs(ode_final - gillespie_final) 
        assert relative_error < 0.1
    
    def test_conservation_laws(self):
        """Test that conservation laws are preserved in both simulation methods."""
        # Create a model with conservation: A + B <-> C
        model = SystemModel("Conservation Test")
        
        A = Species('A', initial_condition=100)
        B = Species('B', initial_condition=100)
        C = Species('C', initial_condition=0)
        
        k_forward = Parameter('k_f', default_value=0.01)
        k_reverse = Parameter('k_r', default_value=0.1)
        
        model.add_species(A).add_species(B).add_species(C)
        model.add_parameter(k_forward).add_parameter(k_reverse)
        
        # Forward: A + B -> C
        forward = MassActionReaction('forward', {A: 1, B: 1}, {C: 1}, k_forward)
        # Reverse: C -> A + B
        reverse = MassActionReaction('reverse', {C: 1}, {A: 1, B: 1}, k_reverse)
        
        model.add_reaction(forward).add_reaction(reverse)
        
        # ODE simulation
        ode_result = simulate_ode(model, t_span=[0, 50])
        
        # Check conservation: [A] + [C] should be constant (initially 100)
        A_plus_C = ode_result['solution'].y[0] + ode_result['solution'].y[2]
        initial_total = A_plus_C[0]
        
        # Should be conserved within numerical precision
        np.testing.assert_allclose(A_plus_C, initial_total, rtol=1e-10)
        
        # Similarly for B + C
        B_plus_C = ode_result['solution'].y[1] + ode_result['solution'].y[2]
        np.testing.assert_allclose(B_plus_C, 100.0, rtol=1e-10)


class TestSimulationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_model_simulation(self):
        """Test simulation of model with no reactions."""
        model = SystemModel("Empty")
        A = Species('A', initial_condition=100.0)
        model.add_species(A)
        
        # Should still work but give constant solution
        result = simulate_ode(model, t_span=[0, 10])
        assert result['success'] is True
        
        # Concentration should remain constant
        np.testing.assert_array_almost_equal(
            result['solution'].y[0], 
            100.0 * np.ones_like(result['solution'].t)
        )
    
    def test_zero_initial_conditions(self):
        """Test simulation with zero initial conditions."""
        model = SystemModel("Zero Initial")
        
        A = Species('A', initial_condition=0.0)
        k = Parameter('k', default_value=1.0)
        
        model.add_species(A).add_parameter(k)
        
        # Birth reaction: 0 -> A
        birth = MassActionReaction('birth', {}, {A: 1}, k)
        model.add_reaction(birth)
        
        result = simulate_ode(model, t_span=[0, 5])
        assert result['success'] is True
        
        # Should have linear growth: A(t) = k*t
        expected = result['solution'].t  # k=1, so A(t) = t
        np.testing.assert_allclose(result['solution'].y[0], expected, rtol=1e-6)
