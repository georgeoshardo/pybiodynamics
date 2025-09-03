"""
Tests for reaction classes.

This module tests reaction types:
- MassActionReaction
- LogisticGrowthReaction
- Reaction base class
"""

import pytest
import sympy as sp
from pybiodynamics.core.models import Species, Parameter
from pybiodynamics.core.reactions import Reaction, MassActionReaction, LogisticGrowthReaction


class TestReactionBase:
    """Test cases for the base Reaction class."""
    
    def test_reaction_is_abstract(self):
        """Test that Reaction cannot be instantiated directly."""
        species_a = Species('A')
        param_k = Parameter('k', default_value=1.0)
        
        with pytest.raises(TypeError):
            # Should fail because _generate_rate_law is abstract
            Reaction('test', {species_a: 1}, {}, param_k)


class TestMassActionReaction:
    """Test cases for MassActionReaction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.species_a = Species('A', initial_condition=100.0)
        self.species_b = Species('B', initial_condition=50.0)
        self.species_c = Species('C', initial_condition=0.0)
        self.param_k = Parameter('k', default_value=0.1)
    
    def test_simple_unimolecular_reaction(self):
        """Test A -> B reaction."""
        reaction = MassActionReaction(
            name='conversion',
            reactants={self.species_a: 1},
            products={self.species_b: 1},
            rate=self.param_k
        )
        
        assert reaction.name == 'conversion'
        assert reaction.reactants == {self.species_a: 1}
        assert reaction.products == {self.species_b: 1}
        assert reaction.rate == self.param_k
        
        # Rate law should be k * [A]
        expected_rate_law = self.param_k.symbol * self.species_a.symbol
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_bimolecular_reaction(self):
        """Test A + B -> C reaction."""
        reaction = MassActionReaction(
            name='combination',
            reactants={self.species_a: 1, self.species_b: 1},
            products={self.species_c: 1},
            rate=self.param_k
        )
        
        # Rate law should be k * [A] * [B]
        expected_rate_law = (self.param_k.symbol * 
                           self.species_a.symbol * 
                           self.species_b.symbol)
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_higher_order_reaction(self):
        """Test 2A + B -> C reaction."""
        reaction = MassActionReaction(
            name='higher_order',
            reactants={self.species_a: 2, self.species_b: 1},
            products={self.species_c: 1},
            rate=self.param_k
        )
        
        # Rate law should be k * [A]^2 * [B]
        expected_rate_law = (self.param_k.symbol * 
                           self.species_a.symbol**2 * 
                           self.species_b.symbol)
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_decay_reaction(self):
        """Test A -> 0 (decay) reaction."""
        reaction = MassActionReaction(
            name='decay',
            reactants={self.species_a: 1},
            products={},  # No products
            rate=self.param_k
        )
        
        # Rate law should be k * [A]
        expected_rate_law = self.param_k.symbol * self.species_a.symbol
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_birth_reaction(self):
        """Test 0 -> A (birth) reaction."""
        reaction = MassActionReaction(
            name='birth',
            reactants={},  # No reactants
            products={self.species_a: 1},
            rate=self.param_k
        )
        
        # Rate law should be just k (zero-order kinetics)
        expected_rate_law = self.param_k.symbol
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_autocatalytic_reaction(self):
        """Test A -> 2A (autocatalytic) reaction."""
        reaction = MassActionReaction(
            name='autocatalysis',
            reactants={self.species_a: 1},
            products={self.species_a: 2},
            rate=self.param_k
        )
        
        # Rate law should be k * [A]
        expected_rate_law = self.param_k.symbol * self.species_a.symbol
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_reaction_repr(self):
        """Test string representation of reaction."""
        reaction = MassActionReaction(
            name='test_reaction',
            reactants={self.species_a: 2, self.species_b: 1},
            products={self.species_c: 1},
            rate=self.param_k
        )
        
        repr_str = repr(reaction)
        assert 'test_reaction' in repr_str
        assert '2A + B' in repr_str or 'A + B' in repr_str  # Order might vary
        assert '-> C' in repr_str or '→ C' in repr_str


class TestLogisticGrowthReaction:
    """Test cases for LogisticGrowthReaction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.population = Species('x', initial_condition=10.0)
        self.growth_rate = Parameter('r', default_value=0.5)
        self.carrying_capacity = Parameter('K', default_value=1000.0)
    
    def test_logistic_reaction_creation(self):
        """Test basic logistic growth reaction creation."""
        reaction = LogisticGrowthReaction(
            name='growth',
            species=self.population,
            r=self.growth_rate,
            K=self.carrying_capacity
        )
        
        assert reaction.name == 'growth'
        assert reaction.species == self.population
        assert reaction.r == self.growth_rate
        assert reaction.K == self.carrying_capacity
        
        # Should set up as x -> 2x conceptually
        assert reaction.reactants == {self.population: 1}
        assert reaction.products == {self.population: 2}
    
    def test_logistic_rate_law(self):
        """Test logistic growth rate law generation."""
        reaction = LogisticGrowthReaction(
            name='growth',
            species=self.population,
            r=self.growth_rate,
            K=self.carrying_capacity
        )
        
        # Rate law should be r * x * (1 - x/K)
        x = self.population.symbol
        r = self.growth_rate.symbol
        K = self.carrying_capacity.symbol
        
        expected_rate_law = r * x * (1 - x/K)
        assert reaction.rate_law.equals(expected_rate_law)
    
    def test_logistic_rate_law_expansion(self):
        """Test that the logistic rate law expands correctly."""
        reaction = LogisticGrowthReaction(
            name='growth',
            species=self.population,
            r=self.growth_rate,
            K=self.carrying_capacity
        )
        
        # Expand the rate law
        expanded = sp.expand(reaction.rate_law)
        
        x = self.population.symbol
        r = self.growth_rate.symbol
        K = self.carrying_capacity.symbol
        
        # Should expand to r*x - r*x^2/K
        expected_expanded = r*x - r*x**2/K
        assert expanded.equals(expected_expanded)
    
    def test_logistic_at_limits(self):
        """Test logistic growth rate law at boundary conditions."""
        reaction = LogisticGrowthReaction(
            name='growth',
            species=self.population,
            r=self.growth_rate,
            K=self.carrying_capacity
        )
        
        x = self.population.symbol
        r = self.growth_rate.symbol
        K = self.carrying_capacity.symbol
        
        rate_law = reaction.rate_law
        
        # At x = 0, rate should be 0
        rate_at_zero = rate_law.subs(x, 0)
        assert rate_at_zero == 0
        
        # At x = K, rate should be 0
        rate_at_K = rate_law.subs(x, K)
        assert rate_at_K == 0
        
        # At x = K/2, rate should be r*K/4
        rate_at_half_K = rate_law.subs(x, K/2)
        expected_max_rate = r*K/4
        assert rate_at_half_K.equals(expected_max_rate)


class TestReactionIntegration:
    """Integration tests for reactions working together."""
    
    def test_multiple_mass_action_reactions(self):
        """Test multiple mass action reactions with shared species."""
        A = Species('A', initial_condition=100.0)
        B = Species('B', initial_condition=50.0)
        C = Species('C', initial_condition=0.0)
        
        k1 = Parameter('k1', default_value=0.1)
        k2 = Parameter('k2', default_value=0.05)
        
        # A -> B
        reaction1 = MassActionReaction('r1', {A: 1}, {B: 1}, k1)
        # B -> C
        reaction2 = MassActionReaction('r2', {B: 1}, {C: 1}, k2)
        
        # Check that reactions have correct rate laws
        assert reaction1.rate_law.equals(k1.symbol * A.symbol)
        assert reaction2.rate_law.equals(k2.symbol * B.symbol)
        
        # Check that reactions are independent
        assert A not in reaction2.reactants
        assert C not in reaction1.products
    
    def test_reversible_reaction_pair(self):
        """Test forward and reverse reactions."""
        A = Species('A', initial_condition=100.0)
        B = Species('B', initial_condition=0.0)
        
        k_forward = Parameter('k_f', default_value=0.1)
        k_reverse = Parameter('k_r', default_value=0.01)
        
        # A -> B (forward)
        forward = MassActionReaction('forward', {A: 1}, {B: 1}, k_forward)
        # B -> A (reverse)
        reverse = MassActionReaction('reverse', {B: 1}, {A: 1}, k_reverse)
        
        # Check rate laws
        assert forward.rate_law.equals(k_forward.symbol * A.symbol)
        assert reverse.rate_law.equals(k_reverse.symbol * B.symbol)
        
        # Net rate for A should be -k_f*A + k_r*B
        # Net rate for B should be +k_f*A - k_r*B
    
    def test_competitive_reactions(self):
        """Test reactions competing for the same reactant."""
        A = Species('A', initial_condition=100.0)
        B = Species('B', initial_condition=0.0)
        C = Species('C', initial_condition=0.0)
        
        k1 = Parameter('k1', default_value=0.1)
        k2 = Parameter('k2', default_value=0.2)
        
        # A -> B
        reaction1 = MassActionReaction('pathway1', {A: 1}, {B: 1}, k1)
        # A -> C (competing pathway)
        reaction2 = MassActionReaction('pathway2', {A: 1}, {C: 1}, k2)
        
        # Both reactions should depend on [A]
        assert reaction1.rate_law.equals(k1.symbol * A.symbol)
        assert reaction2.rate_law.equals(k2.symbol * A.symbol)
        
        # Total consumption rate of A should be (k1 + k2) * [A]
