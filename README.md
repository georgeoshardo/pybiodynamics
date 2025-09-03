# PyBiodynamics

A Python library for modeling and simulating biological dynamic systems with both deterministic (ODE) and stochastic (Gillespie) approaches. 

#### Current features:

- **Model Building**: Define species, parameters, and reactions
- **Multiple Reaction Types**: Support for mass-action kinetics, logistic growth, and custom reaction types
- **Simulation Methods**: 
  - Deterministic simulation using scipy's ODE solvers
  - Stochastic simulation using the Gillespie algorithm (SSA)
- **Plotting**:Some built-in plotting functions for simulation results
- **Parameter Analysis**: Some support for parameter sweeps and sensitivity analysis
- **LaTeX**: Display system ODEs/propensity functions as LaTeX

#### Planned:

- Unit integration with Pint for model validation
- Currently models have a graph structure for reactions, but the graph itself is not extensively used. This will be used for system analysis soon hopefully.
- SBML import 


## Installation

### From Source
```bash
git clone https://github.com/georgeoshardo/pybiodynamics.git
cd pybiodynamics
pip install -e .[jupyter]
```

### Development Installation
```bash
git clone https://github.com/georgeoshardo/pybiodynamics.git
cd pybiodynamics
pip install -e ".[dev]"
```


## Core Components

### Species
Represents biological entities (molecules, cells, populations) with initial conditions:

```python
bacteria = Species('bacteria', initial_condition=100.0, positive=True)
substrate = Species('substrate', initial_condition=1000.0, positive=True)
```

### Parameters
Model parameters with default values and optional parameter sweep values:

```python
growth_rate = Parameter('mu', default_value=0.1, positive=True)
yield_coeff = Parameter('Y', default_value=0.5, positive=True)
```

### Reactions
Define how species interact:

#### Mass Action Kinetics
```python
# A + B -> C (rate = k * [A] * [B])
reaction = MassActionReaction(
    name='conversion',
    reactants={A: 1, B: 1},
    products={C: 1},
    rate=rate_constant
)
```

#### Logistic Growth
```python
# dx/dt = r*x*(1 - x/K)
reaction = LogisticGrowthReaction(
    name='growth',
    species=population,
    r=growth_rate,
    K=carrying_capacity
)
```

#### Custom Reactions
Extend the `Reaction` base class:

```python
class MichaelisMentenReaction(Reaction):
    def __init__(self, name, enzyme, substrate, product, vmax, km):
        self.vmax = vmax
        self.km = km
        super().__init__(name, {substrate: 1}, {product: 1}, vmax)
    
    def _generate_rate_law(self):
        S = self.reactants[substrate].symbol
        return self.vmax.symbol * S / (self.km.symbol + S)
```

### System Models
Container for complete biological systems:

```python
model = SystemModel("My System")
model.add_species(species1).add_species(species2)
model.add_parameter(param1).add_parameter(param2)
model.add_reaction(reaction1).add_reaction(reaction2)
```

## Simulation Methods

### Deterministic (ODE) Simulation
Uses scipy's robust ODE solvers:

```python
from pybiodynamics import simulate_ode

result = simulate_ode(
    model,
    t_span=[0, 100],
    method='LSODA'  # or 'RK45', 'Radau', etc.
)
```

### Stochastic (Gillespie) Simulation
Exact stochastic simulation algorithm:

```python
from pybiodynamics import run_gillespie_simulation

result = run_gillespie_simulation(
    model,
    max_iter=100000,
    use_numba=True,  # for performance
    seed=42  # for reproducibility
)
```

## Plotting

### Parameter Sensitivity
```python
from pybiodynamics.simulation.ode import parameter_sweep

results = parameter_sweep(
    model,
    parameter_name='growth_rate',
    parameter_values=[0.1, 0.2, 0.3, 0.4, 0.5],
    t_span=[0, 50]
)
```

### Comparison Plots
```python
from pybiodynamics.simulation.ode import compare_ode_gillespie

compare_ode_gillespie(model, t_span=[0, 30], gillespie_results)
```

### Phase Space Analysis
```python
from pybiodynamics.visualization.plotting import plot_phase_space

plot_phase_space(result, species_x='prey', species_y='predator')
```

## Examples

The `examples/` directory contains complete example scripts:

- `lotka_volterra.ipynb`: Classic predator-prey dynamics with ODE and stochastic simulation
- `logistic_growth_example.ipynb`: Population growth with carrying capacity, also ODE and stochastic examples

