# jaxkineticmodel documentation — Loading SBML models

Source: https://abeellab.github.io/jaxkineticmodel/SBML/

## SBML loader and simulation

SBML models can be loaded and simulated as follows.

```python
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import pandas as pd

filepath = "models/sbml_models/working_models/simple_sbml.xml"

# load model from file_path
model = SBMLModel(filepath)

# replaces assignment rules, boundary conditions, etc.
S = model._get_stoichiometric_matrix()

JaxKmodel = model.get_kinetic_model()

ts = jnp.linspace(0, 100, 2000)

# simulate given the initial conditions defined in the sbml
ys = JaxKmodel(ts=ts,
               y0=model.y0,
               params=model.parameters)
ys = pd.DataFrame(ys, columns=S.index)
```

Key objects/attributes:
- `SBMLModel(filepath)` — loads and pre-processes an SBML file (resolves
  assignment rules, boundary conditions, etc.)
- `model._get_stoichiometric_matrix()` — returns the stoichiometric matrix S
  (species as index, reactions as columns)
- `model.get_kinetic_model()` — returns a JAX-compiled callable simulator
- `model.y0` — initial species concentrations from the SBML file
- `model.parameters` — parameter dict from the SBML file

## SBML test-suite compatibility

Not all SBML .xml files are compatible/simulatable in jaxkineticmodel — this
depends on the SBML model level and version. Compatibility was benchmarked
against libroadrunner using the sbml-test-suite (500 models, steady-state
models filtered out). Overall, libroadrunner simulates a somewhat higher
fraction of models than jaxkineticmodel, largely because some SBML event
rules are not yet implemented in jaxkineticmodel.

| SBML level/version | jaxkineticmodel similar sims | libroadrunner similar sims | total tested |
|---|---|---|---|
| l2v1 | 169 | 185 | 289 |
| l2v2 | 188 | 219 | 343 |
| l2v3 | 189 | 220 | 343 |
| l2v4 | 191 | 219 | 133 |
| l3v1 | 173 | 197 | 441 |
| l3v2 | 171 | 197 | 460 |

## References
1. Hass, H., Loos, C., Raimundez-Alvarez, E., Timmer, J., Hasenauer, J., & Kreutz, C.
   (2019). Benchmark problems for dynamic modeling of intracellular processes.
   Bioinformatics, 35(17), 3073-3082.
2. Somogyi, E. T., et al. (2015). libRoadRunner: a high performance SBML
   simulation and analysis library. Bioinformatics, 31(20), 3315-3321.
