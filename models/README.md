# Model files used by the tests and examples

Files here are found by name, like those in `data/sbml_models/` (which is searched first).

| File | Format | What it is |
|---|---|---|
| `pdp_mm_bistable.txt` | XPP-AUT | Phosphorylation--dephosphorylation cycle with Michaelis--Menten kinetics, with parameter sets for bistability, a pitchfork and bistability with a transcritical point (the active set is the uncommented `par` block); bifurcation parameter `ATOT` |
| `saddle_node.xml` | SBML | Hill-type autoregulation with a saddle-node (fold) |
| `transcritical.xml` | SBML | Michaelis--Menten autoregulation with a transcritical point |
| `pitchfork.xml` | SBML | Mutual inhibition of two proteins, with a pitchfork |
| `pitchfork_model.xml` | SBML | A reduced phosphorylation--dephosphorylation model (A, APP) with a pitchfork in `ATOT` |
| `mm_reduced.ode` | XPP-AUT | Reduced Michaelis--Menten model (inflow, conversion with the enzyme in quasi-steady state, outflow), for stochastic tests |
| `ferrell_cell_cycle_1d.ode`, `_2d.ode`, `_3d.ode` | XPP-AUT | Ferrell, Tsai and Yang (2011) CDK1/APC cell-cycle models in one, two and three variables (only the three-variable one oscillates) |

The published models (BioModels entries and the models built from papers) are in `data/sbml_models/`.
