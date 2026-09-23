# OLED drift-diffusion simulation

A one-dimensional Python example for steady-state charge transport and recombination in a multilayer OLED. It solves coupled hole and electron continuity equations with Poisson electrostatics, field-dependent mobility, interfacial barriers, and contact injection. Given the carrier solution, it computes local steady singlet and triplet populations and exports device-physics profiles and figures.

This repository contains one illustrative five-layer stack with parameters embedded in `Device.__init__`. The values are an example for exploring the numerical method; the code does not represent a calibrated panel or a general device-design package.

## Install

Python 3.10 or newer is recommended. From the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run an example

```bash
python recombination.py
```

By default the program sweeps −0.4 to 2.0 V in 0.1 V steps and finds a voltage near 3 mA/cm² if the converged sweep brackets that current. Output is written to `results/`:

| File | Contents |
| --- | --- |
| `JV.csv` | Voltage and current density in mA/cm² |
| `diagnostics.json` | Solver settings, convergence and current-continuity checks |
| `target_profile.csv` | Spatial profile at the requested current, when bracketed |
| `last_voltage_profile.csv` | Spatial profile at the last sweep voltage |
| `sweep.npz` and profile `.npz` files | Scaled carrier states and numerical arrays |
| `summary_dashboard.png` | Six-panel summary of J–V, carrier density, electric field, recombination zone, singlets, and triplets |
| Individual `.png` figures | Separate plots for each quantity shown in the dashboard |

For a quicker single-voltage check:

```bash
python recombination.py --vmin 1.4 --vmax 1.4 --output results
```

The target-current search needs at least two sweep points unless a point already matches the target. A warning about an unbracketed target in the single-voltage example is expected; the last-voltage profile is still exported. Use `--no-plot` for batch or headless runs that only need numerical output.

## Model and numerical choices

- Position is discretized in 1 nm cells by default. Layer thicknesses are `[60, 5, 30, 5, 31]` nm, and the nominal EML spans 65–95 nm. Change `Device.__init__` to explore other stacks and material parameters.
- Carrier densities are internally scaled by `1e17 cm^-3`. SciPy's BDF integrator advances the carrier continuity equations to a checked DC state. A voltage sweep reuses the preceding state as its initial condition.
- Current spread and normalized carrier residuals are reported for each voltage. A solution reaching `max_time` without satisfying both criteria is explicitly marked `steady: false`.
- Singlet and triplet populations are solved locally after carrier convergence because this example does not feed excitons back into the charge equations. `full_rhs` is available for coupled carrier/exciton transient integration under those same one-way coupling assumptions.
- The default `--poisson legacy` preserves the original discrete voltage quadrature. `--poisson trapezoid` enforces the corresponding trapezoidal discrete voltage integral. `--triplet-loss corrected` uses a loss proportional to triplet density; `--triplet-loss legacy` reproduces the original expression for comparison.

Units are cm, s, V, cm⁻³, and A/cm² internally, with output current density in mA/cm². `target_profile.csv` labels each exported column and its units.

## Validation

```bash
python validate.py
```

This checks algebraic agreement with an independent translation of the original positive-density RHS, field and contact edge cases, particle conservation, grid and voltage quadrature, a 25-point DC sweep, target-current search, exciton residuals, solver tolerance sensitivity, transient equivalence, and Python timing comparisons. It writes `validation/validation_report.json`, numerical fixtures, and sweep states. The timing comparison is between Python solver configurations; it is not a measured comparison with MATLAB or `ode15s`. The full run can take several minutes.
