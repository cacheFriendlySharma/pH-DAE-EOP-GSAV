# pH-DAE-EOP-GSAV


Code and numerical experiments accompanying the paper

**A Generalized Scalar Auxiliary Variable Method for Structure-Preserving and Efficient Integration of Nonlinear port-Hamiltonian DAEs**

by Aashutosh Sharma, Andreas Bartel, and Manuel Schaller.

## Overview

This repository contains the implementation and numerical experiments for an
energy-optimal generalized scalar auxiliary variable (EOP-GSAV) framework for
nonlinear index-one port-Hamiltonian differential-algebraic equations
(pH-DAEs).

The proposed BDF1 and BDF2 EOP-GSAV schemes separate the state-dependent
nonlinear terms from a constant implicit core. For fixed step size and method
order, the resulting linear system can be factorized once and reused throughout
the integration, requiring one linear solve per time step.

The numerical experiments investigate

- temporal convergence,
- auxiliary-energy and Hamiltonian behavior,
- discrete passivity and algebraic-constraint accuracy,
- robustness under strongly state-dependent nonlinearities,
- comparison with implicit midpoint using full, modified, and frozen-Jacobian
  Newton iterations,
- matched-accuracy work-precision behavior,
- problem-size scaling,
- sensitivity to nonlinear strength,
- comparison with SUNDIALS IDA.

## Repository structure
.
├── benchmark/
│   └── results/
│       ├── convergence/
│       ├── energy_tracking/
│       ├── ida/
│       ├── nonlinearity_strength/
│       ├── size_scaling/
│       ├── structure/
│       ├── toy_nonlinear_solve/
│       └── work_precision/
├── scripts/
│   ├── convergence.py
│   ├── energy_tracking.py
│   ├── ida_gsav_structure.py
│   ├── ida_order2_diagnostic.py
│   ├── ida_work_precision.py
│   ├── matched_accuracy.py
│   ├── nonlinearity_strength.py
│   ├── size_scaling.py
│   ├── structure.py
│   ├── toy_nonlinear_solve.py
│   └── work_precision.py
├── src/
│   ├── implicit_midpoint.py
│   ├── inputs.py
│   ├── linear_solvers.py
│   ├── newton_solvers.py
│   ├── plotting_config.py
│   ├── potentials.py
│   ├── system.py
│   └── time_integrators.py
├── LICENSE
└── README.md


src/ contains the numerical methods and benchmark model implementation,
while scripts/ contains the experiments reported in the paper.
The corresponding generated data and plots are stored under benchmark/results/, including
the low-dimensional nonlinear stress test in toy_nonlinear_solve/.

## Requirements

The main experiments require

- Python 3
- NumPy
- SciPy
- Matplotlib

The comparison with SUNDIALS IDA additionally requires

- sundials4py 7.8.0

The SUNDIALS dependency is required only for the IDA experiments.

## Running the experiments

Run individual experiments from the repository root, for example:

python scripts/convergence.py
python scripts/energy_tracking.py
python scripts/work_precision.py
python scripts/size_scaling.py


The corresponding results are written to subdirectories of

text
benchmark/results/


For the SUNDIALS comparison:

bash
python scripts/ida_work_precision.py
python scripts/ida_order2_diagnostic.py
python scripts/ida_gsav_structure.py


## Benchmark

The main benchmark is a nonlinear FPU-beta/Maxwell mass-spring-damper chain
formulated as an index-one port-Hamiltonian DAE. It combines nonlinear energy
storage, state-dependent interconnection, nonlinear dissipation, algebraic
constraints, and external forcing.

A separate low-dimensional nonlinear stress test is used to examine solver
robustness when the Jacobian varies strongly along the trajectory.

## Reproducibility

The scripts use the parameter choices, time-step sizes, tolerances, and solver
configurations reported in the paper.

Reported wall-clock measurements are based on repeated complete integrations
and use the median runtime. The primary numerical experiments are single-core
measurements. The SUNDIALS IDA experiments are performed separately in the
environment described in the paper.

For detailed descriptions of the algorithms, analysis, benchmark construction,
and computational environment, please refer to the accompanying paper.

## Citation

If you use this code, please cite

bibtex
@article{sharma2026eopgsav,
  title   = {A generalized scalar auxiliary variable method for
             structure-preserving and efficient integration of nonlinear
             port-Hamiltonian DAEs},
  author  = {Sharma, Aashutosh and Bartel, Andreas and Schaller, Manuel},
  year    = {2026}
}


The citation information will be updated when the paper is publicly available.

## License

This repository is released under the MIT License. See LICENSE.
