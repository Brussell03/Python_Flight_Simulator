# 6-DOF Flight Dynamics Simulator

## Overview
This repository contains a modular, high-fidelity, 6-Degree-of-Freedom (6-DOF) flight dynamics simulation engine. The architecture is engineered for advanced stability and control analysis, full-state trajectory integration, and comprehensive numerical linearization across the flight envelope.

## Key Capabilities

* **Modular Environmental Models:** Decoupled environmental definitions allow for swapping Earth models and wind field models with just the job configuration.
* **WGS-84 Ellipsoidal Earth Model:** Incorporates the J2 gravity model, local radii of curvature (meridian and prime vertical), transport rates, and Coriolis accelerations to support highly accurate navigation frame (NED) kinematics.
* **Quaternion Kinematics:** Utilizes an Euler parameter (quaternion) formulation for rigid-body rotation to inherently prevent mathematical singularities at extreme pitch angles.
* **Non-Linear Trim Solver:** Features a highly constrained `scipy.optimize` (SLSQP) routine for establishing equilibrium flight conditions.
* **Numerical Linearization & Analysis:** Extracts local linear state-space models ($A$ and $B$ matrices) via central finite differencing. Includes modular routines for eigenvalue extraction, mode shape analysis, and linear response modeling.

## System Architecture

* `main.py`: Primary execution script handling initialization, job execution, and output dispatch.
* `src/utils/config_parser.py`: Robustly handles job configuration parsing into standardized simulation instructions and variables.
* `src/engine/eom_solver.py`: Core rigid-body equations of motion.
* `src/engine/trim_solver.py`: Optimization routines enforcing kinematic constraints for algorithmic trimming.
* `src/engine/linearization.py`: Perturbation-based state-space derivation and frequency-domain stability analysis.
* `src/utils/`: Helper modules, unit conversions, plotting classes, and constants.
* `models/`: Vehicle models. Current implementation supports DAVE-ML vehicles as well as custom vehicle implementations.
* `models/X15/`: Encapsulates the X-15 aerodynamic database.

## Dependencies

Execution requires a standard scientific Python environment. Core dependencies include:
* `numpy`
* `scipy`
* `matplotlib`
* `numba`
* `control`
* `pyJanus`

## Execution

The framework is configuration-driven. Execute simulations by passing a directory path containing a config.yaml file to the primary script:

```bash
python main.py jobs/<your_job_directory>
