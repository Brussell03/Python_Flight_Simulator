# X-15 6-DOF Flight Dynamics Simulator

## Overview
This repository contains a high-fidelity, 6-Degree-of-Freedom (6-DOF) flight dynamics simulation framework centered on the North American X-15 hypersonic aircraft. The architecture is engineered for advanced stability and control analysis, full-state trajectory integration, and comprehensive numerical linearization across the flight envelope.

## Key Capabilities

* **WGS-84 Ellipsoidal Earth Model:** Incorporates the Somigliana gravity model, local radii of curvature (meridian and prime vertical), transport rates, and Coriolis accelerations to support highly accurate navigation frame (NED) kinematics.
* **Quaternion Kinematics:** Utilizes an Euler parameter (quaternion) formulation for rigid-body rotation to inherently prevent mathematical singularities (gimbal lock) at extreme pitch angles.
* **Non-Linear Trim Solver:** Features a highly constrained `scipy.optimize` (SLSQP) routine for establishing equilibrium flight conditions. Supported modes include steady glide, moment equilibrium, and descending turn coordination.
* **Numerical Linearization & Analysis:** Extracts local bare-airframe linear state-space models ($A$ and $B$ matrices) via central finite differencing. Includes modular routines for eigenvalue extraction, mode shape visualization, and linear initial-condition response simulation via the Python `control` library.
* **XLR99 Engine & Mass Variance:** Couples continuous mass properties (variable inertia tensors) with fuel-burn tracking for the XLR99 rocket engine.
* **Aerodynamic Table Lookups:** Evaluates longitudinal and lateral-directional coefficients using multidimensional interpolation routines mapped against Mach number, angle of attack, and control deflections.

## System Architecture

* `main.py`: Primary execution script handling initialization, loop control, and output dispatch.
* `src/dynamics/eom_wgs84.py`: Core rigid-body equations of motion mapped to the WGS-84 standard.
* `src/engine/trim_solver.py`: Optimization routines enforcing kinematic constraints for algorithmic trimming.
* `src/engine/linearization.py`: Perturbation-based state-space derivation and frequency-domain stability analysis.
* `models/X15/`: Encapsulates the X-15 aerodynamic database, XLR99 engine model, and dynamic mass properties.
* `src/utils/`: Contains data visualization classes (`SimulatorPlotter`), RK4 integrators, and environment constants.

## Dependencies

Execution requires a standard scientific Python environment. Core dependencies include:
* `numpy`
* `scipy`
* `matplotlib`
* `control`

## Execution

Run the primary simulation job via the command line, passing the target configuration file:

```bash
python main.py configs/x15_descending_turn.yaml