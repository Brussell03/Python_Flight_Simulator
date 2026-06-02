import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.engine.compare_data import compare_data_to_file
from src.utils.config_parser import load_simulation_config, resolve_path
from src.engine.trim_solver import trim_solver
from src.engine.linearization import analyze_mode_shapes, compute_state_space, analyze_eigenvalues, advanced_stability_analysis, plot_linear_response
from src.utils.interpolators import fastInterp1
from src.engine.numerical_integrators import RK4, adaptive_integration
from src.utils.plotting import SimulatorPlotter

def run_job(input_path):
    # Resolve the configuration file path
    if os.path.isdir(input_path):
        config_path = os.path.join(input_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Expected 'config.yaml' inside directory: {input_path}")
    else:
        config_path = input_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # 1. Initialization
    eom, vehicle, amod, meta_cfg, instruction_cfg, output_cfg, compare_cfg, trim_cfg, control_cfg, x0, job_dir = load_simulation_config(config_path)
    
    job_name = meta_cfg['job_name']
    
    t0_s, tf_s, dt_s = instruction_cfg['t0_s'], instruction_cfg['tf_s'], instruction_cfg['dt_s']
    state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'lat', 'long', 'h', 'dela', 'dele', 'delr', 'm_fuel']
    
    # 2. Trim & Analysis Dispatch
    u_trim = np.zeros(4)
    if instruction_cfg.get('perform_trim', False):
        print("\n--- Executing Trim Solver ---")
        x_trim, u_trim, msg = trim_solver(eom, vehicle, amod, control_cfg, trim_cfg, x0)
        
        if x_trim is not None:
            x0 = x_trim # Override initial conditions with trim state
            
            if instruction_cfg.get('perform_linearization', False):
                print("\n--- Executing Linearization ---")
                
                state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'lat', 'long', 'h', 'dela', 'dele', 'delr', 'm_fuel']
                
                A, B, core_state_names, core_control_names = compute_state_space(x_trim, u_trim, vehicle, amod, control_cfg, state_names)
                
                analyze_mode_shapes(A, core_state_names)
                plot_linear_response(A, B, t_end=30.0) # Set to 30s to watch the unstable modes diverge
                advanced_stability_analysis(A, B, core_state_names, core_control_names)
    
    # 3. Execution Loop
    print("\n--- Running 6-DOF Simulation ---")
    t_s = np.arange(t0_s, tf_s + dt_s, dt_s)
    nt_s = t_s.size
    
    # Route Integrator Configuration
    integrator_type = instruction_cfg.get('integrator', 'RK45')
    adaptive_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
    
    if integrator_type in adaptive_methods:
        t_span = (t0_s, tf_s + dt_s)
        
        t_s, x, aux_data_accum = adaptive_integration(
            eom.solve_eom, t_span, t_s, x0, vehicle, amod, control_cfg, u_trim, 
            method=integrator_type, rtol=1e-6, atol=1e-6
        )
        
    else:
        # Fallback to Fixed-Step RK4
        print(f"[Fixed-Step {integrator_type} Integration Engine Active]")
        
        x = np.zeros((len(x0), nt_s))
        x[:, 0] = x0
        aux_data_accum = np.zeros((4, nt_s))
        
        dx_tmp = np.empty(x.shape[0], dtype=float)
        aux_tmp = np.empty(4, dtype=float)
        
        for i in tqdm(range(nt_s - 1), desc="Simulating", unit="steps", leave=True):
            t_i = t_s[i]
            x_i = x[:, i]
            
            x[:, i + 1], aux_tmp = RK4(eom.solve_eom, t_i, x_i, dt_s, vehicle, amod, control_cfg, u_trim, dx_tmp, aux_tmp)
            aux_data_accum[:, i + 1] = aux_tmp
            
        # Ensure aux data lines up for t=0
        _, aux_data_accum[:, 0] = eom.solve_eom(t_s[0], x[:, 0], dx_tmp, aux_tmp, vehicle, amod, control_cfg, u_trim)

    # Vectorized Post-Processing
    print("\n--- Post-Processing Data ---")

    #==============================================================================
    # Save the data. Use pointers to automatically accommodate updates to 
    # state and variable indices in plots.
    #==============================================================================
    #
    # Sim_Data column order 
    # 0 t_s 
    # 1 u_b_mps, axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS
    # 2 v_b_mps, lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS
    # 3 w_b_mps, vertical velocity of CM wrt inertial CS resolved in aircraft body fixed CS
    # 4 p_b_rps, roll angular velocity of body fixed CS with respect to inertial CS
    # 5 q_b_rps, pitch angular velocity of body fixed CS with respect to inertial CS
    # 6 r_b_rps, yaw angular velocity of body fixed CS with respect to inertial CS
    # 7 q0 quaternion scalar component
    # 8 q1 quaternion vector i-axis component
    # 9 q2 quaternion vector j-axis component
    # 10 q3 quaternion vector k-axis component
    # 11 lat_rad, geodetic latitude of aircraft resolved in WGS84 CS
    # 12 long_rad, longitude of aircraft resolved in WGS84 CS
    # 13 h_m, altitude of aircraft resolved in WGS84 CS
    # 14 dela_ach_deg, achieved blended aileron position
    # 15 dele_ach_deg, achieved blended elevator position
    # 16 delr_ach_deg, achieved blended rudder position
    # 17 m_fuel_kg, mass of fuel
    # 18 Cs_mps, speed of sound interpolated from atmosphere model
    # 19 Rho_kgpm3, Air density interpolated from atmosphere model
    # 20 Mach, Mach number
    # 21 Alpha_rad, Angle of attack
    # 22 Beta_rad, Angle of sideslip
    # 23 True_Airspeed_mps, True airspeed
    # 24 phi_rad, roll angle
    # 25 theta_rad, pitch angle
    # 26 psi_rad, yaw angle
    # 27 u_n_mps, axial velocity of CM wrt inertial CS resolved in NED CS
    # 28 v_n_mps, lateral velocity of CM wrt inertial CS resolved in NED CS
    # 29 w_n_mps, vertical velocity of CM wrt inertial CS resolved in NED CS
    # 30 dela_cmd_deg, aileron command
    # 31 dele_cmd_deg, elevator command
    # 32 delr_cmd_deg, rudder command
    # 33 delt_percent, throttle as a percent from 0 to 100.
    #
    #==============================================================================
    
    sim_data = eom.post_process(x, t_s, amod, aux_data_accum)

    # 5. Output Management
    if output_cfg:
        
        # Define Job-Specific Directories
        out_dir = os.path.join(job_dir, 'output')
        data_dir = os.path.join(out_dir, "data")
        plot_dir = os.path.join(out_dir, "plots")
        
        # Save Numerical Data
        if output_cfg.get('save_data', False):
            os.makedirs(data_dir, exist_ok=True)
            save_path = os.path.join(data_dir, f"{job_name}.npz")
            meta = {'job_name': job_name}
            np.savez(save_path, data=sim_data, meta=meta)
            print(f"\n--- Saving Output ---")
            print(f"Data saved to: {save_path}")

        # Dispatch Plots Based on Config Booleans
        plot_cfg = output_cfg.get('plots', {})
        show_plots = output_cfg.get('show_plots', False)
        
        if any(plot_cfg.values()): 
            # Only instantiate plotter and make dir if at least one plot is True
            os.makedirs(plot_dir, exist_ok=True)
            plotter = SimulatorPlotter({'name': job_name, 'data': sim_data}, plot_dir=plot_dir)
            
            if plot_cfg.get('6dof', False):         plotter.plot_6dof(show=show_plots)
            if plot_cfg.get('attitude', False):     plotter.plot_attitude(show=show_plots)
            if plot_cfg.get('controls', False):     plotter.plot_controls(show=show_plots)
            if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics(show=show_plots)
            if plot_cfg.get('geodetic', False):     plotter.plot_geodetic(show=show_plots)
            if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity(show=show_plots)
            print(f"Plots saved to: {plot_dir}")
            
            if show_plots:
                print("Displaying plots. Close all plot windows to terminate script.")
                plt.show(block=True)
            else:
                # Force close all background figures before moving to comparison
                plt.close('all')
    
    if instruction_cfg.get('compare', False) and compare_cfg.get('path') is not None:
        show_plots = compare_cfg.get('show_plots', False)
        compare_path = resolve_path(job_dir, compare_cfg.get('path'))
        save_path = os.path.join(job_dir, "output/comparisons/") if compare_cfg.get('save_compare', False) else None
        compare_data_to_file({'name': job_name, 'data': sim_data}, file_path=compare_path, save_dir=save_path, show_plots=show_plots)

if __name__ == "__main__":
    # Allows running via command line: python main.py jobs/x15_descending_turn
    input_path = sys.argv[1] if len(sys.argv) > 1 else "jobs/x15_descending_turn"
    
    try:
        run_job(input_path)
    except Exception as e:
        print(f"Simulation failed: {e}")
        sys.exit(1)