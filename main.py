import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from src.engine.compare_data import compare_data_to_files
from src.utils.config_parser import load_simulation_config
from src.engine.trim_solver import trim_solver
from src.engine.linearization import analyze_mode_shapes, compute_state_space, analyze_eigenvalues, advanced_stability_analysis, plot_linear_response
from src.engine.numerical_integrators import adaptive_integration, fixed_integration
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
    eom, vehicle, amod, meta_cfg, instruction_cfg, output_cfg, compare_cfg, trim_cfg, control_cfg, x0, job_dir, wind_model = load_simulation_config(config_path)
    
    job_name = meta_cfg['job_name']
    
    t0_s, tf_s, dt_s = instruction_cfg['t0_s'], instruction_cfg['tf_s'], instruction_cfg['dt_s']
    
    # 2. Trim & Analysis Dispatch
    x_trim_ref = None
    if instruction_cfg.get('perform_trim', False):
        print("\n--- Executing Trim Solver ---")
        x_trim, x_trim_ref, msg = trim_solver(eom, trim_cfg, x0)
        
        if x_trim is not None:
            x0 = x_trim # Override initial conditions with trim state
            
            if instruction_cfg.get('perform_linearization', False):
                print("\n--- Executing Linearization ---")
                
                state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'lat', 'long', 'h', 'm_fuel', 'dela', 'dele', 'delr', 'delt']
                A, B, core_state_names, core_control_names = compute_state_space(x_trim, x_trim_ref, vehicle, control_cfg, state_names)
                
                analyze_mode_shapes(A, core_state_names)
                plot_linear_response(A, B, t_end=30.0) # Set to 30s to watch the unstable modes diverge
                advanced_stability_analysis(A, B, core_state_names, core_control_names)
    
    # 3. Execution Loop
    if not instruction_cfg.get('simulate', False): return
    
    print("\n--- Running 6-DOF Simulation ---")
    t_s = np.arange(t0_s, tf_s + dt_s, dt_s)
    nt_s = t_s.size
    
    # Route Integrator Configuration
    integrator_type = instruction_cfg.get('integrator', 'RK45')
    adaptive_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
    
    if integrator_type in adaptive_methods:
        t_span = (t0_s, tf_s + dt_s)
        
        t_s, x, aux_data_accum = adaptive_integration(
            eom.solve_eom, t_span, t_s, x0, x_trim_ref,
            method=integrator_type, rtol=1e-6, atol=1e-6
        )
        
    else:
        # Fallback to Fixed-Step RK4
        t_s, x, aux_data_accum = fixed_integration(eom.solve_eom, t_s, dt_s, x0, x_trim_ref)

    # Vectorized Post-Processing
    print("\n--- Post-Processing Data ---")
    
    sim_data = eom.post_process(x, t_s, aux_data_accum, job_name, meta_cfg['description'], integrator_type)
    data_name = 'Simulation'
    # sim_dict = {'name': data_name, 'data': sim_data}

    # 5. Output Management
    if output_cfg:
        
        # Define Job-Specific Directories
        out_dir = os.path.join(job_dir, 'output')
        data_dir = os.path.join(out_dir, "data")
        plot_dir = os.path.join(out_dir, "plots")
        
        # Save Numerical Data
        if output_cfg.get('save_data', False):
            os.makedirs(data_dir, exist_ok=True)
            save_path = os.path.join(data_dir, job_name)
            sim_data.save_npz(save_path + ".npz")
            sim_data.save_csv(save_path + ".csv")
            print(f"\n--- Saving Output ---")
            print(f"Data saved to: {data_dir}")
        else:
            plot_dir = None

        # Dispatch Plots Based on Config Booleans
        plot_cfg = output_cfg.get('plots', {})
        show_plots = output_cfg.get('show_plots', False)
        
        if any(plot_cfg.values()): 
            # Only instantiate plotter and make dir if at least one plot is True
            if output_cfg.get('save_data', False): os.makedirs(plot_dir, exist_ok=True)
            plotter = SimulatorPlotter(sim_data, title_prefix=meta_cfg['description'], plot_dir=plot_dir)
            
            if plot_cfg.get('6dof', False):         plotter.plot_6dof(show=show_plots)
            if plot_cfg.get('attitude', False):     plotter.plot_attitude(show=show_plots)
            if plot_cfg.get('controls', False):     plotter.plot_controls(show=show_plots)
            if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics(show=show_plots)
            if plot_cfg.get('air_data', False):     plotter.plot_air_data(show=show_plots)
            if plot_cfg.get('geodetic', False):     plotter.plot_geodetic(show=show_plots)
            if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity(show=show_plots)
            if plot_cfg.get('forces', False):       plotter.plot_forces(show=show_plots)
            if plot_cfg.get('load_factors', False): plotter.plot_load_factors(show=show_plots)
            if plot_cfg.get('3d_trajectory', False): plotter.plot_3d_trajectory(show=show_plots)
            print(f"Plots saved to: {plot_dir}")
            
            if show_plots:
                print("Displaying plots. Close all plot windows to terminate script.")
                plt.show(block=True)
            else:
                # Force close all background figures before moving to comparison
                plt.close('all')
    
    if instruction_cfg.get('compare', False) and compare_cfg.get('path') is not None:
        compare_data_to_files(sim_data, compare_cfg, job_dir, title_prefix=meta_cfg['description'], wind_model=wind_model)

if __name__ == "__main__":
    # Allows running via command line: python main.py jobs/x15_descending_turn
    input_path = sys.argv[1] if len(sys.argv) > 1 else "jobs/x15_descending_turn"
    
    run_job(input_path)
    # try:
    #     run_job(input_path)
    # except Exception as e:
    #     print(f"Simulation failed: {e}")
    #     sys.exit(1)