
import sys
import os
import concurrent
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from rich.live import Live
from rich.table import Table
from rich import box

from src.utils.compare_data import compare_data_to_files
from src.engine.config_parser import load_job_config
from src.engine.trim_solver import trim_solver
from src.engine.linearization import execute_linearization_commands
from src.engine.numerical_integrators import adaptive_integration, fixed_integration
from src.utils.plotting import SimulatorPlotter

def run_job(config_path, disable_plots=False, log_details=True, status_dict=None):
    """
    Executes a single simulation job. 
    If disable_plots is True, overrides configuration to suppress Matplotlib blocking calls.
    status_dict: A multiprocessing.Manager().dict() for IPC status tracking.
    """
    # Helper to push updates to the main process
    def update_status(msg):
        if status_dict is not None:
            # Extract just the filename for cleaner display
            short_name = os.path.basename(config_path)
            status_dict[short_name] = msg
    
    # ==========================================
    # 1. Initialization
    # ==========================================
    update_status("Loading Configuration...")
    config_payload = load_job_config(config_path, log_details)
    
    # Validation trap for invalid YAML payloads
    if not config_payload:
        update_status("FAILED: Invalid YAML")
        raise ValueError(f"YAML parsing failed or configuration is invalid for: {config_path}")
        
    eom, meta_cfg, instruction_cfg, output_cfg, trim_cfg, x0, job_dir = config_payload
    job_name = meta_cfg['job_name']
    
    if log_details: print(f"\n{'='*60}\n--- Starting Job: {job_name} ---\n{'='*60}")
    
    # ==========================================
    # 2. Initial Trim
    # ==========================================
    x_trim_ref = None
    if instruction_cfg.get('perform_trim', False):
        update_status("Executing Trim Solver...")
        if log_details: print("\n--- Executing Trim Solver ---")
        x_trim, x_trim_ref, msg = trim_solver(eom, trim_cfg, x0, log_details)
        
        if x_trim is not None:
            x0 = x_trim # Override initial conditions with trim state
    
    # ==========================================
    # 3. Simulation Loop
    # ==========================================
    simulation_cfg = instruction_cfg.get('simulation', {})
    if simulation_cfg.get('enabled', False):
        update_status("Running 6-DOF Integration...")
        
        t0_s, tf_s, dt_s = simulation_cfg['t0_s'], simulation_cfg['tf_s'], simulation_cfg['dt_s']
        
        if log_details: print("\n--- Running 6-DOF Simulation ---")
        t_s = np.arange(t0_s, tf_s + dt_s, dt_s)
        
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
        if log_details: print("\n--- Post-Processing Data ---")
        
        update_status("Post-Processing Data...")
        sim_data = eom.post_process(x, t_s, aux_data_accum, job_name, meta_cfg['description'], integrator_type)

        # --- Output ---
        if output_cfg:
            update_status("Saving Output & Rendering Plots...")
            
            # Define Job-Specific Directories
            out_dir = os.path.join(job_dir, f'output/{job_name}')
            data_dir = os.path.join(out_dir, "data")
            plot_dir = os.path.join(out_dir, "plots")
            
            # Save Numerical Data
            if output_cfg.get('save_data', False):
                os.makedirs(data_dir, exist_ok=True)
                save_path = os.path.join(data_dir, job_name)
                sim_data.save_npz(save_path + ".npz")
                sim_data.save_csv(save_path + ".csv")
                if log_details: print(f"\n--- Saving Output ---")
                if log_details: print(f"Data saved to: {data_dir}")
            else:
                plot_dir = None

            # Dispatch Plots Based on Config Booleans
            plot_cfg = output_cfg.get('plots', {})
            
            # Force plot suppression if running in a parallel batch
            show_plots = False if disable_plots else output_cfg.get('show_plots', False)
            
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
                if log_details: print(f"Plots saved to: {plot_dir}")
                
                if show_plots:
                    if log_details: print("Displaying plots. Close all plot windows to terminate script.")
                    plt.show(block=True)
                else:
                    if log_details: print(f"Plots rendered and saved to: {plot_dir} (Display Suppressed)")
                    # Force close all background figures before moving to comparison and prevent memory leaks in worker processes
                    plt.close('all')
    
    # ==========================================
    # 4. Analysis
    # ==========================================
    analysis_cfg = instruction_cfg.get('analysis', {})
    
    # Comaprison
    compare_cfg = analysis_cfg.get('comparison', {})
    if compare_cfg.get('enabled', False) and compare_cfg.get('path') is not None:
        update_status("Running Comparison...")
        if log_details: print(f"\n--- Running Comparison ---")
        compare_data_to_files(sim_data, compare_cfg, job_dir, title_prefix=meta_cfg['description'], wind_model=eom.wind_model)
    
    # Linearization
    linearization_cfg = analysis_cfg.get('linearization', {})
    if linearization_cfg.get('enabled', False):
        update_status("Executing Linearization...")
        if log_details: print("\n--- Executing Linearization ---")
        
        # Suppress plotting in the downstream linearization suite if in batch mode
        if disable_plots:
            linearization_cfg['show_plots'] = False
        
        state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'x', 'y', 'z', 'm_fuel', 'dela', 'dele', 'delr', 'delt']
        control_names = ['dela', 'dele', 'delr', 'delt']
        
        execute_linearization_commands(linearization_cfg, eom, 0, x0, x_trim_ref, state_names, control_names, log_details)
    
    update_status("COMPLETED")
    return True

if __name__ == "__main__":
    # Allows running via command line: python main.py jobs/
    input_path = sys.argv[1] if len(sys.argv) > 1 else "jobs/x15_descending_turn"
    
    configs_to_run = []

    # 1. Path Resolution and File Discovery
    if os.path.isfile(input_path):
        if input_path.endswith('.yaml'):
            configs_to_run.append(input_path)
        else:
            raise ValueError(f"Provided file is not a .yaml extension: {input_path}")
    elif os.path.isdir(input_path):
        # The user provided a directory; recursively find all config.yaml files
        print(f"Scanning directory for configurations: {input_path}")
        for root, dirs, files in os.walk(input_path):
            for file in files:
                # Target all files terminating in .yaml
                if file.endswith(".yaml"):
                    configs_to_run.append(os.path.join(root, file))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # 2. Execution Queue Validation
    if not configs_to_run:
        print(f"Execution Aborted: No 'config.yaml' files found at or beneath '{input_path}'")
        sys.exit(0)

    num_jobs = len(configs_to_run)
    is_batch = num_jobs > 1
    
    print(f"Found {num_jobs} configuration file(s).")
    
    successful_jobs = 0
    failed_jobs = []

    # 3. Execution Dispatcher
    if is_batch:
        print("Batch mode activated. Plot rendering will be disabled to protect system resources.")
        print("Starting multiprocessing pool...\n")
        
        max_workers = max(1, (os.cpu_count() or 2) - 1)
        
        manager = Manager()
        status_dict = manager.dict()
        
        def generate_table():
            """Dynamically rebuilds the rich Table based on the current status_dict."""
            table = Table(
                title=f"Batch Execution Dashboard ({successful_jobs + len(failed_jobs)}/{num_jobs} Complete)",
                box=box.SIMPLE_HEAVY,
                header_style="bold cyan"
            )
            table.add_column("Configuration", width=35, no_wrap=True)
            table.add_column("Current Status", width=40)

            for path in configs_to_run:
                short_name = os.path.basename(path)
                status = status_dict.get(short_name, "Queued...")
                
                # Apply dynamic color coding based on status keywords
                if "COMPLETED" in status: 
                    color = "[bold green]"
                elif "FAILED" in status: 
                    color = "[bold red]"
                elif "Queued" in status: 
                    color = "[dim]"
                else: 
                    color = "[yellow]"
                
                table.add_row(short_name, f"{color}{status}[/]")
            return table

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_job, cfg_path, True, False, status_dict): cfg_path 
                for cfg_path in configs_to_run
            }
            
            # The Live context manager handles the terminal redrawing cleanly without spam
            with Live(generate_table(), refresh_per_second=4, transient=False) as live:
                while futures:
                    done, not_done = concurrent.futures.wait(
                        futures, timeout=0.25, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        cfg_path = futures.pop(future)
                        try:
                            future.result() 
                            successful_jobs += 1
                            # Force a final status update in case the worker missed it
                            short_name = os.path.basename(cfg_path)
                            status_dict[short_name] = "COMPLETED"
                        except Exception as e:
                            short_name = os.path.basename(cfg_path)
                            status_dict[short_name] = f"FAILED: {type(e).__name__}"
                            failed_jobs.append((cfg_path, str(e)))
                    
                    # Update the live display with the new state
                    live.update(generate_table())

    else:
        # Single file execution (Sequential)
        cfg_path = configs_to_run[0]
        run_job(cfg_path, disable_plots=False)
        successful_jobs += 1
        # try:
        #     run_job(cfg_path, disable_plots=False)
        #     successful_jobs += 1
        # except Exception as e:
        #     print(f"\n[ERROR] Simulation failed for {cfg_path}: {e}")
        #     failed_jobs.append((cfg_path, str(e)))

    # 4. Batch Summary Report
    print(f"\n{'='*60}")
    print(f"BATCH EXECUTION COMPLETE")
    print(f"Total Jobs: {len(configs_to_run)} | Successful: {successful_jobs} | Failed: {len(failed_jobs)}")
    
    if failed_jobs:
        print("\nFailed Configurations:")
        for failed_cfg, error_msg in failed_jobs:
            print(f" - {failed_cfg}: {error_msg}")
        sys.exit(1) # Exit with error code if any jobs failed