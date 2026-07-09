import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.engine.config_parser import load_job_config, load_initial_state
from src.engine.trim_solver import trim_solver
from src.engine.numerical_integrators import adaptive_integration, fixed_integration
from src.engine.mc_analysis import process_mc_results

def mc_worker_task(config_path, run_id):
    """
    Isolated worker for a single Monte Carlo dispersion run.
    Returns a scalar KPI dictionary to prevent IPC serialization bloat.
    """
    try:
        config_payload = load_job_config(config_path)
        if not config_payload:
            return {'run_id': run_id, 'status': 'FAILED: Invalid YAML'}

        eom, meta_cfg, instruction_cfg, output_cfg, init_cond_cfg, trim_cfg, job_dir = config_payload
        
        # Trigger dispersions by passing nominal=False and injecting the run_id as the seed
        x0 = load_initial_state(
            init_cond_cfg, eom.atmo_model, eom.earth_model, 
            log_details=False, nominal=False, seed=run_id
        )

        # 1. Trim 
        x_trim_ref = None
        if instruction_cfg.get('perform_trim', False):
            x_trim, x_trim_ref, _ = trim_solver(eom, trim_cfg, x0, log_details=False)
            if x_trim is not None:
                x0 = x_trim 

        # 2. Simulate
        simulation_cfg = instruction_cfg.get('simulation', {})
        if not simulation_cfg.get('enabled', False):
            return {'run_id': run_id, 'status': 'SKIPPED: Sim Disabled'}

        t0_s, tf_s, dt_s = simulation_cfg['t0_s'], simulation_cfg['tf_s'], simulation_cfg['dt_s']
        t_s = np.arange(t0_s, tf_s + dt_s, dt_s)
        integrator_type = instruction_cfg.get('integrator', 'RK45')
        adaptive_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

        if integrator_type in adaptive_methods:
            t_span = (t0_s, tf_s + dt_s)
            t_s, x, aux_data_accum = adaptive_integration(
                eom.solve_eom, t_span, t_s, x0, x_trim_ref,
                method=integrator_type, rtol=1e-6, atol=1e-6
            )
        else:
            t_s, x, aux_data_accum = fixed_integration(eom.solve_eom, t_s, dt_s, x0, x_trim_ref)

        # 3. Process & Extract KPIs
        # We process the data, but we DO NOT return the sim_data object back to the main thread.
        sim_data = eom.post_process(x, t_s, aux_data_accum, "MC_Run", "MC", integrator_type)

        kpi = {
            'run_id': run_id,
            'status': 'SUCCESS',
            'final_lat_deg': sim_data.lat_deg[-1],
            'final_long_deg': sim_data.long_deg[-1],
            'final_alt_m': sim_data.h_m[-1],
            'max_mach': float(np.max(sim_data.mach)),
            'max_nz': float(np.max(np.abs(sim_data.n_z)))
        }
        return kpi

    except Exception as e:
        return {'run_id': run_id, 'status': f'FAILED: {type(e).__name__}'}


def run_monte_carlo_suite(config_path, mc_cfg):
    """
    Orchestrates the Monte Carlo batch for a single configuration.
    """
    n_runs = mc_cfg.get('num_runs', 100)
    max_workers = mc_cfg.get('num_cores', max(1, (os.cpu_count() or 2) - 1))
    job_name = os.path.basename(config_path)

    print(f"\n{'='*60}")
    print(f"MONTE CARLO SUITE: {job_name} ({n_runs} Runs | {max_workers} Cores)")
    print(f"{'='*60}")

    results = []

    # Rich Progress Bar for clean terminal tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        
        task = progress.add_task(f"[cyan]Executing dispersed trajectories...", total=n_runs)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Queue all iterations
            futures = [executor.submit(mc_worker_task, config_path, i) for i in range(n_runs)]

            # Process as they complete to update the progress bar
            for future in as_completed(futures):
                results.append(future.result())
                progress.advance(task)

    # Summary
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if 'FAILED' in r['status']]
    
    print(f"Suite Complete: {len(successful)} Successful | {len(failed)} Failed")
    
    # Re-extract job_dir and job_name for file routing
    config_payload = load_job_config(config_path)
    if config_payload:
        _, meta_cfg, _, _, _, _, job_dir = config_payload
        job_name = meta_cfg['job_name']
        
        process_mc_results(results, mc_cfg, job_dir, job_name)
    
    return results