import sys
import os
import math
import numpy as np

from src.utils.config_parser import load_simulation_config
from src.engine.trim_solver import trim_solver
from src.engine.linearization import analyze_mode_shapes, compute_state_space, analyze_eigenvalues, advanced_stability_analysis, plot_linear_response
from src.dynamics.eom_wgs84 import eom_wgs84
from src.utils.interpolators import fastInterp1
from src.engine.numerical_integrators import RK4
from src.utils.plotting import SimulatorPlotter
from src.utils.constants import D2R, R2D, A_WGS84_M, E_WGS84, OMEGA_E_RPS

def run_job(config_path):
    # 1. Initialization
    vehicle, amod, meta_cfg, instruction_cfg, output_cfg, trim_cfg, control_cfg, x0 = load_simulation_config(config_path)
    
    t0_s, tf_s, dt_s = instruction_cfg['t0_s'], instruction_cfg['tf_s'], instruction_cfg['dt_s']
    state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'lat', 'long', 'h', 'dela', 'dele', 'delr', 'm_fuel']
    
    # 2. Trim & Analysis Dispatch
    u_trim = np.zeros(4)
    if instruction_cfg.get('perform_trim', False):
        print("\n--- Executing Trim Solver ---")
        x_trim, u_trim, msg = trim_solver(vehicle, amod, control_cfg, trim_cfg, x0)
        
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
    x = np.zeros((len(x0), nt_s))
    x[:, 0] = x0
    aux_data_accum = np.zeros((16, nt_s))

    t_s, x, aux_data_accum = RK4(eom_wgs84, t_s, x, dt_s, vehicle, amod, control_cfg, u_trim, aux_data_accum)

    # 4. Vectorized Post-Processing
    print("\n--- Post-Processing Data ---")
    # Pointers so we plot the right data
    sel_state_u_b_mps   = 0
    sel_state_v_b_mps   = 1
    sel_state_w_b_mps   = 2
    sel_state_p_b_rps   = 3
    sel_state_q_b_rps   = 4
    sel_state_r_b_rps   = 5
    sel_state_q0        = 6
    sel_state_q1        = 7
    sel_state_q2        = 8
    sel_state_q3        = 9
    sel_state_lat_rad   = 10
    sel_state_long_rad  = 11
    sel_state_h_m       = 12
    sel_state_dela_ach_deg = 13
    sel_state_dele_ach_deg = 14
    sel_state_delr_ach_deg = 15
    sel_state_m_fuel_kg    = 16
    
    sel_auxillary_data_dela_cmd_deg     = 0
    sel_auxillary_data_dele_cmd_deg     = 1
    sel_auxillary_data_delr_cmd_deg     = 2
    sel_auxillary_data_delt_percent = 3
        
    # Preallocate variables
    Cs_mps      = np.zeros((nt_s,1))
    Rho_kgpm3   = np.zeros((nt_s,1))
    u_n_mps     = np.zeros((nt_s,1)) 
    v_n_mps     = np.zeros((nt_s,1))
    w_n_mps     = np.zeros((nt_s,1)) 
    phi_rad   = np.zeros((nt_s,1)) 
    theta_rad = np.zeros((nt_s,1))
    psi_rad   = np.zeros((nt_s,1))
    
    altitude_m_arr = x[sel_state_h_m, :]
    Cs_mps    = np.array([fastInterp1(amod["alt_m"], amod["c_mps"],     alt) for alt in altitude_m_arr])[:, np.newaxis]
    Rho_kgpm3 = np.array([fastInterp1(amod["alt_m"], amod["rho_kgpm3"], alt) for alt in altitude_m_arr])[:, np.newaxis]

    # Extract Quaternions
    q0 = x[sel_state_q0, :]
    q1 = x[sel_state_q1, :]
    q2 = x[sel_state_q2, :]
    q3 = x[sel_state_q3, :]

    # Compute DCM for all timesteps simultaneously
    C_b2n = np.zeros((len(t_s), 3, 3))
    C_b2n[:, 0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    C_b2n[:, 0, 1] = 2 * (q1*q2 - q0*q3)
    C_b2n[:, 0, 2] = 2 * (q1*q3 + q0*q2)
    C_b2n[:, 1, 0] = 2 * (q1*q2 + q0*q3)
    C_b2n[:, 1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    C_b2n[:, 1, 2] = 2 * (q2*q3 - q0*q1)
    C_b2n[:, 2, 0] = 2 * (q1*q3 - q0*q2)
    C_b2n[:, 2, 1] = 2 * (q2*q3 + q0*q1)
    C_b2n[:, 2, 2] = q0**2 - q1**2 - q2**2 + q3**2

    # Euler Angle Recovery
    phi_rad   = np.arctan2(2 * (q0*q1 + q2*q3), 1 - 2 * (q1**2 + q2**2))[:, np.newaxis]
    theta_rad = np.arcsin(np.clip(2 * (q0*q2 - q3*q1), -1.0, 1.0))[:, np.newaxis]
    psi_rad   = np.arctan2(2 * (q0*q3 + q1*q2), 1 - 2 * (q2**2 + q3**2))[:, np.newaxis]

    # Extract and stack all Body Velocities into an (N, 3, 1) matrix column tensor
    vel_b = np.vstack([
        x[sel_state_u_b_mps, :],
        x[sel_state_v_b_mps, :],
        x[sel_state_w_b_mps, :]
    ]).T[:, :, np.newaxis]

    # Fast Batch Matrix Multiplication
    vel_n = C_b2n @ vel_b

    # Unpack final Nav frame coordinates into (N, 1) column vectors
    u_n_mps = vel_n[:, 0, :]
    v_n_mps = vel_n[:, 1, :]
    w_n_mps = vel_n[:, 2, :]
    
    # Airspeed
    True_Airspeed_mps  = np.zeros((nt_s,1))
    for i, element in enumerate(t_s):
        True_Airspeed_mps[i,0] = math.sqrt(x[sel_state_u_b_mps,i]**2 + x[sel_state_v_b_mps,i]**2 + x[sel_state_w_b_mps,i]**2)
        
    # Angle of attack
    Alpha_rad = np.zeros((nt_s,1))
    for i, element in enumerate(t_s):  
        if x[sel_state_u_b_mps,i] == 0:
            w_over_u = 0
        else:
            w_over_u = x[2,i]/x[sel_state_u_b_mps,i]
        Alpha_rad[i,0] = math.atan(w_over_u)
        
    # Angle of side slip
    Beta_rad = np.zeros((nt_s,1))
    for i, element in enumerate(t_s):  
        if True_Airspeed_mps[i,0] == 0:
            v_over_VT = 0
        else:
            v_over_VT = x[sel_state_v_b_mps,i]/True_Airspeed_mps[i,0]
        Beta_rad[i,0] = math.asin(v_over_VT)
        
    # Mach Number
    Mach = np.zeros((nt_s,1))
    for i, element in enumerate(t_s):
        Mach[i,0] = True_Airspeed_mps[i,0]/Cs_mps[i,0]

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
    # 18 Cs_mps, speed of sound interpolated from atmosphere model (row vector)
    # 19 Rho_kgpm3, Air density interpolated from atmosphere model (row vector)
    # 20 Mach, Mach number (row vector)
    # 21 Alpha_rad, Angle of attack (row vector)
    # 22 Beta_rad, Angle of sideslip (row vector)
    # 23 True_Airspeed_mps, True airspeed (row vector)
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

    # Get auxillary data 
    dela_cmd_deg   = aux_data_accum[sel_auxillary_data_dela_cmd_deg,:]
    dela_cmd_deg   = dela_cmd_deg[:, np.newaxis]
    dele_cmd_deg   = aux_data_accum[sel_auxillary_data_dele_cmd_deg,:]
    dele_cmd_deg   = dele_cmd_deg[:, np.newaxis]
    delr_cmd_deg   = aux_data_accum[sel_auxillary_data_delr_cmd_deg,:]
    delr_cmd_deg   = delr_cmd_deg[:, np.newaxis]
    delt_percent   = aux_data_accum[sel_auxillary_data_delt_percent,:]
    delt_percent   = delt_percent[:, np.newaxis]

    # Combine state date with time and other post processed data
    t_s = t_s[:, np.newaxis]
    sim_data = np.concatenate( (t_s, x.T, Cs_mps, Rho_kgpm3, Mach, Alpha_rad, \
                                Beta_rad, True_Airspeed_mps, phi_rad, theta_rad, psi_rad, \
                                u_n_mps, v_n_mps, w_n_mps, dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, \
                                delt_percent), axis=1 )

    # 5. Output Management
    if output_cfg:
        job_name = meta_cfg['job_name']
        base_out_dir = output_cfg.get('save_dir', './output_data/')
        
        # Define Job-Specific Directories
        job_dir = os.path.join(base_out_dir, job_name)
        data_dir = os.path.join(job_dir, "data")
        plot_dir = os.path.join(job_dir, "plots")
        
        # Save Numerical Data
        if output_cfg.get('save_data', False):
            os.makedirs(data_dir, exist_ok=True)
            save_path = os.path.join(data_dir, f"{job_name}.npy")
            np.save(save_path, sim_data)
            print(f"\n--- Saving Output ---")
            print(f"Data saved to: {save_path}")

        # Dispatch Plots Based on Config Booleans
        plot_cfg = output_cfg.get('figures', {})
        if any(plot_cfg.values()): 
            # Only instantiate plotter and make dir if at least one plot is True
            os.makedirs(plot_dir, exist_ok=True)
            plotter = SimulatorPlotter(sim_data, plot_dir=plot_dir)
            
            if plot_cfg.get('6dof', False):         plotter.plot_6dof()
            if plot_cfg.get('attitude', False):     plotter.plot_attitude()
            if plot_cfg.get('controls', False):     plotter.plot_controls()
            if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics()
            if plot_cfg.get('geodetic', False):     plotter.plot_geodetic()
            if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity()
            print(f"Plots saved to: {plot_dir}")

if __name__ == "__main__":
    # Allows running via command line: python main.py configs/x15_descending_turn.yaml
    target_config = sys.argv[1] if len(sys.argv) > 1 else "configs/x15_descending_turn.yaml"
    run_job(target_config)