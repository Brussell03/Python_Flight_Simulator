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
    vehicle, amod, cmod, sim_params, ic, x0, x_guess, u_guess, analysis, config = load_simulation_config(config_path)
    
    t0_s, tf_s, dt_s = sim_params['t0_s'], sim_params['tf_s'], sim_params['dt_s']
    state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'lat', 'long', 'h', 'dela', 'dele', 'delr', 'm_fuel']
    
    # 2. Trim & Analysis Dispatch
    if analysis.get('perform_trim', False):
        print("\n--- Executing Trim Solver ---")
        x_trim, u_trim, msg = trim_solver(vehicle, amod, cmod, analysis, x_guess, u_guess, ic['psidot_target_dps'] * D2R)
        
        if x_trim is not None:
            x0 = x_trim # Override initial conditions with trim state
            
            if analysis.get('perform_linearization', False):
                print("\n--- Executing Linearization ---")
                
                state_names = ['u', 'v', 'w', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3', 'lat', 'long', 'h', 'dela', 'dele', 'delr', 'm_fuel']
                
                A, B, core_state_names, core_control_names = compute_state_space(x_trim, u_trim, vehicle, amod, cmod, state_names)
                
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

    t_s, x, aux_data_accum = RK4(eom_wgs84, t_s, x, dt_s, vehicle, amod, cmod, aux_data_accum)

    # 4. Vectorized Post-Processing
    print("--- Post-Processing Data ---")
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
    
    sel_auxillary_data_dela_deg     = 0
    sel_auxillary_data_dele_deg     = 1
    sel_auxillary_data_delr_deg     = 2
    sel_auxillary_data_delt_percent = 3
        
    # Preallocate variables
    Cs_mps      = np.zeros((nt_s,1))
    Rho_kgpm3   = np.zeros((nt_s,1))
    C_phi       = np.zeros((nt_s,1))
    C_theta     = np.zeros((nt_s,1))
    C_psi       = np.zeros((nt_s,1))
    S_phi       = np.zeros((nt_s,1))
    S_theta     = np.zeros((nt_s,1))
    S_psi       = np.zeros((nt_s,1))
    T_theta     = np.zeros((nt_s,1))
    C_b2n_11    = np.zeros((nt_s,1))
    C_b2n_12    = np.zeros((nt_s,1))
    C_b2n_13    = np.zeros((nt_s,1))
    C_b2n_21    = np.zeros((nt_s,1))
    C_b2n_22    = np.zeros((nt_s,1))
    C_b2n_23    = np.zeros((nt_s,1))
    C_b2n_31    = np.zeros((nt_s,1))
    C_b2n_32    = np.zeros((nt_s,1))
    C_b2n_33    = np.zeros((nt_s,1))
    u_n_mps     = np.zeros((nt_s,1)) 
    v_n_mps     = np.zeros((nt_s,1))
    w_n_mps     = np.zeros((nt_s,1)) 
    phi_2_rad   = np.zeros((nt_s,1)) 
    theta_2_rad = np.zeros((nt_s,1))
    psi_2_rad   = np.zeros((nt_s,1)) 
    Transport_Rate_rps = np.zeros((nt_s,1))
    Coriolis_mps2      = np.zeros((nt_s,1))
    North_dist_m       = np.zeros((nt_s,1))
    East_dist_m        = np.zeros((nt_s,1))
    
    for i, element in enumerate(t_s):
        Altitude_m       = x[sel_state_h_m,i]
        Cs_mps[i,0]      = fastInterp1(amod["alt_m"], amod["c_mps"],     Altitude_m)
        Rho_kgpm3[i,0]   = fastInterp1(amod["alt_m"], amod["rho_kgpm3"], Altitude_m)
        
        # Extract Quaternion
        q0, q1, q2, q3 = x[sel_state_q0,i], x[sel_state_q1,i], x[sel_state_q2,i], x[sel_state_q3,i]
        
        # Compute DCM from Quaternion
        C_b2n_11[i,0] = q0**2 + q1**2 - q2**2 - q3**2
        C_b2n_12[i,0] = 2 * (q1*q2 - q0*q3)
        C_b2n_13[i,0] = 2 * (q1*q3 + q0*q2)
        C_b2n_21[i,0] = 2 * (q1*q2 + q0*q3)
        C_b2n_22[i,0] = q0**2 - q1**2 + q2**2 - q3**2
        C_b2n_23[i,0] = 2 * (q2*q3 - q0*q1)
        C_b2n_31[i,0] = 2 * (q1*q3 - q0*q2)
        C_b2n_32[i,0] = 2 * (q2*q3 + q0*q1)
        C_b2n_33[i,0] = q0**2 - q1**2 - q2**2 + q3**2
        
        # Recover Euler Angles for Plots
        phi_2_rad[i,0]   = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        theta_2_rad[i,0] = math.asin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
        psi_2_rad[i,0]   = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        
        # Backfill legacy trig arrays for output consistency
        C_phi[i,0], S_phi[i,0]     = math.cos(phi_2_rad[i,0]), math.sin(phi_2_rad[i,0])
        C_theta[i,0], S_theta[i,0] = math.cos(theta_2_rad[i,0]), math.sin(theta_2_rad[i,0])
        C_psi[i,0], S_psi[i,0]     = math.cos(psi_2_rad[i,0]), math.sin(psi_2_rad[i,0])
        T_theta[i,0]               = math.tan(theta_2_rad[i,0])
        
        u_b_mps = x[sel_state_u_b_mps,i]
        v_b_mps = x[sel_state_v_b_mps,i]
        w_b_mps = x[sel_state_w_b_mps,i]
        
        u_n_mps[i,0]     =  C_b2n_11[i,0]*u_b_mps + C_b2n_12[i,0]*v_b_mps + C_b2n_13[i,0]*w_b_mps
        v_n_mps[i,0]     =  C_b2n_21[i,0]*u_b_mps + C_b2n_22[i,0]*v_b_mps + C_b2n_23[i,0]*w_b_mps
        w_n_mps[i,0]     =  C_b2n_31[i,0]*u_b_mps + C_b2n_32[i,0]*v_b_mps + C_b2n_33[i,0]*w_b_mps
        
        # WGS-84 Local parameters
        lat_rad = x[sel_state_lat_rad, i]
        long_rad = x[sel_state_long_rad, i]
        den = math.sqrt(1.0 - (E_WGS84 * math.sin(lat_rad))**2)
        RN = A_WGS84_M / den
        RM = (A_WGS84_M * (1.0 - E_WGS84**2)) / (den**3)
        
        # WGS-84 Cartesian Trajectory Reconstruction
        North_dist_m[i,0] = (lat_rad - x[sel_state_lat_rad,0]) * RM
        East_dist_m[i,0]  = (long_rad - x[sel_state_long_rad,0]) * RN * math.cos(x[sel_state_lat_rad,0])
        
        # WGS-84 Kinematic Quantities
        v_N, v_E, v_D = u_n_mps[i,0], v_n_mps[i,0], w_n_mps[i,0]
        
        omega_en_N = v_E / (RN + Altitude_m)
        omega_en_E = -v_N / (RM + Altitude_m)
        omega_en_D = -v_E * math.tan(lat_rad) / (RN + Altitude_m)
        Transport_Rate_rps[i,0] = math.sqrt(omega_en_N**2 + omega_en_E**2 + omega_en_D**2)

        omega_ie_N = OMEGA_E_RPS * math.cos(lat_rad)
        omega_ie_E = 0.0
        omega_ie_D = -OMEGA_E_RPS * math.sin(lat_rad)

        cor_N = 2 * (omega_ie_E * v_D - omega_ie_D * v_E)
        cor_E = 2 * (omega_ie_D * v_N - omega_ie_N * v_D)
        cor_D = 2 * (omega_ie_N * v_E - omega_ie_E * v_N)
        Coriolis_mps2[i,0] = math.sqrt(cor_N**2 + cor_E**2 + cor_D**2)
    
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
    # 7 phi_rad, roll angle
    # 8 theta_rad, pitch angle
    # 9 psi_rad, yaw angle
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
    # 24 C_phi, cosine of roll angle (row vector)
    # 25 C_theta, cosine of pitch angle (row vector)
    # 26 C_psi, cosine of yaw angle (row vector)
    # 27 S_phi, sine of roll angle (row vector)
    # 28 S_theta, sine of pitch angle (row vector)
    # 29 S_psi, sine of yaw angle (row vector)
    # 30 T_theta, tangent of pitch angle (row vector)
    # 31 phi_2_rad, roll angle
    # 32 theta_2_rad, pitch angle
    # 33 psi_2_rad, yaw angle
    # 34 u_n_mps, axial velocity of CM wrt inertial CS resolved in NED CS
    # 35 v_n_mps, lateral velocity of CM wrt inertial CS resolved in NED CS
    # 36 w_n_mps, vertical velocity of CM wrt inertial CS resolved in NED CS
    # 37 dela_deg, aileron command
    # 38 dele_deg, elevator command
    # 39 delr_deg, rudder command
    # 40 delt_percent, throttle as a percent from 0 to 100.
    #
    #==============================================================================

    # Get auxillary data 
    dela_deg        = aux_data_accum[sel_auxillary_data_dela_deg,:]
    dela_deg        = dela_deg[:, np.newaxis]
    dele_deg        = aux_data_accum[sel_auxillary_data_dele_deg,:]
    dele_deg        = dele_deg[:, np.newaxis]
    delr_deg        = aux_data_accum[sel_auxillary_data_delr_deg,:]
    delr_deg        = delr_deg[:, np.newaxis]
    delt_percent    = aux_data_accum[sel_auxillary_data_delt_percent,:]
    delt_percent    = delt_percent[:, np.newaxis]

    # Combine state date with time and other post processed data
    t_s = t_s[:, np.newaxis]
    sim_data = np.concatenate( (t_s, x.T, Cs_mps, Rho_kgpm3, Mach, Alpha_rad, \
                                Beta_rad, True_Airspeed_mps, C_phi, C_theta, C_psi, S_phi, \
                                S_theta, S_psi, T_theta, phi_2_rad, theta_2_rad, psi_2_rad, \
                                u_n_mps, v_n_mps, w_n_mps, dela_deg, dele_deg, delr_deg, \
                                delt_percent), axis=1 )

    # 5. Output Management
    output_cfg = config.get('output', {})
    if output_cfg:
        job_name = config['meta']['job_name']
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
            print(f"\n--- Output Saved ---")
            print(f"Data saved to: {save_path}")

        # Dispatch Plots Based on Config Booleans
        plot_cfg = output_cfg.get('figures', {})
        if any(plot_cfg.values()): 
            # Only instantiate plotter and make dir if at least one plot is True
            os.makedirs(plot_dir, exist_ok=True)
            plotter = SimulatorPlotter(sim_data, plot_dir=plot_dir)
            
            print(f"Generating plots to: {plot_dir}")
            if plot_cfg.get('6dof', False):         plotter.plot_6dof()
            if plot_cfg.get('attitude', False):     plotter.plot_attitude()
            if plot_cfg.get('controls', False):     plotter.plot_controls()
            if plot_cfg.get('aerodynamics', False): plotter.plot_aerodynamics()
            if plot_cfg.get('geodetic', False):     plotter.plot_geodetic()
            if plot_cfg.get('ned_velocity', False): plotter.plot_ned_velocity()

if __name__ == "__main__":
    # Allows running via command line: python main.py configs/x15_descending_turn.yaml
    target_config = sys.argv[1] if len(sys.argv) > 1 else "configs/x15_descending_turn.yaml"
    run_job(target_config)