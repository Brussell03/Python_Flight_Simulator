import warnings
import numpy as np
import math
from scipy.optimize import minimize

from src.engine.state_mapping import AuxIdx, AuxIdxSlices, StateIdx, StateIdxSlices, TrimStateIdx, TrimStateIdxSlices
from src.utils.constants import D2R, R2D
from src.utils.interpolators import fastInterp1
from src.utils.math_utils import dcm_to_quat, quat_to_dcm

def trim_solver(eom, tmod, x, log_details=False):
    """
    Finds the trimmed flight state by minimizing linear and angular accelerations subject to kinematic constraints.
    """
    
    # Helper Function to Resolve Air-Relative States
    def get_air_relative(x_trim):
        u_b_mps, v_b_mps, w_b_mps = x_trim[TrimStateIdxSlices.VEL_SLICE]
        phi_rad, theta_rad, psi_rad = x_trim[TrimStateIdxSlices.ANGLE_SLICE]
        h_m = x_trim[TrimStateIdx.H_M]
        
        W_N, W_E, W_D = eom.wind_model.get_velocity(h_m)
        
        cphi, sphi = math.cos(phi_rad), math.sin(phi_rad)
        cthe, sthe = math.cos(theta_rad), math.sin(theta_rad)
        cpsi, spsi = math.cos(psi_rad), math.sin(psi_rad)
        
        C_b2n = np.array([
            [cpsi*cthe, cpsi*sthe*sphi - spsi*cphi, cpsi*sthe*cphi + spsi*sphi],
            [spsi*cthe, spsi*sthe*sphi + cpsi*cphi, spsi*sthe*cphi - cpsi*sphi],
            [-sthe,     cthe*sphi,                  cthe*cphi]
        ])
        C_n2b = C_b2n.T
        
        W_b = C_n2b @ np.array([W_N, W_E, W_D])
        
        u_air_b_mps = u_b_mps - W_b[0]
        v_air_b_mps = v_b_mps - W_b[1]
        w_air_b_mps = w_b_mps - W_b[2]
        
        V_T_air_mps = math.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
        alpha_air_mps = math.atan2(w_air_b_mps, u_air_b_mps) if u_air_b_mps != 0 else 0.0
        beta_air_mps = math.asin(np.clip(v_air_b_mps / V_T_air_mps, -1.0, 1.0)) if V_T_air_mps > 0 else 0.0
        
        return V_T_air_mps, alpha_air_mps, beta_air_mps
    
    # Helper Function to get local effective gravity vector
    def get_local_gravity(x_trim):
            lat_rad, long_rad, h_m = x_trim[TrimStateIdxSlices.POS_SLICE]
            x_e_m, y_e_m, z_e_m = eom.earth_model.geodetic_to_ecef(lat_rad, long_rad, h_m)
            
            g_vec = eom.earth_model.get_gravity_ecef(x_e_m, y_e_m, z_e_m)
            return np.linalg.norm(g_vec)

    # Define Internal Optimizer Functions
    def cost_function(x_trim):
        # Unpack optimizer state
        u, v, w = x_trim[TrimStateIdxSlices.VEL_SLICE]
        p, q, r = x_trim[TrimStateIdxSlices.ROT_SLICE]
        phi_rad, theta_rad, psi_rad = x_trim[TrimStateIdxSlices.ANGLE_SLICE]
        lat_rad, long_rad, h_m = x_trim[TrimStateIdxSlices.POS_SLICE]
        dela, dele, delr, delt = x_trim[TrimStateIdxSlices.ACT_TRIM_SLICE]
        m_fuel = x_trim[TrimStateIdx.M_FUEL_KG]

        # 1. Build the Body-to-NED DCM
        cphi, sphi = math.cos(phi_rad), math.sin(phi_rad)
        cthe, sthe = math.cos(theta_rad), math.sin(theta_rad)
        cpsi, spsi = math.cos(psi_rad), math.sin(psi_rad)

        C_b2n = np.array([
            [cpsi*cthe, cpsi*sthe*sphi - spsi*cphi, cpsi*sthe*cphi + spsi*sphi],
            [spsi*cthe, spsi*sthe*sphi + cpsi*cphi, spsi*sthe*cphi - cpsi*sphi],
            [-sthe,     cthe*sphi,                  cthe*cphi]
        ])

        # Geodetic to ECEF Conversion for EOM Compliance
        x_e_m, y_e_m, z_e_m = eom.earth_model.geodetic_to_ecef(lat_rad, long_rad, h_m)

        # Build the local ECEF-to-NED DCM to resolve the true orientation wrt ECEF
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(long_rad), math.cos(long_rad)
        
        C_e2n = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
            [-sin_lon,            cos_lon,            0.0    ],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        
        # Combine transformations and extract the true q_b2e quaternion
        C_b2e = C_e2n.T @ C_b2n
        q0, q1, q2, q3 = dcm_to_quat(C_b2e)

        x_full = np.zeros(StateIdx.NUM_STATES)
        x_full[StateIdx.U_B_MPS]      = u
        x_full[StateIdx.V_B_MPS]      = v
        x_full[StateIdx.W_B_MPS]      = w
        x_full[StateIdx.P_B_RPS]      = p
        x_full[StateIdx.Q_B_RPS]      = q
        x_full[StateIdx.R_B_RPS]      = r
        x_full[StateIdx.Q0]           = q0
        x_full[StateIdx.Q1]           = q1
        x_full[StateIdx.Q2]           = q2
        x_full[StateIdx.Q3]           = q3
        x_full[StateIdx.X_E_M]        = x_e_m
        x_full[StateIdx.Y_E_M]        = y_e_m
        x_full[StateIdx.Z_E_M]        = z_e_m
        x_full[StateIdx.M_FUEL_KG]    = m_fuel
        x_full[StateIdx.DELA_ACH_RAD] = dela
        x_full[StateIdx.DELE_ACH_RAD] = dele
        x_full[StateIdx.DELR_ACH_RAD] = delr
        x_full[StateIdx.DELT_ACH_PCT] = delt
        
        dx = np.empty((StateIdx.NUM_STATES,), dtype=float)
        auxillary_data = np.empty((len(AuxIdx),), dtype=float)
        
        # Call the EOM
        dx, auxillary_data = eom.solve_eom(0, x_full, dx, auxillary_data, None)
        
        # Scale angular accelerations to degree-like magnitudes to maintain optimization gradients
        W_rot = R2D**2
        
        if tmod["trim_mode"] in ['steady_glide', 'straight_and_level']:
            cost = dx[StateIdx.U_B_MPS]**2 + dx[StateIdx.V_B_MPS]**2 + dx[StateIdx.W_B_MPS]**2 + W_rot*(dx[StateIdx.P_B_RPS]**2 + dx[StateIdx.Q_B_RPS]**2 + dx[StateIdx.R_B_RPS]**2)
        elif tmod["trim_mode"] == 'moment_equilibrium':
            cost = W_rot*(dx[StateIdx.P_B_RPS]**2 + dx[StateIdx.Q_B_RPS]**2 + dx[StateIdx.R_B_RPS]**2)
        elif tmod["trim_mode"] == 'descending_turn':
            p_nb_rps, q_nb_rps, r_nb_rps = auxillary_data[AuxIdxSlices.NAV_RATE_SLICE] # auxillary_data[4], auxillary_data[5], auxillary_data[6]
            psidot_current = (q_nb_rps * math.sin(phi_rad) + r_nb_rps * math.cos(phi_rad)) / math.cos(theta_rad)
            # psidot_current = (q * math.sin(phi_rad) + r * math.cos(phi_rad)) / math.cos(theta_rad)
            cost = 0*dx[StateIdx.U_B_MPS]**2 + dx[StateIdx.V_B_MPS]**2 + 0*dx[StateIdx.W_B_MPS]**2 + W_rot*(dx[StateIdx.P_B_RPS]**2 + dx[StateIdx.Q_B_RPS]**2 + dx[StateIdx.R_B_RPS]**2) + 1e1*(psidot_current-psidot_target_rps)**2
        else:
            cost = W_rot*(dx[StateIdx.P_B_RPS]**2 + dx[StateIdx.Q_B_RPS]**2 + dx[StateIdx.R_B_RPS]**2) # Fallback
            
        return cost

    def define_trim_constraints():
        def velocity_constraint(x_trim):
            V_T_air_mps, _, _ = get_air_relative(x_trim)
            return V_T_air_mps - V_T_target_mps
        
        def alpha_constraint(x_trim):
            _, alpha_air_rad, _ = get_air_relative(x_trim)
            return alpha_air_rad - alpha_target_rad
        
        def beta_constraint(x_trim):
            _, _, beta_air_rad = get_air_relative(x_trim)
            return beta_air_rad - beta_target_rad
        
        def roll_rate_constraint(x_trim): return x_trim[TrimStateIdx.P_B_RPS] - p_target_rps
        def pitch_rate_constraint(x_trim): return x_trim[TrimStateIdx.Q_B_RPS] - q_target_rps
        def yaw_rate_constraint(x_trim): return x_trim[TrimStateIdx.R_B_RPS] - r_target_rps
        
        def roll_constraint(x_trim): return x_trim[TrimStateIdx.PHI_RAD] - phi_target_rad
        def pitch_constraint(x_trim): return x_trim[TrimStateIdx.THETA_RAD] - theta_target_rad
        def heading_constraint(x_trim): return x_trim[TrimStateIdx.PSI_RAD] - psi_target_rad
        
        def latitude_constraint(x_trim): return x_trim[TrimStateIdx.LAT_RAD] - lat_target_rad
        def longitude_constraint(x_trim): return x_trim[TrimStateIdx.LONG_RAD] - long_target_rad
        def altitude_constraint(x_trim): return x_trim[TrimStateIdx.H_M] - h_target_m
        
        def position_constraint(x_trim): return x_trim[TrimStateIdx.LAT_RAD] - lat_target_rad + x_trim[TrimStateIdx.LONG_RAD] - long_target_rad + x_trim[TrimStateIdx.H_M] - h_target_m
        def mass_constraint(x_trim): return x_trim[TrimStateIdx.M_FUEL_KG] - m_fuel_target_kg
        
        def flight_path_angle_constraint(x_trim):
            _, alpha_air_rad, _ = get_air_relative(x_trim)
            gamma_current_rad = x_trim[TrimStateIdx.THETA_RAD] - alpha_air_rad
            return gamma_current_rad - gamma_target_rad
        
        def theta_rate_of_climb_constraint(x_trim):
            V_T_current_mps, alpha_current_rad, beta_current_rad = get_air_relative(x_trim)
            
            g_local_mps2 = get_local_gravity(x_trim)
            
            # Note: G relies heavily on air-relative velocity here, assuming wind 
            # magnitude is negligible relative to turn radius scale dynamics
            G = (psidot_target_rps * V_T_current_mps) / g_local_mps2
            a = 1 - G*math.tan(alpha_current_rad)*math.sin(beta_current_rad)
            b = math.sin(gamma_target_rad)/math.cos(beta_current_rad)
            c = 1 + G**2*math.cos(beta_current_rad)**2
            
            # Isolate the term inside the first square root
            sqrt_term_phi = c*(1 - b**2) + G**2 * math.sin(beta_current_rad)**2
            
            # Penalize mathematically impossible states probed by the trim solver
            if sqrt_term_phi < 0: return 1e10
            
            tan_phi_target_rad = G*(math.cos(beta_current_rad)/math.cos(alpha_current_rad)) * \
                ((a - b**2) + b*math.tan(alpha_current_rad)*math.sqrt(sqrt_term_phi)) / \
                (a**2 - b**2*(1 + c*math.tan(alpha_current_rad)**2))
            phi_target_rad = math.atan(tan_phi_target_rad)
            
            a = math.cos(alpha_current_rad)*math.cos(beta_current_rad)
            b = math.sin(phi_target_rad)*math.sin(beta_current_rad)+\
                math.cos(phi_target_rad)*math.sin(alpha_current_rad)*\
                math.cos(beta_current_rad)
            sqrt_term_inside = a**2-math.sin(gamma_target_rad)**2+b**2
            
            if sqrt_term_inside < 0: return 1e10
            
            numerator = a*b+math.sin(gamma_target_rad)*math.sqrt(sqrt_term_inside)
            denominator = a**2-math.sin(gamma_target_rad)**2
            theta_target_rad = math.atan(numerator/denominator)
            return x_trim[TrimStateIdx.THETA_RAD] - theta_target_rad
        
        def phi_turn_coord_constraint(x_trim):
            V_T_current_mps, alpha_current_rad, beta_current_rad = get_air_relative(x_trim)
            
            g_local = get_local_gravity(x_trim)
            
            G = (psidot_target_rps * V_T_current_mps) / g_local
            a = 1 - G*math.tan(alpha_current_rad)*math.sin(beta_current_rad)
            b = math.sin(gamma_target_rad)/math.cos(beta_current_rad)
            c = 1 + G**2*math.cos(beta_current_rad)**2
            
            # Isolate the term inside the first square root
            sqrt_term_phi = c*(1 - b**2) + G**2 * math.sin(beta_current_rad)**2
            
            # Penalize mathematically impossible states probed by the trim solver
            if sqrt_term_phi < 0: return 1e10
            
            tan_phi_target_rad = G*(math.cos(beta_current_rad)/math.cos(alpha_current_rad)) * \
                ((a - b**2) + b*math.tan(alpha_current_rad)*math.sqrt(sqrt_term_phi)) / \
                (a**2 - b**2*(1 + c*math.tan(alpha_current_rad)**2))
            phi_target_rad = math.atan(tan_phi_target_rad)
            return x_trim[TrimStateIdx.PHI_RAD] - phi_target_rad

        if tmod["trim_mode"] == 'steady_glide':
            return [
                {'type': 'eq', 'fun': velocity_constraint},
                {'type': 'eq', 'fun': beta_constraint},
                {'type': 'eq', 'fun': roll_rate_constraint},
                {'type': 'eq', 'fun': pitch_rate_constraint},
                {'type': 'eq', 'fun': yaw_rate_constraint},
                # {'type': 'eq', 'fun': altitude_constraint},
                # {'type': 'eq', 'fun': latitude_constraint},
                # {'type': 'eq', 'fun': longitude_constraint},
                {'type': 'eq', 'fun': roll_constraint},
                {'type': 'eq', 'fun': heading_constraint},
                # {'type': 'eq', 'fun': mass_constraint}
            ]
        elif tmod["trim_mode"] == 'moment_equilibrium':
            return [
                {'type': 'eq', 'fun': velocity_constraint},
                {'type': 'eq', 'fun': beta_constraint},
                {'type': 'eq', 'fun': roll_rate_constraint},
                {'type': 'eq', 'fun': pitch_rate_constraint},
                {'type': 'eq', 'fun': yaw_rate_constraint},
                # {'type': 'eq', 'fun': altitude_constraint},
                {'type': 'eq', 'fun': roll_constraint},
                {'type': 'eq', 'fun': flight_path_angle_constraint},
                # {'type': 'eq', 'fun': latitude_constraint},
                # {'type': 'eq', 'fun': longitude_constraint},
                {'type': 'eq', 'fun': heading_constraint},
                # {'type': 'eq', 'fun': mass_constraint}
            ]
        elif tmod["trim_mode"] == 'descending_turn':
            return [
                {'type': 'eq', 'fun': velocity_constraint},
                # {'type': 'eq', 'fun': altitude_constraint},
                {'type': 'eq', 'fun': theta_rate_of_climb_constraint},
                {'type': 'eq', 'fun': phi_turn_coord_constraint},
                {'type': 'eq', 'fun': roll_rate_constraint},
                # {'type': 'eq', 'fun': latitude_constraint},
                # {'type': 'eq', 'fun': longitude_constraint},
                {'type': 'eq', 'fun': heading_constraint},
                # {'type': 'eq', 'fun': mass_constraint}
            ]
        elif tmod["trim_mode"] == 'straight_and_level':
            return [
                {'type': 'eq', 'fun': velocity_constraint},
                {'type': 'eq', 'fun': beta_constraint},
                {'type': 'eq', 'fun': roll_rate_constraint},
                {'type': 'eq', 'fun': pitch_rate_constraint},
                {'type': 'eq', 'fun': yaw_rate_constraint},
                {'type': 'eq', 'fun': roll_constraint},
                {'type': 'eq', 'fun': flight_path_angle_constraint},
                {'type': 'eq', 'fun': heading_constraint},
            ]
    
    # Setup and Execution
    if log_details: print("--- Unpowered Trim Solver ---")
    
    eom.control_model['trim_flag'] = eom.control_model.get('trim_flag', False) # Defaults to off if missing
    eom.control_model['linearization_flag'] = eom.control_model.get('linearization_flag', False)
    
    # Extract initial guesses from the passed configuration vectors
    lat_current_rad, long_current_rad, h_current_m = eom.earth_model.ecef_to_geodetic(x[StateIdx.X_E_M], x[StateIdx.Y_E_M], x[StateIdx.Z_E_M])
    h_target_m = tmod.get('h_m', h_current_m)
    
    # Cs_mps = fastInterp1(eom.atmo_model["alt_m"], eom.atmo_model["c_mps"], h_current_m)
    # c_snd = fastInterp1(eom.atmo_model['alt_m'], eom.atmo_model['c_mps'], h_target_m)
    Cs_mps = eom.atmo_model.get_soundspeed(h_current_m)
    c_snd = eom.atmo_model.get_soundspeed(h_target_m)
    
    sin_lat, cos_lat = math.sin(lat_current_rad), math.cos(lat_current_rad)
    sin_lon, cos_lon = math.sin(long_current_rad), math.cos(long_current_rad)
    
    C_e2n = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [-sin_lon,            cos_lon,            0.0    ],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])
    
    q0, q1, q2, q3 = x[StateIdxSlices.QUAT_SLICE]
    C_b2e = quat_to_dcm(q0, q1, q2, q3)
    C_e2b = C_b2e.T
    C_n2b = C_e2b @ C_e2n.T
    C_b2n = C_n2b.T
    
    W_N_mps, W_E_mps, W_D_mps = eom.wind_model.get_velocity(h_current_m)
    W_n_mps = np.array([W_N_mps, W_E_mps, W_D_mps])
    W_b_mps = C_n2b @ W_n_mps
    
    phi_current_rad   = np.arctan2(C_b2n[2, 1], C_b2n[2, 2])
    theta_current_rad = np.arcsin(np.clip(-C_b2n[2, 0], -1.0, 1.0))
    psi_current_rad   = np.arctan2(C_b2n[1, 0], C_b2n[0, 0])
    
    u_air_b_mps = x[StateIdx.U_B_MPS] - W_b_mps[0]
    v_air_b_mps = x[StateIdx.V_B_MPS] - W_b_mps[1]
    w_air_b_mps = x[StateIdx.W_B_MPS] - W_b_mps[2]
    
    V_T_current_mps = np.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
    alpha_current_rad = np.arctan2(w_air_b_mps, u_air_b_mps)
    beta_current_rad = np.arcsin(np.clip(v_air_b_mps / V_T_current_mps, -1.0, 1.0))
    Mach_current = V_T_current_mps / Cs_mps
    
    V_T_target_mps   = tmod.get('Mach', Mach_current) * c_snd
    alpha_target_rad = tmod['alpha_deg'] * D2R if tmod.get('alpha_deg') is not None else alpha_current_rad
    beta_target_rad  = tmod['beta_deg'] * D2R if tmod.get('beta_deg') is not None else beta_current_rad
    p_target_rps     = tmod['p_rps'] if tmod.get('p_rps') is not None else x[StateIdx.P_B_RPS]
    q_target_rps     = tmod['q_rps'] if tmod.get('q_rps') is not None else x[StateIdx.Q_B_RPS]
    r_target_rps     = tmod['r_rps'] if tmod.get('r_rps') is not None else x[StateIdx.R_B_RPS]
    phi_target_rad   = tmod['phi_deg'] * D2R if tmod.get('phi_deg') is not None else phi_current_rad
    theta_target_rad = tmod['theta_deg'] * D2R if tmod.get('theta_deg') is not None else theta_current_rad
    psi_target_rad   = tmod['psi_deg'] * D2R if tmod.get('psi_deg') is not None else psi_current_rad
    lat_target_rad   = tmod['lat_deg'] * D2R if tmod.get('lat_deg') is not None else lat_current_rad
    long_target_rad  = tmod['long_deg'] * D2R if tmod.get('long_deg') is not None else long_current_rad
    m_fuel_target_kg = tmod.get('m_fuel_kg', x[StateIdx.M_FUEL_KG])
    dela_target_rad  = tmod['dela_ach_deg'] * D2R if tmod.get('dela_ach_deg') is not None else x[StateIdx.DELA_ACH_RAD]
    dele_target_rad  = tmod['dele_ach_deg'] * D2R if tmod.get('dele_ach_deg') is not None else x[StateIdx.DELE_ACH_RAD]
    delr_target_rad  = tmod['delr_ach_deg'] * D2R if tmod.get('delr_ach_deg') is not None else x[StateIdx.DELR_ACH_RAD]
    delt_target_pct  = tmod['delt_ach_pct'] if tmod.get('delt_ach_pct') is not None else x[StateIdx.DELT_ACH_PCT]
    
    psidot_target_rps = tmod.get('psidot_dps', 0.0) * D2R
    
    # Enforce exact zero targets for Straight & Level, ignoring conflicting inputs
    if tmod.get('trim_mode') == 'straight_and_level':
        gamma_target_rad = 0.0
        phi_target_rad = 0.0
        p_target_rps = 0.0
        q_target_rps = 0.0
        r_target_rps = 0.0
    elif tmod.get('gamma_deg') is not None:
        gamma_target_rad = tmod['gamma_deg'] * D2R
    else:
        gamma_target_rad = theta_target_rad - alpha_target_rad
    
    # Transform Air-Relative Targets to Initial Guess Inertial Velocities
    s_alpha_t = math.sin(alpha_target_rad)
    c_alpha_t = math.cos(alpha_target_rad)
    s_beta_t = math.sin(beta_target_rad)
    c_beta_t = math.cos(beta_target_rad)
    
    u_target_air = c_alpha_t * c_beta_t * V_T_target_mps
    v_target_air = s_beta_t * V_T_target_mps
    w_target_air = s_alpha_t * c_beta_t * V_T_target_mps
    
    W_N_tgt, W_E_tgt, W_D_tgt = eom.wind_model.get_velocity(h_target_m)
    cphi_t, sphi_t = math.cos(phi_target_rad), math.sin(phi_target_rad)
    cthe_t, sthe_t = math.cos(theta_target_rad), math.sin(theta_target_rad)
    cpsi_t, spsi_t = math.cos(psi_target_rad), math.sin(psi_target_rad)
    
    C_b2n_t = np.array([
        [cpsi_t*cthe_t, cpsi_t*sthe_t*sphi_t - spsi_t*cphi_t, cpsi_t*sthe_t*cphi_t + spsi_t*sphi_t],
        [spsi_t*cthe_t, spsi_t*sthe_t*sphi_t + cpsi_t*cphi_t, spsi_t*sthe_t*cphi_t - cpsi_t*sphi_t],
        [-sthe_t,       cthe_t*sphi_t,                        cthe_t*cphi_t]
    ])
    
    W_b_tgt = C_b2n_t.T @ np.array([W_N_tgt, W_E_tgt, W_D_tgt])
    
    u_target_b_mps = u_target_air + W_b_tgt[0]
    v_target_b_mps = v_target_air + W_b_tgt[1]
    w_target_b_mps = w_target_air + W_b_tgt[2]

    # Use current state vector overrided by provided guess parameters
    x_guess = np.zeros(len(TrimStateIdx))
    x_guess[TrimStateIdx.U_B_MPS]      = u_target_b_mps
    x_guess[TrimStateIdx.V_B_MPS]      = v_target_b_mps
    x_guess[TrimStateIdx.W_B_MPS]      = w_target_b_mps
    x_guess[TrimStateIdx.P_B_RPS]      = p_target_rps
    x_guess[TrimStateIdx.Q_B_RPS]      = q_target_rps
    x_guess[TrimStateIdx.R_B_RPS]      = r_target_rps
    x_guess[TrimStateIdx.PHI_RAD]      = phi_target_rad
    x_guess[TrimStateIdx.THETA_RAD]    = theta_target_rad
    x_guess[TrimStateIdx.PSI_RAD]      = psi_target_rad
    x_guess[TrimStateIdx.LAT_RAD]      = lat_target_rad
    x_guess[TrimStateIdx.LONG_RAD]     = long_target_rad
    x_guess[TrimStateIdx.H_M]          = h_target_m
    x_guess[TrimStateIdx.M_FUEL_KG]    = m_fuel_target_kg
    x_guess[TrimStateIdx.DELA_TRIM_RAD] = dela_target_rad
    x_guess[TrimStateIdx.DELE_TRIM_RAD] = dele_target_rad
    x_guess[TrimStateIdx.DELR_TRIM_RAD] = delr_target_rad
    x_guess[TrimStateIdx.DELT_TRIM_PCT] = delt_target_pct
    
    if log_details: 
        print("\nTrim guess state:")
        print(f"u_b_mps:   {x_guess[TrimStateIdx.U_B_MPS]:.8f}")
        print(f"v_b_mps:   {x_guess[TrimStateIdx.V_B_MPS]:.8f}")
        print(f"w_b_mps:   {x_guess[TrimStateIdx.W_B_MPS]:.8f}")
        print(f"p_dps:     {x_guess[TrimStateIdx.P_B_RPS]*R2D:.8f}")
        print(f"q_dps:     {x_guess[TrimStateIdx.Q_B_RPS]*R2D:.8f}")
        print(f"r_dps:     {x_guess[TrimStateIdx.R_B_RPS]*R2D:.8f}")
        print(f"phi_deg:   {x_guess[TrimStateIdx.PHI_RAD]*R2D:.8f}")
        print(f"theta_deg: {x_guess[TrimStateIdx.THETA_RAD]*R2D:.8f}")
        print(f"psi_deg:   {x_guess[TrimStateIdx.PSI_RAD]*R2D:.8f}")
        print(f"lat_deg:   {x_guess[TrimStateIdx.LAT_RAD]*R2D:.8f}")
        print(f"long_deg:  {x_guess[TrimStateIdx.LONG_RAD]*R2D:.8f}")
        print(f"alt_m:     {x_guess[TrimStateIdx.H_M]:.8f}")
        print(f"m_fuel_kg: {x_guess[TrimStateIdx.M_FUEL_KG]:.8f}")
        print(f"dela_deg:  {x_guess[TrimStateIdx.DELA_TRIM_RAD]*R2D:.8f}")
        print(f"dele_deg:  {x_guess[TrimStateIdx.DELE_TRIM_RAD]*R2D:.8f}")
        print(f"delr_deg:  {x_guess[TrimStateIdx.DELR_TRIM_RAD]*R2D:.8f}")
        print(f"delt_pct:  {x_guess[TrimStateIdx.DELT_TRIM_PCT]:.8f}")
    
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Values in x were outside bounds during a minimize step")

    bounds = [(-np.inf, np.inf)] * (len(TrimStateIdx))
    bounds[TrimStateIdx.THETA_RAD]  = (-math.pi/3, math.pi/3)
    
    bounds[TrimStateIdx.LAT_RAD]  = (lat_target_rad, lat_target_rad)      # Lock Latitude
    bounds[TrimStateIdx.LONG_RAD] = (long_target_rad, long_target_rad)    # Lock Longitude
    bounds[TrimStateIdx.H_M] = (h_target_m, h_target_m)                   # Lock Altitude
    bounds[TrimStateIdx.M_FUEL_KG] = (m_fuel_target_kg, m_fuel_target_kg)  # Lock Mass
    
    # Needs to be gotten from vehicle
    bounds[TrimStateIdx.DELA_TRIM_RAD] = (-15*D2R, 15*D2R)
    bounds[TrimStateIdx.DELE_TRIM_RAD] = (-35*D2R, 15*D2R)
    bounds[TrimStateIdx.DELR_TRIM_RAD] = (-7.5*D2R, 7.5*D2R)
    bounds[TrimStateIdx.DELT_TRIM_PCT] = (0, 100)
    
    eom.control_model["trim_flag"] = True

    if log_details: print("\nSolving for trim state...")
    result = minimize(
        fun = cost_function,
        x0 = x_guess,
        method = 'SLSQP',
        bounds = bounds,
        constraints = define_trim_constraints(),
        options={'disp': log_details, 'ftol': 1e-9, 'maxiter': 500}
    )
    
    eom.control_model["trim_flag"] = False
    
    # Process Results
    x_trim = result.x
    phi_rad = x_trim[TrimStateIdx.PHI_RAD]
    theta_rad = x_trim[TrimStateIdx.THETA_RAD]
    psi_rad = x_trim[TrimStateIdx.PSI_RAD]
    lat_rad = x_trim[TrimStateIdx.LAT_RAD]
    long_rad = x_trim[TrimStateIdx.LONG_RAD]
    h_m = x_trim[TrimStateIdx.H_M]
    
    # Transformation for final reconstruction block
    cphi, sphi = math.cos(phi_rad), math.sin(phi_rad)
    cthe, sthe = math.cos(theta_rad), math.sin(theta_rad)
    cpsi, spsi = math.cos(psi_rad), math.sin(psi_rad)

    C_b2n = np.array([
        [cpsi*cthe, cpsi*sthe*sphi - spsi*cphi, cpsi*sthe*cphi + spsi*sphi],
        [spsi*cthe, spsi*sthe*sphi + cpsi*cphi, spsi*sthe*cphi - cpsi*sphi],
        [-sthe,     cthe*sphi,                  cthe*cphi]
    ])

    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(long_rad), math.cos(long_rad)
    
    C_e2n = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [-sin_lon,            cos_lon,            0.0    ],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])
    
    C_b2e = C_e2n.T @ C_b2n
    q0, q1, q2, q3 = dcm_to_quat(C_b2e)
    
    dx = np.empty((StateIdx.NUM_STATES,), dtype=float)
    auxillary_data = np.empty((len(AuxIdx),), dtype=float)
    
    # Construct verified ECEF array for final solve output
    x_e_m, y_e_m, z_e_m = eom.earth_model.geodetic_to_ecef(lat_rad, long_rad, h_m)

    x_trim_full = np.zeros(StateIdx.NUM_STATES)
    x_trim_full[StateIdx.U_B_MPS]      = x_trim[TrimStateIdx.U_B_MPS]
    x_trim_full[StateIdx.V_B_MPS]      = x_trim[TrimStateIdx.V_B_MPS]
    x_trim_full[StateIdx.W_B_MPS]      = x_trim[TrimStateIdx.W_B_MPS]
    x_trim_full[StateIdx.P_B_RPS]      = x_trim[TrimStateIdx.P_B_RPS]
    x_trim_full[StateIdx.Q_B_RPS]      = x_trim[TrimStateIdx.Q_B_RPS]
    x_trim_full[StateIdx.R_B_RPS]      = x_trim[TrimStateIdx.R_B_RPS]
    x_trim_full[StateIdx.Q0]           = q0
    x_trim_full[StateIdx.Q1]           = q1
    x_trim_full[StateIdx.Q2]           = q2
    x_trim_full[StateIdx.Q3]           = q3
    x_trim_full[StateIdx.X_E_M]        = x_e_m
    x_trim_full[StateIdx.Y_E_M]        = y_e_m
    x_trim_full[StateIdx.Z_E_M]        = z_e_m
    x_trim_full[StateIdx.M_FUEL_KG]    = x_trim[TrimStateIdx.M_FUEL_KG]
    x_trim_full[StateIdx.DELA_ACH_RAD] = x_trim[TrimStateIdx.DELA_TRIM_RAD]
    x_trim_full[StateIdx.DELE_ACH_RAD] = x_trim[TrimStateIdx.DELE_TRIM_RAD]
    x_trim_full[StateIdx.DELR_ACH_RAD] = x_trim[TrimStateIdx.DELR_TRIM_RAD]
    x_trim_full[StateIdx.DELT_ACH_PCT] = x_trim[TrimStateIdx.DELT_TRIM_PCT]

    x_trim_ref = x_trim.copy()
    
    # Wind Engine Interface
    W_N_mps, W_E_mps, W_D_mps = eom.wind_model.get_velocity(h_m)
    
    W_n_mps = np.array([W_N_mps, W_E_mps, W_D_mps])
    W_b_mps = C_n2b @ W_n_mps

    # Air-Relative Translational States
    u_air_b_mps = x[StateIdx.U_B_MPS] - W_b_mps[0]
    v_air_b_mps = x[StateIdx.V_B_MPS] - W_b_mps[1]
    w_air_b_mps = x[StateIdx.W_B_MPS] - W_b_mps[2]
    
    # rho_kgpm3 = fastInterp1(eom.atmo_model["alt_m"], eom.atmo_model["rho_kgpm3"], h_m)
    rho_kgpm3 = eom.atmo_model.get_density(h_m)
    true_airspeed_mps = math.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
    
    alpha_rad = math.atan2(w_air_b_mps, u_air_b_mps)
    beta_rad = math.asin(v_air_b_mps / true_airspeed_mps) if true_airspeed_mps > 0 else 0.0
    
    # Extract the dynamic SAS commands reacting to the trimmed body rates
    eom.vehicle.set_gnc_inputs(0, eom.control_model, eom.atmo_model, lat_rad, long_rad, h_m, alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, x[StateIdx.P_B_RPS], x[StateIdx.Q_B_RPS], x[StateIdx.R_B_RPS], true_airspeed_mps, rho_kgpm3, x_trim_ref)
    dela_dyn_rad = eom.vehicle.roll_control(0, x_trim[TrimStateIdx.P_B_RPS], x_trim[TrimStateIdx.R_B_RPS], eom.control_model)
    dele_dyn_rad = eom.vehicle.pitch_control(0, x_trim[TrimStateIdx.Q_B_RPS], eom.control_model)
    delr_dyn_rad = eom.vehicle.yaw_control(0, x_trim[TrimStateIdx.R_B_RPS], eom.control_model)
    delt_dyn_pct = eom.vehicle.throttle_control(0, eom.control_model)
    
    # Subtract the SAS contribution to isolate the true baseline pilot trim
    x_trim_ref[TrimStateIdx.DELA_TRIM_RAD] -= dela_dyn_rad
    x_trim_ref[TrimStateIdx.DELE_TRIM_RAD] -= dele_dyn_rad
    x_trim_ref[TrimStateIdx.DELR_TRIM_RAD] -= delr_dyn_rad
    x_trim_ref[TrimStateIdx.DELT_TRIM_PCT] -= delt_dyn_pct
    
    dx, auxillary_data = eom.solve_eom(0, x_trim_full, dx, auxillary_data, x_trim_ref)

    if result.success:
        
        # Calculate proper Euler Rates using kinematic equations
        p_nb_rps, q_nb_rps, r_nb_rps = auxillary_data[AuxIdxSlices.NAV_RATE_SLICE]
        
        phi_rad_dot   = p_nb_rps + q_nb_rps * math.sin(phi_rad) * math.tan(theta_rad) + r_nb_rps * math.cos(phi_rad) * math.tan(theta_rad)
        theta_rad_dot = q_nb_rps * math.cos(phi_rad) - r_nb_rps * math.sin(phi_rad)
        psi_rad_dot   = (q_nb_rps * math.sin(phi_rad) + r_nb_rps * math.cos(phi_rad)) / math.cos(theta_rad)
        
        if log_details:
            print(f"Trim Successful! Cost function value: {result.fun:.3e}")
            print("-" * 25)
            print(f"VT_mps:    {math.sqrt(x_trim[TrimStateIdx.U_B_MPS]**2 + x_trim[TrimStateIdx.V_B_MPS]**2 + x_trim[TrimStateIdx.W_B_MPS]**2):.8f}")
            print(f"alpha_deg: {math.atan2(x_trim[TrimStateIdx.W_B_MPS], x_trim[TrimStateIdx.U_B_MPS])*R2D:.8f}")
            print(f"beta_deg:  {math.asin(x_trim[1]/math.sqrt(x_trim[TrimStateIdx.U_B_MPS]**2 + x_trim[TrimStateIdx.V_B_MPS]**2 + x_trim[TrimStateIdx.W_B_MPS]**2))*R2D:.8f}")
            print(f"p_dps:     {x_trim[TrimStateIdx.P_B_RPS]*R2D:.8f}")
            print(f"q_dps:     {x_trim[TrimStateIdx.Q_B_RPS]*R2D:.8f}")
            print(f"r_dps:     {x_trim[TrimStateIdx.R_B_RPS]*R2D:.8f}")
            print(f"phi_deg:   {phi_rad*R2D:.8f}")
            print(f"theta_deg: {theta_rad*R2D:.8f}")
            print(f"psi_deg:   {psi_rad*R2D:.8f}")
            print(f"lat_deg:   {x_trim[TrimStateIdx.LAT_RAD]*R2D:.8f}")
            print(f"long_deg:  {x_trim[TrimStateIdx.LONG_RAD]*R2D:.8f}")
            print(f"alt_m:     {x_trim[TrimStateIdx.H_M]:.8f}")
            print(f"m_fuel_kg: {x_trim[TrimStateIdx.M_FUEL_KG]:.8f}")
            print(f"dela_deg:  {x_trim[TrimStateIdx.DELA_TRIM_RAD]*R2D:.8f}")
            print(f"dele_deg:  {x_trim[TrimStateIdx.DELE_TRIM_RAD]*R2D:.8f}")
            print(f"delr_deg:  {x_trim[TrimStateIdx.DELR_TRIM_RAD]*R2D:.8f}")
            print(f"delt_pct:  {x_trim[TrimStateIdx.DELT_TRIM_PCT]:.8f}")
            print(" ")
            print(f"u_b_mps-dot:      {dx[StateIdx.U_B_MPS]: .8f}")
            print(f"v_b_mps-dot:      {dx[StateIdx.V_B_MPS]: .8f}")
            print(f"w_b_mps-dot:      {dx[StateIdx.W_B_MPS]: .8f}")
            print(f"p_b_dps-dot:      {dx[StateIdx.P_B_RPS]*R2D: .8f}")
            print(f"q_b_dps-dot:      {dx[StateIdx.Q_B_RPS]*R2D: .8f}")
            print(f"r_b_dps-dot:      {dx[StateIdx.R_B_RPS]*R2D: .8f}")
            print(f"phi_deg-dot:      {phi_rad_dot*R2D: .8f}")
            print(f"theta_deg-dot:    {theta_rad_dot*R2D: .8f}")
            print(f"psi_deg-dot:      {psi_rad_dot*R2D: .8f}")
            print(f"x_e_m-dot:        {dx[StateIdx.X_E_M]: .8f}")
            print(f"y_e_m-dot:        {dx[StateIdx.Y_E_M]: .8f}")
            print(f"z_e_m-dot:        {dx[StateIdx.Z_E_M]: .8f}")
            print(f"m_fuel_kg-dot:    {dx[StateIdx.M_FUEL_KG]: .8f}")
            print(f"dela_ach_deg-dot: {dx[StateIdx.DELA_ACH_RAD]*R2D: .8f}")
            print(f"dele_ach_deg-dot: {dx[StateIdx.DELE_ACH_RAD]*R2D: .8f}")
            print(f"delr_ach_deg-dot: {dx[StateIdx.DELR_ACH_RAD]*R2D: .8f}")
            print(f"delt_ach_pct-dot: {dx[StateIdx.DELT_ACH_PCT]: .8f}")
            print(" ")
            print(f"dela_trim_deg: {x_trim_full[StateIdx.DELA_ACH_RAD]: .8f}")
            print(f"dele_trim_deg: {x_trim_full[StateIdx.DELE_ACH_RAD]: .8f}")
            print(f"delr_trim_deg: {x_trim_full[StateIdx.DELR_ACH_RAD]: .8f}")
            print(f"delt_trim_deg: {x_trim_full[StateIdx.DELT_ACH_PCT]: .8f}")
        
        return x_trim_full, x_trim_ref, result.message
    else:
        if log_details:
            print(f"\n!!! TRIM FAILED TO CONVERGE !!!")
            print(f"Message: {result.message}")
        return None, None, result.message