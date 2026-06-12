import warnings
import numpy as np
import math
from scipy.optimize import minimize

from src.utils.constants import D2R, NUM_AUX, NUM_STATE, R2D
from src.utils.interpolators import fastInterp1
from src.utils.kinematics import dcm_to_quat, quat_to_dcm

def trim_solver(eom, vehicle, amod, cmod, tmod, wmod, x):
    """
    Finds the trimmed flight state by minimizing angular accelerations subject to kinematic constraints.
    """
    
    # Helper Function to Resolve Air-Relative States
    def get_air_relative(x_trim):
        u_b_mps, v_b_mps, w_b_mps = x_trim[0:3]
        phi_rad, theta_rad, psi_rad = x_trim[6:9]
        h_m = x_trim[11]
        
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
            lat_rad, long_rad, h_m = x_trim[9:12]
            x_e, y_e, z_e = eom.earth_model.wgs84_to_cartesian(lat_rad, long_rad, h_m)
            
            g_vec = eom.earth_model.get_gravity_ecef(x_e, y_e, z_e)
            return np.linalg.norm(g_vec)

    # Define Internal Optimizer Functions
    def cost_function(x_trim, vehicle, cmod):
        # Unpack optimizer state
        u, v, w = x_trim[0:3]
        p, q, r = x_trim[3:6]
        phi_rad, theta_rad, psi_rad = x_trim[6:9]

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
        lat_rad, long_rad, h_m = x_trim[9:12]
        x_e, y_e, z_e = eom.earth_model.geodetic_to_ecef(lat_rad, long_rad, h_m)

        # 2. Build the local ECEF-to-NED DCM to resolve the true orientation wrt ECEF
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(long_rad), math.cos(long_rad)
        
        C_e2n = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
            [-sin_lon,            cos_lon,            0.0    ],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        
        # 3. Combine transformations and extract the true q_b2e quaternion
        C_b2e = C_e2n.T @ C_b2n
        q0, q1, q2, q3 = dcm_to_quat(C_b2e)

        x_full = np.array([
            u, v, w, p, q, r, q0, q1, q2, q3, 
            x_e, y_e, z_e, 
            x_trim[12], x_trim[13], x_trim[14], x_trim[15], x_trim[16]
        ], dtype=float)
        
        dx = np.empty((NUM_STATE,), dtype=float)
        auxillary_data = np.empty((NUM_AUX,), dtype=float)
        
        # Call the EOM
        dx, auxillary_data = eom.solve_eom(0, x_full, dx, auxillary_data, None, vehicle, cmod)
        
        # Scale angular accelerations to degree-like magnitudes to maintain optimization gradients
        W_rot = R2D**2
        
        if tmod["trim_mode"] in ['steady_glide', 'straight_and_level']:
            cost = dx[0]**2 + dx[1]**2 + dx[2]**2 + W_rot*(dx[3]**2 + dx[4]**2 + dx[5]**2)
        elif tmod["trim_mode"] == 'moment_equilibrium':
            cost = W_rot*(dx[3]**2 + dx[4]**2 + dx[5]**2)
        elif tmod["trim_mode"] == 'descending_turn':
            p_nb_rps, q_nb_rps, r_nb_rps = auxillary_data[4], auxillary_data[5], auxillary_data[6]
            psidot_current = (q_nb_rps * math.sin(phi_rad) + r_nb_rps * math.cos(phi_rad)) / math.cos(theta_rad)
            # psidot_current = (q * math.sin(phi_rad) + r * math.cos(phi_rad)) / math.cos(theta_rad)
            cost = 0*dx[0]**2 + dx[1]**2 + 0*dx[2]**2 + W_rot*(dx[3]**2 + dx[4]**2 + dx[5]**2) + 1e1*(psidot_current-psidot_target_rps)**2
        else:
            cost = W_rot*(dx[3]**2 + dx[4]**2 + dx[5]**2) # Fallback
            
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
        
        def roll_rate_constraint(x_trim): return x_trim[3] - p_target_rps
        def pitch_rate_constraint(x_trim): return x_trim[4] - q_target_rps
        def yaw_rate_constraint(x_trim): return x_trim[5] - r_target_rps
        
        def roll_constraint(x_trim): return x_trim[6] - phi_target_rad
        def pitch_constraint(x_trim): return x_trim[7] - theta_target_rad
        def heading_constraint(x_trim): return x_trim[8] - psi_target_rad
        
        def latitude_constraint(x_trim): return x_trim[9] - lat_target_rad
        def longitude_constraint(x_trim): return x_trim[10] - long_target_rad
        def altitude_constraint(x_trim): return x_trim[11] - h_target_m
        
        def position_constraint(x_trim): return x_trim[9] - lat_target_rad + x_trim[10] - long_target_rad + x_trim[11] - h_target_m
        def mass_constraint(x_trim): return x_trim[12] - m_fuel_target_kg
        
        def flight_path_angle_constraint(x_trim):
            _, alpha_air_rad, _ = get_air_relative(x_trim)
            gamma_current_rad = x_trim[7] - alpha_air_rad
            return gamma_current_rad - gamma_target_rad
        
        def theta_rate_of_climb_constraint(x_trim):
            V_T_current_mps, alpha_current_rad, beta_current_rad = get_air_relative(x_trim)
            # gamma_current_rad = x_trim[7] - alpha_current_rad
            
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
            return x_trim[7] - theta_target_rad
        
        def phi_turn_coord_constraint(x_trim):
            V_T_current_mps, alpha_current_rad, beta_current_rad = get_air_relative(x_trim)
            # gamma_current_rad = x_trim[7] - alpha_current_rad
            
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
            return x_trim[6] - phi_target_rad

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
    print("--- Unpowered Trim Solver ---")
    
    cmod['trim_flag'] = cmod.get('trim_flag', False) # Defaults to off if missing
    cmod['linearization_flag'] = cmod.get('linearization_flag', False)
    
    # Extract initial guesses from the passed configuration vectors
    lat_current_rad, long_current_rad, h_current_m = eom.earth_model.ecef_to_geodetic(x[10], x[11], x[12])
    h_target_m = tmod.get('h_m', h_current_m)
    
    Cs_mps = fastInterp1(amod["alt_m"], amod["c_mps"], h_current_m)
    c_snd = fastInterp1(amod['alt_m'], amod['c_mps'], h_target_m)
    
    sin_lat, cos_lat = math.sin(lat_current_rad), math.cos(lat_current_rad)
    sin_lon, cos_lon = math.sin(long_current_rad), math.cos(long_current_rad)
    
    C_e2n = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [-sin_lon,            cos_lon,            0.0    ],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])
    
    q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
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
    
    u_air_b_mps = x[0] - W_b_mps[0]
    v_air_b_mps = x[1] - W_b_mps[1]
    w_air_b_mps = x[2] - W_b_mps[2]
    
    V_T_current_mps = np.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
    alpha_current_rad = np.arctan2(w_air_b_mps, u_air_b_mps)
    beta_current_rad = np.arcsin(np.clip(v_air_b_mps / V_T_current_mps, -1.0, 1.0))
    Mach_current = V_T_current_mps / Cs_mps
    
    V_T_target_mps   = tmod.get('Mach', Mach_current) * c_snd
    alpha_target_rad = tmod['alpha_deg'] * D2R if tmod.get('alpha_deg') is not None else alpha_current_rad
    beta_target_rad  = tmod['beta_deg'] * D2R if tmod.get('beta_deg') is not None else beta_current_rad
    p_target_rps     = tmod['p_rps'] if tmod.get('p_rps') is not None else x[3]
    q_target_rps     = tmod['q_rps'] if tmod.get('q_rps') is not None else x[4]
    r_target_rps     = tmod['r_rps'] if tmod.get('r_rps') is not None else x[5]
    phi_target_rad   = tmod['phi_deg'] * D2R if tmod.get('phi_deg') is not None else phi_current_rad
    theta_target_rad = tmod['theta_deg'] * D2R if tmod.get('theta_deg') is not None else theta_current_rad
    psi_target_rad   = tmod['psi_deg'] * D2R if tmod.get('psi_deg') is not None else psi_current_rad
    lat_target_rad   = tmod['lat_deg'] * D2R if tmod.get('lat_deg') is not None else lat_current_rad
    long_target_rad  = tmod['long_deg'] * D2R if tmod.get('long_deg') is not None else long_current_rad
    m_fuel_target_kg = tmod.get('m_fuel_kg', x[13])
    dela_target_rad  = tmod['dela_ach_deg'] * D2R if tmod.get('dela_ach_deg') is not None else x[14]
    dele_target_rad  = tmod['dele_ach_deg'] * D2R if tmod.get('dele_ach_deg') is not None else x[15]
    delr_target_rad  = tmod['delr_ach_deg'] * D2R if tmod.get('delr_ach_deg') is not None else x[16]
    delt_target_pct  = tmod['delt_ach_pct'] if tmod.get('delt_ach_pct') is not None else x[17]
    
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
    x_guess = np.zeros(NUM_STATE - 1)
    x_guess[0]  = u_target_b_mps
    x_guess[1]  = v_target_b_mps
    x_guess[2]  = w_target_b_mps
    x_guess[3]  = p_target_rps
    x_guess[4]  = q_target_rps
    x_guess[5]  = r_target_rps
    x_guess[6]  = phi_target_rad
    x_guess[7]  = theta_target_rad
    x_guess[8]  = psi_target_rad
    x_guess[9]  = lat_target_rad
    x_guess[10] = long_target_rad
    x_guess[11] = h_target_m
    x_guess[12] = m_fuel_target_kg
    x_guess[13] = dela_target_rad
    x_guess[14] = dele_target_rad
    x_guess[15] = delr_target_rad
    x_guess[16] = delt_target_pct
    
    print("\nTrim guess state:")
    print(f"u_b_mps:   {x_guess[0]:.8f}")
    print(f"v_b_mps:   {x_guess[1]:.8f}")
    print(f"w_b_mps:   {x_guess[2]:.8f}")
    print(f"p_dps:     {x_guess[3]*R2D:.8f}")
    print(f"q_dps:     {x_guess[4]*R2D:.8f}")
    print(f"r_dps:     {x_guess[5]*R2D:.8f}")
    print(f"phi_deg:   {x_guess[6]*R2D:.8f}")
    print(f"theta_deg: {x_guess[7]*R2D:.8f}")
    print(f"psi_deg:   {x_guess[8]*R2D:.8f}")
    print(f"lat_deg:   {x_guess[9]*R2D:.8f}")
    print(f"long_deg:  {x_guess[10]*R2D:.8f}")
    print(f"alt_m:     {x_guess[11]:.8f}")
    print(f"m_fuel_kg: {x_guess[12]:.8f}")
    print(f"dela_deg:  {x_guess[13]*R2D:.8f}")
    print(f"dele_deg:  {x_guess[14]*R2D:.8f}")
    print(f"delr_deg:  {x_guess[15]*R2D:.8f}")
    print(f"delt_pct:  {x_guess[16]:.8f}")
    
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Values in x were outside bounds during a minimize step")

    bounds = [(-np.inf, np.inf)] * (NUM_STATE - 1)
    bounds[7]  = (-math.pi/3, math.pi/3)
    
    bounds[9]  = (lat_target_rad, lat_target_rad)      # Lock Latitude
    bounds[10] = (long_target_rad, long_target_rad)    # Lock Longitude
    bounds[11] = (h_target_m, h_target_m)              # Lock Altitude
    bounds[12] = (m_fuel_target_kg, m_fuel_target_kg)  # Lock Mass
    
    # Needs to be gotten from vehicle
    bounds[13] = (-15*D2R, 15*D2R)
    bounds[14] = (-35*D2R, 15*D2R)
    bounds[15] = (-7.5*D2R, 7.5*D2R)
    bounds[16] = (0, 100)
    
    cmod["trim_flag"] = True

    print("\nSolving for trim state...")
    result = minimize(
        fun = cost_function,
        x0 = x_guess,
        args = (vehicle, cmod),
        method = 'SLSQP',
        bounds = bounds,
        constraints = define_trim_constraints(),
        options={'disp': True, 'ftol': 1e-9, 'maxiter': 500}
    )
    
    cmod["trim_flag"] = False
    
    # Process Results
    x_trim = result.x
    phi_rad = x_trim[6]
    theta_rad = x_trim[7]
    psi_rad = x_trim[8]
    lat_rad = x_trim[9]
    long_rad = x_trim[10]
    h_m = x_trim[11]
    
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
    
    dx = np.empty((NUM_STATE,), dtype=float)
    auxillary_data = np.empty((NUM_AUX,), dtype=float)
    
    # Construct verified ECEF array for final solve output
    x_e_m, y_e_m, z_e_m = eom.earth_model.geodetic_to_ecef(lat_rad, long_rad, h_m)

    x_trim_full = np.array([
        x_trim[0], x_trim[1], x_trim[2], x_trim[3], x_trim[4], x_trim[5],
        q0, q1, q2, q3,
        x_e_m, y_e_m, z_e_m,
        x_trim[12], x_trim[13], x_trim[14], x_trim[15], x_trim[16]
    ], dtype=float)

    x_trim_ref = np.array([
        x_trim[0], x_trim[1], x_trim[2], x_trim[3], x_trim[4], x_trim[5],
        phi_rad, theta_rad, psi_rad,
        lat_rad, long_rad, h_m,
        x_trim[12],
        x_trim[13],
        x_trim[14],
        x_trim[15],
        x_trim[16]
    ], dtype=float)
    
    # Wind Engine Interface
    W_N_mps, W_E_mps, W_D_mps = wmod.get_velocity(h_m)
    
    W_n_mps = np.array([W_N_mps, W_E_mps, W_D_mps])
    W_b_mps = C_n2b @ W_n_mps

    # Air-Relative Translational States
    u_air_b_mps = x[0] - W_b_mps[0]
    v_air_b_mps = x[1] - W_b_mps[1]
    w_air_b_mps = x[2] - W_b_mps[2]
    
    rho_kgpm3 = fastInterp1(amod["alt_m"], amod["rho_kgpm3"], h_m)
    true_airspeed_mps = math.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
    
    alpha_rad = math.atan2(w_air_b_mps, u_air_b_mps)
    beta_rad = math.asin(v_air_b_mps / true_airspeed_mps) if true_airspeed_mps > 0 else 0.0
    
    # Extract the dynamic SAS commands reacting to the trimmed body rates
    vehicle.set_gnc_inputs(cmod, amod, lat_rad, long_rad, h_m, alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, x[3], x[4], x[5], true_airspeed_mps, rho_kgpm3, x_trim_ref)
    dela_dyn_rad = vehicle.roll_control(0, x_trim[3], x_trim[5], cmod, 0)
    dele_dyn_rad = vehicle.pitch_control(0, x_trim[4], cmod, 0)
    delr_dyn_rad = vehicle.yaw_control(0, x_trim[5], cmod, 0)
    delt_dyn_pct = vehicle.throttle_control(0, cmod, 0)
    
    # Subtract the SAS contribution to isolate the true baseline pilot trim
    x_trim_ref[13] -= dela_dyn_rad
    x_trim_ref[14] -= dele_dyn_rad
    x_trim_ref[15] -= delr_dyn_rad
    x_trim_ref[16] -= delt_dyn_pct
    
    dx, auxillary_data = eom.solve_eom(0, x_trim_full, dx, auxillary_data, x_trim_ref, vehicle, cmod)

    if result.success:
        
        # Calculate proper Euler Rates using kinematic equations
        p_nb_rps, q_nb_rps, r_nb_rps = auxillary_data[4], auxillary_data[5], auxillary_data[6]
        
        phi_rad_dot   = p_nb_rps + q_nb_rps * math.sin(phi_rad) * math.tan(theta_rad) + r_nb_rps * math.cos(phi_rad) * math.tan(theta_rad)
        theta_rad_dot = q_nb_rps * math.cos(phi_rad) - r_nb_rps * math.sin(phi_rad)
        psi_rad_dot   = (q_nb_rps * math.sin(phi_rad) + r_nb_rps * math.cos(phi_rad)) / math.cos(theta_rad)
        
        print(f"Trim Successful! Cost function value: {result.fun:.3e}")
        print("-" * 25)
        print(f"VT_mps:    {math.sqrt(x_trim[0]**2 + x_trim[1]**2 + x_trim[2]**2):.8f}")
        print(f"alpha_deg: {math.atan2(x_trim[2], x_trim[0])*R2D:.8f}")
        print(f"beta_deg:  {math.asin(x_trim[1]/math.sqrt(x_trim[0]**2 + x_trim[1]**2 + x_trim[2]**2))*R2D:.8f}")
        print(f"p_dps:     {x_trim[3]*R2D:.8f}")
        print(f"q_dps:     {x_trim[4]*R2D:.8f}")
        print(f"r_dps:     {x_trim[5]*R2D:.8f}")
        print(f"phi_deg:   {phi_rad*R2D:.8f}")
        print(f"theta_deg: {theta_rad*R2D:.8f}")
        print(f"psi_deg:   {psi_rad*R2D:.8f}")
        print(f"lat_deg:   {x_trim[9]*R2D:.8f}")
        print(f"long_deg:  {x_trim[10]*R2D:.8f}")
        print(f"alt_m:     {x_trim[11]:.8f}")
        print(f"m_fuel_kg: {x_trim[12]:.8f}")
        print(f"dela_deg:  {x_trim[13]*R2D:.8f}")
        print(f"dele_deg:  {x_trim[14]*R2D:.8f}")
        print(f"delr_deg:  {x_trim[15]*R2D:.8f}")
        print(f"delt_pct:  {x_trim[16]:.8f}")
        print(" ")
        print(f"u_b_mps-dot:      {dx[0]: .8f}")
        print(f"v_b_mps-dot:      {dx[1]: .8f}")
        print(f"w_b_mps-dot:      {dx[2]: .8f}")
        print(f"p_b_dps-dot:      {dx[3]*R2D: .8f}")
        print(f"q_b_dps-dot:      {dx[4]*R2D: .8f}")
        print(f"r_b_dps-dot:      {dx[5]*R2D: .8f}")
        print(f"phi_deg-dot:      {phi_rad_dot*R2D: .8f}")
        print(f"theta_deg-dot:    {theta_rad_dot*R2D: .8f}")
        print(f"psi_deg-dot:      {psi_rad_dot*R2D: .8f}")
        print(f"x_e_m-dot:        {dx[10]: .8f}")
        print(f"y_e_m-dot:        {dx[11]: .8f}")
        print(f"z_e_m-dot:        {dx[12]: .8f}")
        print(f"m_fuel_kg-dot:    {dx[13]: .8f}")
        print(f"dela_ach_deg-dot: {dx[14]*R2D: .8f}")
        print(f"dele_ach_deg-dot: {dx[15]*R2D: .8f}")
        print(f"delr_ach_deg-dot: {dx[16]*R2D: .8f}")
        print(f"delt_ach_pct-dot: {dx[17]: .8f}")
        print(" ")
        print(f"dela_trim_deg: {x_trim_full[13]: .8f}")
        print(f"dele_trim_deg: {x_trim_full[14]: .8f}")
        print(f"delr_trim_deg: {x_trim_full[15]: .8f}")
        print(f"delt_trim_deg: {x_trim_full[16]: .8f}")
        
        return x_trim_full, x_trim_ref, result.message
    else:
        print(f"\n!!! TRIM FAILED TO CONVERGE !!!")
        print(f"Message: {result.message}")
        return None, None, result.message