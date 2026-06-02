import warnings
import numpy as np
import math
from scipy.optimize import minimize

from src.utils.constants import D2R, R2D, A_WGS84_M, G0_MPS2
from src.utils.interpolators import fastInterp1

def trim_solver(eom, vehicle, amod, cmod, tmod, x):
    """
    Finds the trimmed flight state by minimizing angular accelerations subject to kinematic constraints.
    """

    # 2. Define Internal Optimizer Functions
    def cost_function(x_trim, vehicle, amod, cmod):
        # Unpack optimizer state
        u, v, w = x_trim[0:3]
        p, q, r = x_trim[3:6]
        phi_rad, theta_rad, psi_rad = x_trim[6:9]
        rest_of_state = x_trim[9:].copy()
        if is_flat_earth:
            rest_of_state[2] = -x_trim[11] # Map positive altitude to NED Z-axis (p3)

        # Convert Euler to Quaternion (XYZ/321 sequence)
        cphi, sphi = np.cos(phi_rad/2), np.sin(phi_rad/2)
        cthe, sthe = np.cos(theta_rad/2), np.sin(theta_rad/2)
        cpsi, spsi = np.cos(psi_rad/2), np.sin(psi_rad/2)

        q0 = cphi * cthe * cpsi + sphi * sthe * spsi
        q1 = sphi * cthe * cpsi - cphi * sthe * spsi
        q2 = cphi * sthe * cpsi + sphi * cthe * spsi
        q3 = cphi * cthe * spsi - sphi * sthe * cpsi

        # Construct 17-element vector for EOM
        x_full = np.concatenate(([u, v, w, p, q, r], [q0, q1, q2, q3], rest_of_state))
        
        dx = np.empty((17,), dtype=float)
        auxillary_data = np.empty((7,), dtype=float)
        
        # Call the EOM
        dx, auxillary_data = eom.solve_eom(0, x_full, dx, auxillary_data, None, vehicle, amod, cmod)
        
        if tmod["trim_mode"] == 'steady_glide':
            cost = dx[0]**2 + dx[1]**2 + dx[2]**2 + dx[3]**2 + dx[4]**2 + dx[5]**2
        elif tmod["trim_mode"] == 'moment_equilibrium':
            cost = dx[3]**2 + dx[4]**2 + dx[5]**2
        elif tmod["trim_mode"] == 'descending_turn':
            # p_nb_rps, q_nb_rps, r_nb_rps = auxillary_data[4], auxillary_data[5], auxillary_data[6]
            # psidot_current = (q_nb_rps * math.sin(phi_rad) + r_nb_rps * math.cos(phi_rad)) / math.cos(theta_rad)
            psidot_current = (q * math.sin(phi_rad) + r * math.cos(phi_rad)) / math.cos(theta_rad)
            cost = 0*dx[0]**2 + dx[1]**2 + 0*dx[2]**2 + dx[3]**2 + dx[4]**2 + dx[5]**2 + 1e1*(psidot_current-psidot_target_rps)**2
        else:
            cost = dx[3]**2 + dx[4]**2 + dx[5]**2 # Fallback
            
        return cost

    def define_trim_constraints():
        def velocity_constraint(x_trim):
            V_T_current_mps = math.sqrt(x_trim[0]**2 + x_trim[1]**2 + x_trim[2]**2)
            return V_T_current_mps - V_T_target_mps
        
        def alpha_constraint(x_trim):
            alpha_current_rad = math.atan(x_trim[2]/x_trim[0])
            return alpha_current_rad - 0
        
        def beta_constraint(x_trim):
            V_T_current_mps = math.sqrt(x_trim[0]**2 + x_trim[1]**2 + x_trim[2]**2)
            beta_current_rad = math.asin(x_trim[1]/V_T_current_mps)
            return beta_current_rad - beta_target_rad
        
        def roll_rate_constraint(x_trim): return x_trim[3] - p_target_rps
        def pitch_rate_constraint(x_trim): return x_trim[4] - q_target_rps
        def yaw_rate_constraint(x_trim): return x_trim[5] - r_target_rps
        def roll_constraint(x_trim): return x_trim[6] - phi_target_rad
        def pitch_constraint(x_trim): return x_trim[7] - theta_target_rad
        def heading_constraint(x_trim): return x_trim[8] - psi_target_rad
        def altitude_constraint(x_trim): return x_trim[11] - h_target_m
        def latitude_constraint(x_trim): return x_trim[9] - lat_target_rad
        def longitude_constraint(x_trim): return x_trim[10] - long_target_rad
        def position_constraint(x_trim): return x_trim[9] - lat_target_rad + x_trim[10] - long_target_rad + x_trim[11] - h_target_m
        def mass_constraint(x_trim): return x_trim[15] - m_fuel_target_kg
        
        def flight_path_angle_constraint(x_trim):
            gamma_current_rad = x_trim[7] - math.atan2(x_trim[2], x_trim[0])
            return gamma_current_rad - gamma_target_rad
        
        def theta_rate_of_climb_constraint(x_trim):
            V_T_current_mps = math.sqrt(x_trim[0]**2 + x_trim[1]**2 + x_trim[2]**2)
            alpha_current_rad = math.atan(x_trim[2]/x_trim[0])
            beta_current_rad = math.asin(x_trim[1]/V_T_current_mps)
            gamma_current_rad = x_trim[7] - alpha_current_rad
            
            h_m_current = x_trim[11]
            
            if is_flat_earth:
                g_local = fastInterp1(amod["alt_m"], amod['g_mps2'], h_m_current)
            else:
                s_lat = math.sin(x_trim[9])
                g_0 = G0_MPS2 * (1.0 + 0.00193185265241 * s_lat**2) / math.sqrt(1.0 - 0.00669437999014 * s_lat**2)
                g_local = g_0 * (A_WGS84_M / (A_WGS84_M + h_m_current))**2
            
            G = (psidot_target_rps * V_T_target_mps) / g_local
            a = 1 - G*math.tan(alpha_current_rad)*math.sin(beta_current_rad)
            b = math.sin(gamma_current_rad)/math.cos(beta_current_rad)
            c = 1 + G**2*math.cos(beta_current_rad)**2
            
            # Isolate the term inside the first square root
            sqrt_term_phi = c*(1 - b**2) + G**2 * math.sin(beta_current_rad)**2
            
            # Penalize mathematically impossible states probed by the trim solver
            if sqrt_term_phi < 0:
                return 1e10
                
            tan_phi_target_rad = G*(math.cos(beta_current_rad)/math.cos(alpha_current_rad)) * \
                ((a - b**2) + b*math.tan(alpha_current_rad)*math.sqrt(sqrt_term_phi)) / \
                (a**2 - b**2*(1 + c*math.tan(alpha_current_rad)**2))
            phi_target_rad = math.atan(tan_phi_target_rad)
            
            a = math.cos(alpha_current_rad)*math.cos(beta_current_rad)
            b = math.sin(phi_target_rad)*math.sin(beta_current_rad)+\
                math.cos(phi_target_rad)*math.sin(alpha_current_rad)*\
                math.cos(beta_current_rad)
            sqrt_term_inside = a**2-math.sin(gamma_current_rad)**2+b**2
            
            if sqrt_term_inside < 0:
                return 1e10
                
            numerator = a*b+math.sin(gamma_current_rad)*math.sqrt(sqrt_term_inside)
            denominator = a**2-math.sin(gamma_current_rad)**2
            theta_target_rad = math.atan(numerator/denominator)
            return x_trim[7] - theta_target_rad
        
        def phi_turn_coord_constraint(x_trim):
            V_T_current_mps = math.sqrt(x_trim[0]**2 + x_trim[1]**2 + x_trim[2]**2)
            alpha_current_rad = math.atan(x_trim[2]/x_trim[0])
            beta_current_rad = math.asin(x_trim[1]/V_T_current_mps)
            gamma_current_rad = x_trim[7] - alpha_current_rad
            
            h_m_current = x_trim[11] 
            
            if is_flat_earth:
                g_local = fastInterp1(amod["alt_m"], amod['g_mps2'], h_m_current)
            else:
                s_lat = math.sin(x_trim[9])
                g_0 = G0_MPS2 * (1.0 + 0.00193185265241 * s_lat**2) / math.sqrt(1.0 - 0.00669437999014 * s_lat**2)
                g_local = g_0 * (A_WGS84_M / (A_WGS84_M + h_m_current))**2
            
            G = (psidot_target_rps * V_T_target_mps) / g_local
            a = 1 - G*math.tan(alpha_current_rad)*math.sin(beta_current_rad)
            b = math.sin(gamma_current_rad)/math.cos(beta_current_rad)
            c = 1 + G**2*math.cos(beta_current_rad)**2
            
            # Isolate the term inside the first square root
            sqrt_term_phi = c*(1 - b**2) + G**2 * math.sin(beta_current_rad)**2
            
            # Penalize mathematically impossible states probed by the trim solver
            if sqrt_term_phi < 0:
                return 1e10
                
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
    
    # 3. Setup and Execution
    print("--- Unpowered Trim Solver ---")
    
    is_flat_earth = type(eom).__name__ == "eom_flat_earth"
    
    cmod['trim_flag'] = cmod.get('trim_flag', False) # Defaults to off if missing
    cmod['linearization_flag'] = cmod.get('linearization_flag', False)
    
    # 1. Extract initial guesses from the passed configuration vectors
    V_T_curr_mps = math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        
    if x[0] == 0:
        w_over_u = 0
    else:
        w_over_u = x[2]/x[0]
    alpha_curr_rad = math.atan(w_over_u)
    
    if V_T_curr_mps == 0:
        v_over_VT = 0
    else:
        v_over_VT = x[1]/V_T_curr_mps
    beta_curr_rad = math.asin(v_over_VT)
    
    h_initial_m = -x[12] if is_flat_earth else x[12]
    h_target_m = tmod.get('h_m', h_initial_m)
    
    Cs_mps = fastInterp1(amod["alt_m"], amod["c_mps"], h_initial_m)
    Mach_curr = V_T_curr_mps/Cs_mps
    
    c_snd = fastInterp1(amod['alt_m'], amod['c_mps'], h_target_m)
    V_T_target_mps = tmod.get('Mach', Mach_curr) * c_snd
    
    alpha_target_rad = tmod['alpha_deg'] * D2R if tmod.get('alpha_deg') is not None else alpha_curr_rad
    beta_target_rad = tmod['beta_deg'] * D2R if tmod.get('beta_deg') is not None else beta_curr_rad
    
    q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
    phi_curr_rad   = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta_curr_rad = math.asin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
    psi_curr_rad   = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

    s_alpha = math.sin(alpha_target_rad)
    c_alpha = math.cos(alpha_target_rad)
    s_beta = math.sin(beta_target_rad)
    c_beta = math.cos(beta_target_rad)
    
    u_target_b_mps   = c_alpha * c_beta * V_T_target_mps
    v_target_b_mps   = s_beta * V_T_target_mps
    w_target_b_mps   = s_alpha * c_beta * V_T_target_mps
    p_target_rps     = tmod['p_rps'] if tmod.get('p_rps') is not None else x[3]
    q_target_rps     = tmod['q_rps'] if tmod.get('q_rps') is not None else x[4]
    r_target_rps     = tmod['r_rps'] if tmod.get('r_rps') is not None else x[5]
    phi_target_rad   = tmod['phi_deg'] * D2R if tmod.get('phi_deg') is not None else phi_curr_rad
    theta_target_rad = tmod['theta_deg'] * D2R if tmod.get('theta_deg') is not None else theta_curr_rad
    psi_target_rad   = tmod['psi_deg'] * D2R if tmod.get('psi_deg') is not None else psi_curr_rad
    lat_target_rad   = tmod['lat_deg'] * D2R if tmod.get('lat_deg') is not None else x[10]
    long_target_rad  = tmod['long_deg'] * D2R if tmod.get('long_deg') is not None else x[11]
    m_fuel_target_kg = tmod.get('m_fuel_kg', x[16])
    
    psidot_target_rps = tmod.get('psidot_dps') * D2R
    gamma_target_rad = theta_target_rad - alpha_target_rad

    # Use current state vector overrided by provided guess parameters
    x_guess = np.zeros(16)
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
    x_guess[12] = tmod.get('dela_ach_deg', x[13])
    x_guess[13] = tmod.get('dele_ach_deg', x[14])
    x_guess[14] = tmod.get('delr_ach_deg', x[15])
    x_guess[15] = m_fuel_target_kg
    
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Values in x were outside bounds during a minimize step")

    bounds = [(-np.inf, np.inf)] * 16
    bounds[7]  = (-math.pi/3, math.pi/3)
    bounds[12] = (-15, 15)
    bounds[13] = (-35, 15)
    bounds[14] = (-7.5, 7.5)
    
    bounds[9]  = (lat_target_rad, lat_target_rad)      # Lock Latitude
    bounds[10] = (long_target_rad, long_target_rad)    # Lock Longitude
    bounds[11] = (h_target_m, h_target_m)              # Lock Altitude
    bounds[15] = (m_fuel_target_kg, m_fuel_target_kg)  # Lock Mass
    
    cmod["trim_flag"] = True

    print("Solving for trim state...")
    result = minimize(
        fun = cost_function,
        x0 = x_guess,
        args = (vehicle, amod, cmod),
        method = 'SLSQP',
        bounds = bounds,
        constraints = define_trim_constraints(),
        options={'disp': True, 'ftol': 1e-8, 'maxiter': 300}
    )
    
    cmod["trim_flag"] = False
    
    # 4. Process Results
    x_trim = result.x
    phi_rad = x_trim[6]
    theta_rad = x_trim[7]
    psi_rad = x_trim[8]
    
    cphi, sphi = np.cos(x_trim[6]/2), np.sin(x_trim[6]/2)
    cthe, sthe = np.cos(x_trim[7]/2), np.sin(x_trim[7]/2)
    cpsi, spsi = np.cos(x_trim[8]/2), np.sin(x_trim[8]/2)

    q0 = cphi * cthe * cpsi + sphi * sthe * spsi
    q1 = sphi * cthe * cpsi - cphi * sthe * spsi
    q2 = cphi * sthe * cpsi + sphi * cthe * spsi
    q3 = cphi * cthe * spsi - sphi * sthe * cpsi
    
    dx = np.empty((17,), dtype=float)
    auxillary_data = np.empty((7,), dtype=float)
    
    rest_of_state_full = x_trim[9:].copy()
    if is_flat_earth:
        rest_of_state_full[2] = -x_trim[11]

    x_trim_full = np.concatenate((x_trim[0:6], [q0, q1, q2, q3], rest_of_state_full))
    dx, auxillary_data = eom.solve_eom(0, x_trim_full, dx, auxillary_data, None, vehicle, amod, cmod)

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
        print(f"p1_n_m:    {x_trim[9]:.8f}"     if is_flat_earth else f"lat_deg:   {x_trim[9]*R2D:.8f}")
        print(f"p2_n_m:    {x_trim[10]:.8f}"    if is_flat_earth else f"long_deg:  {x_trim[10]*R2D:.8f}")
        print(f"p3_n_m:    {-x_trim[11]:.8f}"   if is_flat_earth else f"alt_m:     {x_trim[11]:.8f}")
        print(f"dela_deg:  {x_trim[12]:.8f}")
        print(f"dele_deg:  {x_trim[13]:.8f}")
        print(f"delr_deg:  {x_trim[14]:.8f}")
        print(f"m_fuel_kg: {x_trim[15]:.8f}")
        print(" ")
        print(f"u_b_mps-dot:      {dx[0]: .8f}")
        print(f"v_b_mps-dot:      {dx[1]: .8f}")
        print(f"w_b_mps-dot:      {dx[2]: .8f}")
        print(f"p_b_rps-dot:      {dx[3]: .8f}")
        print(f"q_b_rps-dot:      {dx[4]: .8f}")
        print(f"r_b_rps-dot:      {dx[5]: .8f}")
        print(f"phi_rad-dot:      {phi_rad_dot: .8f}")
        print(f"theta_rad-dot:    {theta_rad_dot: .8f}")
        print(f"psi_rad-dot:      {psi_rad_dot: .8f}")
        print(f"p1_n_m-dot:       {dx[10]: .8f}"       if is_flat_earth else f"lat_deg-dot:      {dx[10]*R2D: .8f}")
        print(f"p2_n_m-dot:       {dx[11]: .8f}"       if is_flat_earth else f"long_deg-dot:     {dx[11]*R2D: .8f}")
        print(f"p3_n_m-dot:       {dx[12]: .8f}"       if is_flat_earth else f"alt_m-dot:        {dx[12]: .8f}")
        print(f"dela_ach_deg-dot: {dx[13]: .8f}")
        print(f"dele_ach_deg-dot: {dx[14]: .8f}")
        print(f"delr_ach_deg-dot: {dx[15]: .8f}")
        print(f"m_fuel_kg-dot:    {dx[16]: .8f}")
        
        # Package the trim controls for external use (like the B Matrix numerical derivation)
        u_trim = np.array([
            x_trim_full[13], # dela_cmd
            x_trim_full[14], # dele_cmd
            x_trim_full[15], # delr_cmd
            cmod['throttle_percent'] # throttle pct
        ])
        
        return x_trim_full, u_trim, result.message
    else:
        print(f"\n!!! TRIM FAILED TO CONVERGE !!!")
        print(f"Message: {result.message}")
        return None, None, result.message