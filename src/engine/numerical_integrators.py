import numpy as np
from scipy.integrate import solve_ivp

from src.utils.constants import A_WGS84_M, E_WGS84

def forward_euler(f, t_s, x, h_s, vmod, amod, cmod, Auxillary_Data_Accumulated):
	"""
	Performs forward Euler integration to approximate the solution of a differential equation.

	Input Args:
		f:   A function representing the right-hand side of the differential equation (dx/dt = f(t, x)).
		t_s: A vector of points in time at which numerical solutions will be approximated
		x:   The numerically approximated solution data to the DE, f
		h_s: The step size in seconds

	Returns:
		t_s: A vector of points in time at which numerical solutions was approximated
		x:   The numerically approximated solution data to the DE, f
		
	"""

	for i in range(1, len(t_s)):
		dx, auxillary_data = f(t_s[i-1], x[:,i-1], vmod, amod, cmod)
		x[:,i] = x[:,i-1] + h_s * dx
		Auxillary_Data_Accumulated[:,i] = auxillary_data
		
		# REQUIRED: Normalize the quaternion
		q_norm = np.linalg.norm(x[6:10, i])
		x[6:10, i] = x[6:10, i] / q_norm

	return t_s, x, Auxillary_Data_Accumulated

def AB2(f, t_s, x, h_s, vmod, amod, cmod, Auxillary_Data_Accumulated):
	"""
	Performs the 2nd order Adams-Bashforth method to approximate the solution of a differential equation.

	Input Args:
		f:   A function representing the right-hand side of the differential equation (dx/dt = f(t, x)).
		t_s: A vector of points in time at which numerical solutions will be approximated
		x:   The numerically approximated solution data to the DE, f
		h_s: The step size in seconds

	Returns:
		t_s: A vector of points in time at which numerical solutions was approximated
		x:   The numerically approximated solution data to the DE, f
		
	"""

	for i in range(1, len(t_s)):
		fim1, auxillary_data = f(t_s[i-1], x[:,i-1], vmod, amod, cmod)
		if i == 0:
			x[:,i] = x[:,i-1] + h_s * fim1
		else:
			fim2, _ = f(t_s[i-2], x[:,i-2], vmod, amod, cmod)
			x[:,i] = x[:,i-1] + 1.5*h_s*fim1 - 0.5*h_s*fim2
		
		# REQUIRED: Normalize the quaternion
		q_norm = np.linalg.norm(x[6:10, i])
		x[6:10, i] = x[6:10, i] / q_norm
		
		Auxillary_Data_Accumulated[:,i] = auxillary_data

	return t_s, x, Auxillary_Data_Accumulated


def RK4(f, t, x, h_s, vmod, amod, cmod, u_trim, dx, auxillary_data):
    """
    Performs the 4th order Runge-Kutta method to approximate the solution of a differential equation.
    """
    fim1_k1, auxillary_data = f(t, x, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k1 = h_s*fim1_k1
    fim1_k2, _ = f(t + 0.5*h_s, x + 0.5*k1, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k2 = h_s*fim1_k2
    fim1_k3, _ = f(t + 0.5*h_s, x + 0.5*k2, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k3 = h_s*fim1_k3
    fim1_k4, _ = f(t + h_s, x + k3, dx, auxillary_data, u_trim, vmod, amod, cmod)
    k4 = h_s*fim1_k4 
    x_new = x + 1/6*(k1 + 2.0*k2 + 2.0*k3 + k4)

    return x_new, auxillary_data

def adaptive_integration(eom_func, t_span, t_eval, x0, vehicle, amod, cmod, u_trim, method='RK45', rtol=1e-6, atol=1e-6):
    """
    Adaptive integrator wrapper using scipy's solve_ivp. 
    Supports 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', and 'LSODA'.
    """
    print(f"[{method} Integration Engine Active]")
    
    # Preallocate arrays to prevent memory overhead inside the ODE evaluator
    dx_tmp = np.empty(len(x0), dtype=float)
    aux_tmp = np.empty(7, dtype=float)

    def eom_wrapper(t, x):
        # SciPy provides 't' as a float and 'x' as a 1D array
        eom_func(t, x, dx_tmp, aux_tmp, u_trim, vehicle, amod, cmod)
        return dx_tmp

    # Phase 1: State Integration
    res = solve_ivp(
        fun=eom_wrapper,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )

    if not res.success:
        print(f"\n!!! Integration Warning !!!\nSolver terminated early: {res.message}")

    x_out = res.y
    t_out = res.t
    
    # # --- Vectorized Kinematic Reconstruction ---
    # u_b, v_b, w_b = x_out[0, :], x_out[1, :], x_out[2, :]
    # p_b, q_b, r_b = x_out[3, :], x_out[4, :], x_out[5, :]
    # q0, q1, q2, q3 = x_out[6, :], x_out[7, :], x_out[8, :], x_out[9, :]
    # lat_rad = x_out[10, :]
    # h_m = x_out[12, :]

    # # WGS-84 Radii
    # sin_lat = np.sin(lat_rad)
    # den_wgs84 = np.sqrt(1.0 - (E_WGS84 * sin_lat)**2)
    # RN_m = A_WGS84_M / den_wgs84
    # RM_m = (A_WGS84_M * (1.0 - E_WGS84**2)) / (den_wgs84**3)

    # # C_b2n Components (Unrolled for matrix math speed)
    # C_b2n_00 = q0**2 + q1**2 - q2**2 - q3**2
    # C_b2n_01 = 2 * (q1*q2 - q0*q3)
    # C_b2n_02 = 2 * (q1*q3 + q0*q2)
    # C_b2n_10 = 2 * (q1*q2 + q0*q3)
    # C_b2n_11 = q0**2 - q1**2 + q2**2 - q3**2
    # C_b2n_12 = 2 * (q2*q3 - q0*q1)
    # C_b2n_20 = 2 * (q1*q3 - q0*q2)
    # C_b2n_21 = 2 * (q2*q3 + q0*q1)
    # C_b2n_22 = q0**2 - q1**2 - q2**2 + q3**2

    # # Nav Velocities
    # u_n = C_b2n_00*u_b + C_b2n_01*v_b + C_b2n_02*w_b
    # v_n = C_b2n_10*u_b + C_b2n_11*v_b + C_b2n_12*w_b

    # # Transport Rates in Nav Frame
    # omega_en_n_x = v_n / (RN_m + h_m)
    # omega_en_n_y = -u_n / (RM_m + h_m)
    # omega_en_n_z = -v_n * np.tan(lat_rad) / (RN_m + h_m)

    # # Transport Rates mapped to Body Frame (Using C_n2b, which is C_b2n.T)
    # omega_en_b_x = C_b2n_00*omega_en_n_x + C_b2n_10*omega_en_n_y + C_b2n_20*omega_en_n_z
    # omega_en_b_y = C_b2n_01*omega_en_n_x + C_b2n_11*omega_en_n_y + C_b2n_21*omega_en_n_z
    # omega_en_b_z = C_b2n_02*omega_en_n_x + C_b2n_12*omega_en_n_y + C_b2n_22*omega_en_n_z

    # # Nav-Relative Body Rates
    # p_nb = p_b - omega_en_b_x
    # q_nb = q_b - omega_en_b_y
    # r_nb = r_b - omega_en_b_z

    # --- Commanded Inputs Reconstruction ---
    nt = len(t_out)
    aux_data_accum = np.zeros((4, nt)) 
    
    # Pack tracked kinematics
    # aux_data_accum[4, :] = p_nb
    # aux_data_accum[5, :] = q_nb
    # aux_data_accum[6, :] = r_nb
    
    for i in range(nt):
        if cmod.get("trim_flag") == 'on':
            aux_data_accum[0:4, i] = [x_out[13, i], x_out[14, i], x_out[15, i], u_trim[3]]
        elif cmod.get("linearization_flag") == 'on':
            aux_data_accum[0:4, i] = [cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg'], cmod['throttle_percent']]
        else:
            # Fast SAS routing
            d_a, d_e, d_r = vehicle.get_sas_commands(t_out[i], x_out[:, i], cmod, u_trim)
            aux_data_accum[0:4, i] = [d_a, d_e, d_r, u_trim[3]]

    return t_out, x_out, aux_data_accum