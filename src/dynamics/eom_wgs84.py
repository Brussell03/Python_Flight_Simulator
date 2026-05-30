import math
import numpy as np
from src.utils.interpolators import fastInterp1
from src.control.open_loop_control import open_loop_speed_brake, open_loop_throttle
from src.utils.constants import A_WGS84_M, E_WGS84, OMEGA_E_RPS, G0_MPS2
from src.utils.kinematics import quat_body_to_nav, wind_to_body_dcm

def actuator_kinematics(cmd_deg, ach_deg, tau_s, pos_lims, rate_lim_dps):
    """
    Computes actuator state derivative enforcing rate and position saturation.
    """
    # 1. Compute unbounded linear rate
    rate_dps = (cmd_deg - ach_deg) / tau_s
    
    # 2. Enforce Rate Saturation (Hydraulic limit)
    rate_dps = np.clip(rate_dps, -rate_lim_dps, rate_lim_dps)
    
    # 3. Enforce Position Saturation (Mechanical hard stops)
    # If we are at or beyond the max limit and trying to push further, rate is zero
    if ach_deg >= pos_lims[1] and rate_dps > 0.0:
        rate_dps = 0.0
    # If we are at or below the min limit and trying to push further, rate is zero
    elif ach_deg <= pos_lims[0] and rate_dps < 0.0:
        rate_dps = 0.0
        
    return rate_dps

def eom_wgs84(t, x, dx, auxillary_data, u_trim, vehicle, amod, cmod):

    # State Extraction with Descriptive Naming
    u_b_mps, v_b_mps, w_b_mps = x[0], x[1], x[2]
    p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
    q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
    lat_rad, long_rad, h_m = x[10], x[11], x[12]
    dela_ach_deg, dele_ach_deg, delr_ach_deg = x[13], x[14], x[15]
    m_fuel_kg = x[16]
    
    # Quaternion Normalization Guard
    q_norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    if q_norm > 0:
        q0, q1, q2, q3 = q0/q_norm, q1/q_norm, q2/q_norm, q3/q_norm
    else:
        q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0 # Fallback for catastrophic failure

    # Vehicle Mass State Interface
    m_total_kg = vehicle.m_dry_kg + m_fuel_kg
    Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxz_b_kgm2 = vehicle.get_mass_properties(m_total_kg)

    # Control Routing
    delsb_deg = open_loop_speed_brake()
    throttle_perc = open_loop_throttle() if cmod.get("linearization_flag") != 'on' else cmod['throttle_percent']
    
    # Engine Interface
    m_fuel_dot_kgps = vehicle.get_engine_burn_rate(throttle_perc)
    
    # Trim & Linearization Overrides
    if cmod.get("trim_flag"):
        # Surfaces fixed
        dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = dela_ach_deg, dele_ach_deg, delr_ach_deg
    elif cmod.get("linearization_flag"):
        # Commanded values
        dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg']
    else:
        # Require the vehicle or SAS object to return control deflections
        dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = vehicle.get_sas_commands(t, x, cmod, u_trim)

    # Atmosphere & Air Data
    rho_kgpm3 = fastInterp1(amod["alt_m"], amod["rho_kgpm3"], h_m)
    c_snd_mps = fastInterp1(amod["alt_m"], amod["c_mps"], h_m)
    
    true_airspeed_mps = math.sqrt(u_b_mps**2 + v_b_mps**2 + w_b_mps**2)
    qbar_kgpms2 = 0.5 * rho_kgpm3 * true_airspeed_mps**2
    Mach = true_airspeed_mps / c_snd_mps if c_snd_mps > 0 else 0.0
    
    alpha_rad = math.atan2(w_b_mps, u_b_mps)
    beta_rad = math.asin(v_b_mps / true_airspeed_mps) if true_airspeed_mps > 0 else 0.0

    # Kinematics Interface
    C_w2b = wind_to_body_dcm(alpha_rad, beta_rad)
    C_b2n = quat_body_to_nav(q0, q1, q2, q3)
    C_n2b = C_b2n.T

    # Navigation & WGS-84 Radii
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    den_wgs84 = math.sqrt(1.0 - (E_WGS84 * sin_lat)**2)
    RN_m = A_WGS84_M / den_wgs84
    RM_m = (A_WGS84_M * (1.0 - E_WGS84**2)) / (den_wgs84**3)

    # Velocity Transformations
    v_body_mps = np.array([u_b_mps, v_b_mps, w_b_mps])
    v_ned_mps = C_b2n @ v_body_mps
    u_n_mps, v_n_mps, w_n_mps = v_ned_mps[0], v_ned_mps[1], v_ned_mps[2]

    # Earth & Transport Rates
    omega_ie_n_rps = np.array([OMEGA_E_RPS * cos_lat, 0.0, -OMEGA_E_RPS * sin_lat])
    omega_en_n_rps = np.array([v_n_mps / (RN_m + h_m), -u_n_mps / (RM_m + h_m), -v_n_mps * math.tan(lat_rad) / (RN_m + h_m)])

    omega_ie_b_rps = C_n2b @ omega_ie_n_rps
    omega_en_b_rps = C_n2b @ omega_en_n_rps

    p_nb_rps = p_b_rps - omega_en_b_rps[0]
    q_nb_rps = q_b_rps - omega_en_b_rps[1]
    r_nb_rps = r_b_rps - omega_en_b_rps[2]

    # Gravity
    gz_n_mps2 = G0_MPS2 * (1.0 + 0.00193185265241 * sin_lat**2) / math.sqrt(1.0 - 0.00669437999014 * sin_lat**2) * (A_WGS84_M / (A_WGS84_M + h_m))**2
    g_b_mps2 = C_n2b @ np.array([0.0, 0.0, gz_n_mps2])
    
    # Vehicle returns mapped body forces
    Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2, l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2 = vehicle.get_forces_and_moments(alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                                                                                                                  p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                                                                                                                  delr_ach_deg, delsb_deg, throttle_perc, C_w2b)

    # Velocity Equations (Coriolis)
    omega_cor_b_rps = np.array([p_b_rps, q_b_rps, r_b_rps]) + 2.0 * omega_ie_b_rps
    dx[0] = (Fx_b_kgmps2 / m_total_kg) + g_b_mps2[0] - (omega_cor_b_rps[1]*w_b_mps - omega_cor_b_rps[2]*v_b_mps)
    dx[1] = (Fy_b_kgmps2 / m_total_kg) + g_b_mps2[1] - (omega_cor_b_rps[2]*u_b_mps - omega_cor_b_rps[0]*w_b_mps)
    dx[2] = (Fz_b_kgmps2 / m_total_kg) + g_b_mps2[2] - (omega_cor_b_rps[0]*v_b_mps - omega_cor_b_rps[1]*u_b_mps)

    # Inertia Derivatives via Vehicle Method
    dm_kg = 1.0
    m_plus = np.clip(m_total_kg + dm_kg, vehicle.m_dry_kg, vehicle.m_wet_kg)
    m_minus = np.clip(m_total_kg - dm_kg, vehicle.m_dry_kg, vehicle.m_wet_kg)
    dm_diff = m_plus - m_minus

    if dm_diff > 0:
        J_plus = vehicle.get_mass_properties(m_plus)
        J_minus = vehicle.get_mass_properties(m_minus)
        dJ_dm = [(p - m) / dm_diff for p, m in zip(J_plus, J_minus)]
    else:
        dJ_dm = [0.0, 0.0, 0.0, 0.0]

    Jxx_dot, Jyy_dot, Jzz_dot, Jxz_dot = [dJ * -m_fuel_dot_kgps for dJ in dJ_dm]

    # Rotational Dynamics
    p_i_b_rps = p_b_rps + omega_ie_b_rps[0]
    q_i_b_rps = q_b_rps + omega_ie_b_rps[1]
    r_i_b_rps = r_b_rps + omega_ie_b_rps[2]

    hx_b_kgm2ps = Jxx_b_kgm2 * p_i_b_rps - Jxz_b_kgm2 * r_i_b_rps
    hy_b_kgm2ps = Jyy_b_kgm2 * q_i_b_rps
    hz_b_kgm2ps = -Jxz_b_kgm2 * p_i_b_rps + Jzz_b_kgm2 * r_i_b_rps

    Idot_l_b_kgm2ps2 = Jxx_dot * p_i_b_rps - Jxz_dot * r_i_b_rps
    Idot_m_b_kgm2ps2 = Jyy_dot * q_i_b_rps
    Idot_n_b_kgm2ps2 = -Jxz_dot * p_i_b_rps + Jzz_dot * r_i_b_rps

    gyro_l_b_kgm2ps2 = q_i_b_rps * hz_b_kgm2ps - r_i_b_rps * hy_b_kgm2ps
    gyro_m_b_kgm2ps2 = r_i_b_rps * hx_b_kgm2ps - p_i_b_rps * hz_b_kgm2ps
    gyro_n_b_kgm2ps2 = p_i_b_rps * hy_b_kgm2ps - q_i_b_rps * hx_b_kgm2ps

    l_tot_b_kgm2ps2 = l_b_kgm2ps2 - Idot_l_b_kgm2ps2 - gyro_l_b_kgm2ps2
    m_tot_b_kgm2ps2 = m_b_kgm2ps2 - Idot_m_b_kgm2ps2 - gyro_m_b_kgm2ps2
    n_tot_b_kgm2ps2 = n_b_kgm2ps2 - Idot_n_b_kgm2ps2 - gyro_n_b_kgm2ps2

    Gamma_inv = 1.0 / (Jxx_b_kgm2 * Jzz_b_kgm2 - Jxz_b_kgm2**2)
    pdot_b_rps2 = (Jzz_b_kgm2 * l_tot_b_kgm2ps2 + Jxz_b_kgm2 * n_tot_b_kgm2ps2) * Gamma_inv
    qdot_b_rps2 = m_tot_b_kgm2ps2 / Jyy_b_kgm2
    rdot_b_rps2 = (Jxz_b_kgm2 * l_tot_b_kgm2ps2 + Jxx_b_kgm2 * n_tot_b_kgm2ps2) * Gamma_inv

    # Kinematic & Transport Corrections
    omega_b_rps = np.array([p_b_rps, q_b_rps, r_b_rps])
    kin_cross_rps2 = np.cross(omega_b_rps, omega_ie_b_rps)
    omega_cross_n_rps2 = np.cross(omega_ie_n_rps, omega_en_n_rps)
    transport_cross_rps2 = C_n2b @ omega_cross_n_rps2

    dx[3] = pdot_b_rps2 + kin_cross_rps2[0]# + transport_cross_rps2[0]
    dx[4] = qdot_b_rps2 + kin_cross_rps2[1]# + transport_cross_rps2[1]
    dx[5] = rdot_b_rps2 + kin_cross_rps2[2]# + transport_cross_rps2[2]

    # Quaternions
    
    # Quaternions with Baumgarte Stabilization
    k_quat = 1.0 # Feedback gain to drive error to zero
    q_err = 1.0 - (q_norm**2)
    
    dx[6] = -0.5 * (p_nb_rps*q1 + q_nb_rps*q2 + r_nb_rps*q3) + k_quat * q_err * q0
    dx[7] =  0.5 * (p_nb_rps*q0 - q_nb_rps*q3 + r_nb_rps*q2) + k_quat * q_err * q1
    dx[8] =  0.5 * (q_nb_rps*q0 + p_nb_rps*q3 - r_nb_rps*q1) + k_quat * q_err * q2
    dx[9] =  0.5 * (r_nb_rps*q0 - p_nb_rps*q2 + q_nb_rps*q1) + k_quat * q_err * q3

    # Navigation
    dx[10] = u_n_mps / (RM_m + h_m)
    dx[11] = v_n_mps / ((RN_m + h_m) * cos_lat)
    dx[12] = -w_n_mps

    # Actuation & Fuel
    dx[13] = actuator_kinematics(dela_cmd_deg, dela_ach_deg, vehicle.tau_a_s, vehicle.lim_a_pos_deg, vehicle.lim_a_rate_dps)
    dx[14] = actuator_kinematics(dele_cmd_deg, dele_ach_deg, vehicle.tau_e_s, vehicle.lim_e_pos_deg, vehicle.lim_e_rate_dps)
    dx[15] = actuator_kinematics(delr_cmd_deg, delr_ach_deg, vehicle.tau_r_s, vehicle.lim_r_pos_deg, vehicle.lim_r_rate_dps)
    dx[16] = -m_fuel_dot_kgps

    # Aux Data
    auxillary_data[0:4] = [dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, throttle_perc]
    auxillary_data[4:7] = [p_nb_rps, q_nb_rps, r_nb_rps]

    return dx, auxillary_data