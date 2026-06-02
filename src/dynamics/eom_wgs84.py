import math
import numpy as np

from src.dynamics.eom_base import eom
from src.utils.interpolators import fastInterp1
from src.control.open_loop_control import open_loop_speed_brake, open_loop_throttle
from src.utils.constants import A_WGS84_M, E_WGS84, OMEGA_E_RPS, G0_MPS2
from src.utils.kinematics import quat_body_to_nav, wind_to_body_dcm

class eom_wgs84(eom):
    def solve_eom(self, t, x, dx, auxillary_data, u_trim, vehicle, amod, cmod):

        # State Extraction with Descriptive Naming
        u_b_mps, v_b_mps, w_b_mps = x[0], x[1], x[2]
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
        lat_rad, long_rad, h_m = x[10], x[11], x[12]
        dela_ach_deg, dele_ach_deg, delr_ach_deg = x[13], x[14], x[15]
        m_fuel_kg = x[16]

        # Vehicle Mass State Interface
        m_total_kg = vehicle.m_dry_kg + m_fuel_kg
        Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxz_b_kgm2 = vehicle.get_mass_properties(m_total_kg)
        
        speedbrake = cmod.get("speedbrake", False)

        # Control Routing
        delsb_deg = open_loop_speed_brake()
        throttle_perc = open_loop_throttle() if cmod.get("linearization_flag") != 'on' else cmod['throttle_percent']
        
        # Engine Interface
        m_fuel_dot_kgps = vehicle.get_engine_burn_rate(throttle_perc)
        
        # Trim & Linearization Overrides
        dela_ach_deg_old, dele_ach_deg_old, delr_ach_deg_old = dela_ach_deg, dele_ach_deg, delr_ach_deg
        if cmod.get("trim_flag"):
            # Surfaces fixed
            dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = dela_ach_deg, dele_ach_deg, delr_ach_deg
        elif cmod.get("linearization_flag"):
            # Commanded values
            dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg']
        else:
            # Require the vehicle or SAS object to return control deflections
            dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = vehicle.get_sas_commands(t, x, cmod, u_trim)
        
        if cmod.get("type") == "time_history":
            dela_ach_deg, dele_ach_deg, delr_ach_deg = dela_cmd_deg, dele_cmd_deg, delr_cmd_deg
            x[13], x[14], x[15] = dela_ach_deg, dele_ach_deg, delr_ach_deg

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
                                                                                                                    delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake)

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
        q_norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
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
        dx[13] = vehicle.aileron_kinematics(dela_cmd_deg, dela_ach_deg_old)
        dx[14] = vehicle.elevator_kinematics(dele_cmd_deg, dele_ach_deg_old)
        dx[15] = vehicle.rudder_kinematics(delr_cmd_deg, delr_ach_deg_old)
        dx[16] = -m_fuel_dot_kgps

        # Aux Data
        auxillary_data[0:4] = [dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, throttle_perc]
        auxillary_data[4:7] = [p_nb_rps, q_nb_rps, r_nb_rps]

        return dx, auxillary_data
    
    def post_process(self, x, t_s, amod, auxillary_data, **kwargs):
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
        
        sel_auxillary_data_dela_cmd_deg = 0
        sel_auxillary_data_dele_cmd_deg = 1
        sel_auxillary_data_delr_cmd_deg = 2
        sel_auxillary_data_delt_percent = 3
        
        nt_s = t_s.size
        
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
        
        # Air Data Vectorization
        u_b = x[sel_state_u_b_mps, :]
        v_b = x[sel_state_v_b_mps, :]
        w_b = x[sel_state_w_b_mps, :]
        
        # True Airspeed
        True_Airspeed_mps = np.sqrt(u_b**2 + v_b**2 + w_b**2)[:, np.newaxis]
        
        # Angle of Attack (handle divide by zero)
        Alpha_rad = np.zeros_like(u_b)
        safe_u = u_b != 0
        Alpha_rad[safe_u] = np.arctan(w_b[safe_u] / u_b[safe_u])
        Alpha_rad = Alpha_rad[:, np.newaxis]
        
        # Angle of Sideslip (handle divide by zero)
        Beta_rad = np.zeros_like(True_Airspeed_mps).flatten()
        safe_vt = True_Airspeed_mps.flatten() != 0
        Beta_rad[safe_vt] = np.arcsin(v_b[safe_vt] / True_Airspeed_mps.flatten()[safe_vt])
        Beta_rad = Beta_rad[:, np.newaxis]
        
        # Atmosphere and Mach
        Cs_mps = np.array([fastInterp1(amod["alt_m"], amod["c_mps"], alt) for alt in altitude_m_arr])[:, np.newaxis]
        Rho_kgpm3 = np.array([fastInterp1(amod["alt_m"], amod["rho_kgpm3"], alt) for alt in altitude_m_arr])[:, np.newaxis]
        
        Mach = np.zeros_like(Cs_mps)
        safe_c = Cs_mps != 0
        Mach[safe_c] = True_Airspeed_mps[safe_c] / Cs_mps[safe_c]
        
        # Get auxillary data 
        dela_cmd_deg   = auxillary_data[sel_auxillary_data_dela_cmd_deg,:]
        dela_cmd_deg   = dela_cmd_deg[:, np.newaxis]
        dele_cmd_deg   = auxillary_data[sel_auxillary_data_dele_cmd_deg,:]
        dele_cmd_deg   = dele_cmd_deg[:, np.newaxis]
        delr_cmd_deg   = auxillary_data[sel_auxillary_data_delr_cmd_deg,:]
        delr_cmd_deg   = delr_cmd_deg[:, np.newaxis]
        delt_percent   = auxillary_data[sel_auxillary_data_delt_percent,:]
        delt_percent   = delt_percent[:, np.newaxis]

        # Combine state date with time and other post processed data
        t_s = t_s[:, np.newaxis]
        sim_data = np.concatenate( (t_s, x.T, Cs_mps, Rho_kgpm3, Mach, Alpha_rad, \
                                    Beta_rad, True_Airspeed_mps, phi_rad, theta_rad, psi_rad, \
                                    u_n_mps, v_n_mps, w_n_mps, dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, \
                                    delt_percent), axis=1 )
        
        return sim_data