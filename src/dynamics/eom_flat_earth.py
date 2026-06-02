import math
import numpy as np

from src.dynamics.eom_base import eom
from src.utils.interpolators import fastInterp1
from src.control.open_loop_control import open_loop_speed_brake, open_loop_throttle
from src.utils.kinematics import quat_body_to_nav, wind_to_body_dcm

class eom_flat_earth(eom):
    def __init__(self, lat0_rad, long0_rad):
        self.lat0_rad = lat0_rad
        self.long0_rad = long0_rad
    
    def solve_eom(self, t, x, dx, auxillary_data, u_trim, vehicle, amod, cmod):

        # State Extraction with Descriptive Naming
        u_b_mps, v_b_mps, w_b_mps = x[0], x[1], x[2]
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
        p1_n_m, p2_n_m, p3_n_m = x[10], x[11], x[12]
        dela_ach_deg, dele_ach_deg, delr_ach_deg = x[13], x[14], x[15]
        m_fuel_kg = x[16]

        # Vehicle Mass State Interface
        m_total_kg = vehicle.m_dry_kg + m_fuel_kg
        Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxz_b_kgm2 = vehicle.get_mass_properties(m_total_kg)
        
        speedbrake = cmod.get("speedbrake", False)

        # Control Routing
        delsb_deg = open_loop_speed_brake()
        delt_percent = open_loop_throttle() if cmod.get("linearization_flag") != 'on' else cmod['throttle_percent']
        
        # Engine Interface
        m_fuel_dot_kgps = vehicle.get_engine_burn_rate(delt_percent)
        
        # # Trim & Linearization Overrides
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
        
        # Get current altitude
        h_m = -p3_n_m

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

        # Gravity
        gz_n_mps2 = fastInterp1(amod["alt_m"], amod['g_mps2'], h_m)
        g_b_mps2 = C_n2b @ np.array([0.0, 0.0, gz_n_mps2])
        
        # Vehicle returns mapped body forces
        Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2, l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2 = vehicle.get_forces_and_moments(alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                                                                                                                    p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                                                                                                                    delr_ach_deg, delsb_deg, delt_percent, C_w2b, speedbrake)

        # Velocity Equations
        dx[0] = (Fx_b_kgmps2 / m_total_kg) + g_b_mps2[0] - w_b_mps * q_b_rps + v_b_mps * r_b_rps
        dx[1] = (Fy_b_kgmps2 / m_total_kg) + g_b_mps2[1] - u_b_mps * r_b_rps + w_b_mps * p_b_rps
        dx[2] = (Fz_b_kgmps2 / m_total_kg) + g_b_mps2[2] - v_b_mps * p_b_rps + u_b_mps * q_b_rps

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

        # Effective moments including mass-varying torques (-J_dot * omega)
        l_total = l_b_kgm2ps2 - (Jxx_dot * p_b_rps - Jxz_dot * r_b_rps)
        m_total = m_b_kgm2ps2 - (Jyy_dot * q_b_rps)
        n_total = n_b_kgm2ps2 - (Jzz_dot * r_b_rps - Jxz_dot * p_b_rps)

        # Rotational Dynamics
        Gamma_inv = 1.0 / (Jxx_b_kgm2 * Jzz_b_kgm2 - Jxz_b_kgm2**2)

        dx[3] = (Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2  + Jzz_b_kgm2) * p_b_rps * q_b_rps - \
				(Jzz_b_kgm2 * (Jzz_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2**2) * q_b_rps * r_b_rps + \
				Jzz_b_kgm2 * l_total + Jxz_b_kgm2 * n_total) * Gamma_inv
        
        dx[4] = ((Jzz_b_kgm2 - Jxx_b_kgm2) * p_b_rps * r_b_rps - \
           		Jxz_b_kgm2 * (p_b_rps**2 - r_b_rps**2) + m_total) / Jyy_b_kgm2
        
        dx[5] = ((Jxx_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2) + Jxz_b_kgm2**2) * p_b_rps * q_b_rps - \
				Jxz_b_kgm2 * (Jxx_b_kgm2 - Jyy_b_kgm2 + Jzz_b_kgm2) * q_b_rps * r_b_rps + \
				Jxz_b_kgm2 * l_total + Jxx_b_kgm2 * n_total) * Gamma_inv
        
        # Quaternions with Baumgarte Stabilization
        k_quat = 1.0 # Feedback gain to drive error to zero
        q_norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q_err = 1.0 - (q_norm**2)
        
        dx[6] = -0.5 * (p_b_rps * q1 + q_b_rps * q2 + r_b_rps * q3) + k_quat * q_err * q0
        dx[7] =  0.5 * (p_b_rps * q0 - q_b_rps * q3 + r_b_rps * q2) + k_quat * q_err * q1
        dx[8] =  0.5 * (q_b_rps * q0 + p_b_rps * q3 - r_b_rps * q1) + k_quat * q_err * q2
        dx[9] =  0.5 * (r_b_rps * q0 - p_b_rps * q2 + q_b_rps * q1) + k_quat * q_err * q3

        # Position (Navigation) equations 
        dx[10:13] = C_b2n @ np.array([u_b_mps, v_b_mps, w_b_mps])

        # Actuation & Fuel
        dx[13] = vehicle.aileron_kinematics(dela_cmd_deg, dela_ach_deg)
        dx[14] = vehicle.elevator_kinematics(dele_cmd_deg, dele_ach_deg)
        dx[15] = vehicle.rudder_kinematics(delr_cmd_deg, delr_ach_deg)
        dx[16] = -m_fuel_dot_kgps

        # Aux Data
        auxillary_data[0:4] = [dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, delt_percent]
        auxillary_data[4:7] = [p_b_rps, q_b_rps, r_b_rps]

        return dx, auxillary_data
    
    def post_process(self, x, t_s, amod, auxillary_data):
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
        sel_state_p1_n_m    = 10
        sel_state_p2_n_m    = 11
        sel_state_p3_n_m    = 12
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
        
        altitude_m_arr = -x[sel_state_p3_n_m, :]
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
        
        # Convert north and east position to geodetic latitude and longitude
        p1_n_m = x[sel_state_p1_n_m, :]
        p2_n_m = x[sel_state_p2_n_m, :]
        
        # Equatorial radius of earth
        R = 6378137  
        
        # Earth flattening parameter
        f = 0.00335281
        
        # Compute change in latitude and longitude
        RN = R/(math.sqrt( 1 - (2*f - f**2)*math.sin(self.lat0_rad)**2 ) )
        RM = RN*( 1 - (2*f - f**2) )/( 1 - (2*f - f**2)*math.sin(self.lat0_rad)**2 ) 
        dlat_rad  = (p1_n_m - p1_n_m[0])/RM
        dlong_rad = (p2_n_m - p2_n_m[0])/(RN*math.cos(self.lat0_rad))
        
        # Approximate latitude and longitude
        lat_rad  = self.lat0_rad  +  dlat_rad
        long_rad = self.long0_rad + dlong_rad
        
        x[sel_state_p1_n_m, :] = lat_rad
        x[sel_state_p2_n_m, :] = long_rad
        x[sel_state_p3_n_m, :] = altitude_m_arr
        
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