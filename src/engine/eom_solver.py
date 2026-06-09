import math
import numpy as np

from src.engine.sim_data import SimData
from src.utils.interpolators import fastInterp1
from src.control.open_loop_control import open_loop_speed_brake, open_loop_throttle
from src.utils.kinematics import ecef_to_ned_dcm, quat_to_dcm, quat_to_dcm_vectorized, quaternion_derivative, wind_to_body_dcm

class eom_solver:
    def __init__(self, earth_model, wind_model, atmo_model):
        self.earth_model = earth_model # Injected Earth Model
        self.wind_model = wind_model
        self.atmo_model = atmo_model
    
    def solve_eom(self, t, x, dx, auxillary_data, u_trim, vehicle, cmod):

        # State Extraction
        u_b_mps, v_b_mps, w_b_mps = x[0], x[1], x[2]
        p_b_rps, q_b_rps, r_b_rps = x[3], x[4], x[5]
        q_b2e = x[6:10]
        x_e_m, y_e_m, z_e_m = x[10], x[11], x[12]
        m_fuel_kg = x[13]
        dela_ach_deg, dele_ach_deg, delr_ach_deg, delt_ach_pct = x[14], x[15], x[16], x[17]
        
        norm = math.sqrt(q_b2e[0]**2 + q_b2e[1]**2 + q_b2e[2]**2 + q_b2e[3]**2)
        q_b2e = q_b2e/norm

        # Vehicle Mass State Interface
        m_total_kg = vehicle.m_dry_kg + m_fuel_kg
        Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxy_b_kgm2, Jxz_b_kgm2, Jyz_b_kgm2 = vehicle.get_mass_properties(m_total_kg)
        
        speedbrake = cmod.get("speedbrake", False)

        # Control Routing
        delsb_deg = open_loop_speed_brake()
        
        # Engine Interface
        m_fuel_dot_kgps = vehicle.get_engine_burn_rate(delt_ach_pct)
        
        # Trim & Linearization Overrides
        dela_ach_deg_old, dele_ach_deg_old, delr_ach_deg_old, delt_ach_pct_old = dela_ach_deg, dele_ach_deg, delr_ach_deg, delt_ach_pct
        if cmod.get("trim_flag"):
            # Surfaces fixed
            dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = dela_ach_deg, dele_ach_deg, delr_ach_deg
            delt_cmd_pct = delt_ach_pct
        elif cmod.get("linearization_flag"):
            # Commanded values
            dela_cmd_deg, dele_cmd_deg, delr_cmd_deg = cmod['dela_cmd_deg'], cmod['dele_cmd_deg'], cmod['delr_cmd_deg']
            delt_cmd_pct = cmod['delt_cmd_pct']
        else:
            # Require the vehicle or SAS object to return control deflections
            dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, delt_cmd_pct = vehicle.get_sas_commands(t, x, cmod, u_trim)
        
        if cmod.get("type") == "time_history":
            dela_ach_deg, dele_ach_deg, delr_ach_deg = dela_cmd_deg, dele_cmd_deg, delr_cmd_deg
            x[13], x[14], x[15] = dela_ach_deg, dele_ach_deg, delr_ach_deg
        
        # Resolve Geodetics for Atmosphere and Aerodynamics
        lat_rad, long_rad, h_m = self.earth_model.ecef_to_geodetic(x_e_m, y_e_m, z_e_m)
        
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(long_rad), math.cos(long_rad)
        
        C_e2n = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
            [-sin_lon,            cos_lon,            0.0    ],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        
        C_b2e = quat_to_dcm(*q_b2e)
        C_e2b = C_b2e.T
        C_n2b = C_e2b @ C_e2n.T
        C_b2n = C_n2b.T
        
        # Wind Engine Interface
        W_N_mps, W_E_mps, W_D_mps = self.wind_model.get_velocity(h_m)
        dW_N_dh, dW_E_dh, dW_D_dh = self.wind_model.get_shear(h_m)
        
        W_n_mps = np.array([W_N_mps, W_E_mps, W_D_mps])
        W_b_mps = C_n2b @ W_n_mps

        # Air-Relative Translational States
        u_air_b_mps = u_b_mps - W_b_mps[0]
        v_air_b_mps = v_b_mps - W_b_mps[1]
        w_air_b_mps = w_b_mps - W_b_mps[2]

        # Atmosphere & Air Data
        rho_kgpm3 = fastInterp1(self.atmo_model["alt_m"], self.atmo_model["rho_kgpm3"], h_m)
        c_snd_mps = fastInterp1(self.atmo_model["alt_m"], self.atmo_model["c_mps"], h_m)
        
        true_airspeed_mps = math.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
        qbar_kgpms2 = 0.5 * rho_kgpm3 * true_airspeed_mps**2
        Mach = true_airspeed_mps / c_snd_mps if c_snd_mps > 0 else 0.0
        
        alpha_rad = math.atan2(w_air_b_mps, u_air_b_mps)
        beta_rad = math.asin(v_air_b_mps / true_airspeed_mps) if true_airspeed_mps > 0 else 0.0
        C_w2b = wind_to_body_dcm(alpha_rad, beta_rad)
        
        # Air-Relative Rotational States (Wind Shear Gradient Tensor)
        # Gradient of wind in NED (D = -h)
        grad_W_n = np.array([
            [0.0, 0.0, -dW_N_dh],
            [0.0, 0.0, -dW_E_dh],
            [0.0, 0.0, -dW_D_dh]
        ])
        
        # Transform gradient tensor to body frame
        grad_W_b = C_n2b @ grad_W_n @ C_b2n
        
        # Extract apparent aerodynamic rates induced by shear
        p_wind_b_rps = -grad_W_b[1, 2]
        q_wind_b_rps =  grad_W_b[0, 2]
        r_wind_b_rps = -grad_W_b[0, 1]

        # Calculate Air-Relative Body Rates for Aerodynamics
        p_air_b_rps = p_b_rps - p_wind_b_rps
        q_air_b_rps = q_b_rps - q_wind_b_rps
        r_air_b_rps = r_b_rps - r_wind_b_rps

        # Gravity & Earth Rates
        g_e_mps2 = self.earth_model.get_gravity_ecef(x_e_m, y_e_m, z_e_m)
        g_b_mps2 = C_e2b @ g_e_mps2
        omega_ie_b_rps = C_e2b @ self.earth_model.get_earth_rate_ecef()
        
        # Vehicle returns mapped body forces
        Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2, l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2 = vehicle.get_forces_and_moments(alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps,
                                                                                                                    p_air_b_rps, q_air_b_rps, r_air_b_rps, dele_ach_deg, dela_ach_deg, delr_ach_deg, delsb_deg, delt_ach_pct, C_w2b, speedbrake, h_m)

        # omega_ib_b_rps is the inertial body rate. The Coriolis acceleration for 
        # Earth-relative velocity tracked in the body frame requires (omega_ib + omega_ie) x V
        omega_ib_b_rps = np.array([p_b_rps, q_b_rps, r_b_rps])
        omega_cor_b_rps = omega_ib_b_rps + omega_ie_b_rps
        
        # Velocity Equations
        dx[0] = (Fx_b_kgmps2 / m_total_kg) + g_b_mps2[0] - (omega_cor_b_rps[1]*w_b_mps - omega_cor_b_rps[2]*v_b_mps)
        dx[1] = (Fy_b_kgmps2 / m_total_kg) + g_b_mps2[1] - (omega_cor_b_rps[2]*u_b_mps - omega_cor_b_rps[0]*w_b_mps)
        dx[2] = (Fz_b_kgmps2 / m_total_kg) + g_b_mps2[2] - (omega_cor_b_rps[0]*v_b_mps - omega_cor_b_rps[1]*u_b_mps)

        # Inertia Derivatives via Vehicle Method
        if (m_fuel_dot_kgps != 0):
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

            Jxx_dot, Jyy_dot, Jzz_dot, Jxy_dot, Jxz_dot, Jyz_dot = [dJ * -m_fuel_dot_kgps for dJ in dJ_dm]
        else:
            Jxx_dot, Jyy_dot, Jzz_dot, Jxy_dot, Jxz_dot, Jyz_dot = 0, 0, 0, 0, 0, 0

        # Angular Momentum (H = J * w)
        hx_b_kgm2ps =  Jxx_b_kgm2 * p_b_rps - Jxy_b_kgm2 * q_b_rps - Jxz_b_kgm2 * r_b_rps
        hy_b_kgm2ps = -Jxy_b_kgm2 * p_b_rps + Jyy_b_kgm2 * q_b_rps - Jyz_b_kgm2 * r_b_rps
        hz_b_kgm2ps = -Jxz_b_kgm2 * p_b_rps - Jyz_b_kgm2 * q_b_rps + Jzz_b_kgm2 * r_b_rps

        # Time derivative of inertia acting on angular velocity (J_dot * w)
        Idot_l_b_kgm2ps2 =  Jxx_dot * p_b_rps - Jxy_dot * q_b_rps - Jxz_dot * r_b_rps
        Idot_m_b_kgm2ps2 = -Jxy_dot * p_b_rps + Jyy_dot * q_b_rps - Jyz_dot * r_b_rps
        Idot_n_b_kgm2ps2 = -Jxz_dot * p_b_rps - Jyz_dot * q_b_rps + Jzz_dot * r_b_rps

        # Gyroscopic coupling (w x H)
        gyro_l_b_kgm2ps2 = q_b_rps * hz_b_kgm2ps - r_b_rps * hy_b_kgm2ps
        gyro_m_b_kgm2ps2 = r_b_rps * hx_b_kgm2ps - p_b_rps * hz_b_kgm2ps
        gyro_n_b_kgm2ps2 = p_b_rps * hy_b_kgm2ps - q_b_rps * hx_b_kgm2ps

        # Net moment vector M_net = M_ext - J_dot*w - w x H
        l_tot = l_b_kgm2ps2 - Idot_l_b_kgm2ps2 - gyro_l_b_kgm2ps2
        m_tot = m_b_kgm2ps2 - Idot_m_b_kgm2ps2 - gyro_m_b_kgm2ps2
        n_tot = n_b_kgm2ps2 - Idot_n_b_kgm2ps2 - gyro_n_b_kgm2ps2

        # Analytical inverse of 3x3 symmetric Inertia Tensor
        C11 = Jyy_b_kgm2 * Jzz_b_kgm2 - Jyz_b_kgm2**2
        C22 = Jxx_b_kgm2 * Jzz_b_kgm2 - Jxz_b_kgm2**2
        C33 = Jxx_b_kgm2 * Jyy_b_kgm2 - Jxy_b_kgm2**2
        C12 = Jxy_b_kgm2 * Jzz_b_kgm2 + Jxz_b_kgm2 * Jyz_b_kgm2
        C13 = Jxy_b_kgm2 * Jyz_b_kgm2 + Jyy_b_kgm2 * Jxz_b_kgm2
        C23 = Jxx_b_kgm2 * Jyz_b_kgm2 + Jxy_b_kgm2 * Jxz_b_kgm2

        det_J = Jxx_b_kgm2 * C11 - Jxy_b_kgm2 * C12 - Jxz_b_kgm2 * C13
        inv_det = 1.0 / det_J

        # Final state derivatives for angular velocity
        dx[3] = (C11 * l_tot + C12 * m_tot + C13 * n_tot) * inv_det
        dx[4] = (C12 * l_tot + C22 * m_tot + C23 * n_tot) * inv_det
        dx[5] = (C13 * l_tot + C23 * m_tot + C33 * n_tot) * inv_det
        
        # Quaternion rates depend on Earth-Relative Body Rates
        omega_eb_b_rps = omega_ib_b_rps - omega_ie_b_rps
        dx[6:10] = quaternion_derivative(q_b2e, omega_eb_b_rps)

        # Navigation (Cartesian Velocity Integration)
        dx[10:13] = C_b2e @ np.array([u_b_mps, v_b_mps, w_b_mps])
        
        # Fuel
        dx[13] = -m_fuel_dot_kgps

        # Actuation
        dx[14] = vehicle.aileron_kinematics(dela_cmd_deg, dela_ach_deg_old)
        dx[15] = vehicle.elevator_kinematics(dele_cmd_deg, dele_ach_deg_old)
        dx[16] = vehicle.rudder_kinematics(delr_cmd_deg, delr_ach_deg_old)
        dx[17] = vehicle.throttle_kinematics(delt_cmd_pct, delt_ach_pct_old)

        # Nav-Relative Body Rates
        v_e_mps = dx[10:13]
        
        v_n_mps = C_e2n @ v_e_mps
        den_wgs84 = math.sqrt(1.0 - (self.earth_model.e * sin_lat)**2)
        RN_m = self.earth_model.a / den_wgs84
        RM_m = (self.earth_model.a * (1.0 - self.earth_model.e**2)) / (den_wgs84**3)
        
        omega_en_n_rps = np.array([v_n_mps[1] / (RN_m + h_m), -v_n_mps[0] / (RM_m + h_m), -v_n_mps[1] * math.tan(lat_rad) / (RN_m + h_m)])
        omega_ie_n_rps = np.array([self.earth_model.omega_rps * cos_lat, 0.0, -self.earth_model.omega_rps * sin_lat])
        
        omega_in_b_rps = C_n2b @ (omega_ie_n_rps + omega_en_n_rps)
        omega_nb_b_rps = omega_ib_b_rps - omega_in_b_rps

        # Aux Data Output
        auxillary_data[0:4] = [dela_cmd_deg, dele_cmd_deg, delr_cmd_deg, delt_cmd_pct]
        auxillary_data[4:7] = omega_nb_b_rps
        auxillary_data[7:10] = [Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2]
        auxillary_data[10:13] = [l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2]
        auxillary_data[13:16] = [W_N_mps, W_E_mps, W_D_mps]

        return dx, auxillary_data
    
    def post_process(self, x, t_s, auxillary_data):
        # Flatten state rows immediately
        u_b_mps, v_b_mps, w_b_mps = x[0, :], x[1, :], x[2, :]
        p_b_rps, q_b_rps, r_b_rps = x[3, :], x[4, :], x[5, :]
        q_b2e = x[6:10, :]
        x_e, y_e, z_e = x[10, :], x[11, :], x[12, :]
        
        nt = len(t_s)
        lat_rad, long_rad, h_m = np.zeros(nt), np.zeros(nt), np.zeros(nt)
        
        # Geodetic recovery over time array
        vectorized_ecef_to_geodetic = np.vectorize(self.earth_model.ecef_to_geodetic)
        lat_rad, long_rad, h_m = vectorized_ecef_to_geodetic(x_e, y_e, z_e)
        
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(long_rad), np.cos(long_rad)

        C_e2n = np.zeros((nt, 3, 3))
        C_e2n[:, 0, 0] = -sin_lat * cos_lon
        C_e2n[:, 0, 1] = -sin_lat * sin_lon
        C_e2n[:, 0, 2] =  cos_lat
        C_e2n[:, 1, 0] = -sin_lon
        C_e2n[:, 1, 1] =  cos_lon
        C_e2n[:, 1, 2] =  0.0
        C_e2n[:, 2, 0] = -cos_lat * cos_lon
        C_e2n[:, 2, 1] = -cos_lat * sin_lon
        C_e2n[:, 2, 2] = -sin_lat
        
        C_b2e = quat_to_dcm_vectorized(*q_b2e)
        C_e2b = C_b2e.transpose(0, 2, 1)
        C_n2b = C_e2b @ C_e2n.transpose(0, 2, 1)
        C_b2n = C_n2b.transpose(0, 2, 1)
        
        W_n_mps = auxillary_data[13:16, :]
        # W_b_mps = C_n2b @ W_n_mps
        # Use einsum for (nt, 3, 3) @ (3, nt) -> (3, nt)
        W_b_mps = np.einsum('nij, jn -> in', C_n2b, W_n_mps)

        # Air-Relative Translational States
        u_air_b_mps = u_b_mps - W_b_mps[0]
        v_air_b_mps = v_b_mps - W_b_mps[1]
        w_air_b_mps = w_b_mps - W_b_mps[2]

        # Atmosphere & Air Data
        rho_kgpm3 = np.array([fastInterp1(self.atmo_model["alt_m"], self.atmo_model["rho_kgpm3"], alt) for alt in h_m])
        c_snd_mps = np.array([fastInterp1(self.atmo_model["alt_m"], self.atmo_model["c_mps"],     alt) for alt in h_m])
        p_kgpms2  = np.array([fastInterp1(self.atmo_model["alt_m"], self.atmo_model["p_Npm2"],    alt) for alt in h_m])
        T_K       = np.array([fastInterp1(self.atmo_model["alt_m"], self.atmo_model["T_K"],       alt) for alt in h_m])
        
        true_airspeed_mps = np.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
        mach = np.divide(true_airspeed_mps, c_snd_mps, out=np.zeros_like(true_airspeed_mps), where=c_snd_mps != 0)
        
        alpha_rad = np.arctan2(w_air_b_mps, u_air_b_mps)
        vt_ratio = np.divide(v_air_b_mps, true_airspeed_mps, out=np.zeros_like(v_air_b_mps), where=true_airspeed_mps != 0)
        beta_rad = np.arcsin(np.clip(vt_ratio, -1.0, 1.0))
        
        # Determine gravity magnitude from Earth model
        g_vecs = np.array([self.earth_model.get_gravity_ecef(x, y, z) for x, y, z in zip(x_e, y_e, z_e)])
        g_mag_mps2 = np.linalg.norm(g_vecs, axis=1)

        # Calculate Euler angles directly from the DCM
        # These will produce 'raw' angles with jump discontinuities at the branch cuts
        phi_rad   = np.arctan2(C_b2n[:, 2, 1], C_b2n[:, 2, 2])
        theta_rad = np.arcsin(np.clip(-C_b2n[:, 2, 0], -1.0, 1.0))
        psi_rad   = np.arctan2(C_b2n[:, 1, 0], C_b2n[:, 0, 0])

        # Apply global unwrapping to smooth the trajectories
        # This relies on the time-series continuity, not on switching calculation methods.
        # phi_rad   = np.unwrap(phi_rad)
        # psi_rad   = np.unwrap(psi_rad)
        # theta_rad = np.unwrap(theta_rad)
        
        # Nav Velocities
        vel_b = np.stack([u_b_mps, v_b_mps, w_b_mps], axis=1) # (nt, 3)
        vel_n = np.einsum('nij, nj -> ni', C_b2n, vel_b) # (nt, 3)
        u_n_mps, v_n_mps, w_n_mps = vel_n[:, 0], vel_n[:, 1], vel_n[:, 2]
        
        # Extract Nav-Relative Body Rates
        p_nb_rps = auxillary_data[4, :]
        q_nb_rps = auxillary_data[5, :]
        r_nb_rps = auxillary_data[6, :]

        # Euler Rates Kinematics (Must map from Nav-Relative Rates)
        phi_dot_rps   = p_nb_rps + (q_nb_rps * np.sin(phi_rad) + r_nb_rps * np.cos(phi_rad)) * np.tan(theta_rad)
        theta_dot_rps = q_nb_rps * np.cos(phi_rad) - r_nb_rps * np.sin(phi_rad)
        psi_dot_rps   = (q_nb_rps * np.sin(phi_rad) + r_nb_rps * np.cos(phi_rad)) / np.cos(theta_rad)

        return SimData(
            t_s=t_s.flatten(),
            u_b_mps=u_b_mps, v_b_mps=v_b_mps, w_b_mps=w_b_mps,
            p_b_rps=p_b_rps, q_b_rps=q_b_rps, r_b_rps=r_b_rps,
            q0=q_b2e[0,:], q1=q_b2e[1,:], q2=q_b2e[2,:], q3=q_b2e[3,:],
            lat_rad=lat_rad, long_rad=long_rad, h_m=h_m,
            m_fuel_kg=x[13, :],
            
            phi_rad=phi_rad, theta_rad=theta_rad, psi_rad=psi_rad,
            phi_dot_rps=phi_dot_rps, theta_dot_rps=theta_dot_rps, psi_dot_rps=psi_dot_rps,
            p_nb_rps=p_nb_rps, q_nb_rps=q_nb_rps, r_nb_rps=r_nb_rps,
            
            cs_mps=c_snd_mps, rho_kgpm3=rho_kgpm3, p_kgpms2=p_kgpms2, T_K=T_K,
            mach=mach, alpha_rad=alpha_rad, beta_rad=beta_rad, 
            true_airspeed_mps=true_airspeed_mps, g_mag_mps2=g_mag_mps2,
            
            u_n_mps=u_n_mps, v_n_mps=v_n_mps, w_n_mps=w_n_mps,
            
            Fx_b_kgmps2=auxillary_data[7, :], Fy_b_kgmps2=auxillary_data[8, :], Fz_b_kgmps2=auxillary_data[9, :],
            l_b_kgm2ps2=auxillary_data[10, :], m_b_kgm2ps2=auxillary_data[11, :], n_b_kgm2ps2=auxillary_data[12, :],
            
            dela_ach_deg=x[14, :], dele_ach_deg=x[15, :],
            delr_ach_deg=x[16, :], delt_ach_pct=x[17, :],
            dela_cmd_deg=auxillary_data[0, :], dele_cmd_deg=auxillary_data[1, :],
            delr_cmd_deg=auxillary_data[2, :], delt_cmd_pct=auxillary_data[3, :],
            
            W_N_mps=W_n_mps[0, :], W_E_mps=W_n_mps[1, :], W_D_mps=W_n_mps[2, :]
        )