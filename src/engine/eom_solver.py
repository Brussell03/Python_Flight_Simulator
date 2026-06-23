import math
import numpy as np
from numba import njit
from src.engine.sim_data import SimData
from src.engine.state_mapping import AuxIdx, StateIdxSlices, StateIdx, AuxIdxSlices, TrimStateIdxSlices

from src.utils.math_utils import (
    euler_rates_vectorized, quat_to_dcm, ecef_to_ned_dcm, quat_to_dcm_vectorized, wind_to_body_dcm, b2n_dcm_to_euler,
    true_airspeed, dynamic_pressure, angle_of_attack, angle_of_sideslip,
    quaternion_derivative
)

# ==========================================
# 1. COMPILED KERNELS (Numba @njit)
# ==========================================

@njit
def compute_kinematics(x, earth_model, wind_model, atmo_model):
    """
    Executes the first half of the equations of motion natively in C.
    """
    u_b_mps, v_b_mps, w_b_mps = x[StateIdx.U_B_MPS : StateIdx.W_B_MPS + 1]
    q_b2e = x[StateIdx.Q0 : StateIdx.Q3 + 1]
    x_e_m, y_e_m, z_e_m = x[StateIdx.X_E_M : StateIdx.Z_E_M + 1]
    
    # Quaternion normalization
    norm = math.sqrt(q_b2e[0]**2 + q_b2e[1]**2 + q_b2e[2]**2 + q_b2e[3]**2)
    q_b2e = q_b2e / norm

    # Earth Model jitclass methods
    lat_rad, long_rad, h_m = earth_model.ecef_to_geodetic(x_e_m, y_e_m, z_e_m)
    
    C_e2n = ecef_to_ned_dcm(lat_rad, long_rad)
    C_b2e = quat_to_dcm(q_b2e[0], q_b2e[1], q_b2e[2], q_b2e[3])
    C_e2b = C_b2e.T
    C_n2b = C_e2b @ C_e2n.T
    C_b2n = C_n2b.T

    # Wind Model jitclass methods
    W_N_mps, W_E_mps, W_D_mps = wind_model.get_velocity(h_m)
    dW_N_dh, dW_E_dh, dW_D_dh = wind_model.get_shear(h_m)
    W_n_mps = np.array([W_N_mps, W_E_mps, W_D_mps], dtype=np.float64)
    W_b_mps = C_n2b @ W_n_mps

    # Air-Relative Translational States
    u_air_b_mps = u_b_mps - W_b_mps[0]
    v_air_b_mps = v_b_mps - W_b_mps[1]
    w_air_b_mps = w_b_mps - W_b_mps[2]

    # Atmosphere & Air Data
    rho_kgpm3, c_snd_mps = atmo_model.get_density_and_soundspeed(h_m)
    
    true_airspeed_mps = true_airspeed(u_air_b_mps, v_air_b_mps, w_air_b_mps)
    qbar_kgpms2 = dynamic_pressure(rho_kgpm3, true_airspeed_mps)
    Mach = true_airspeed_mps / c_snd_mps if c_snd_mps > 0 else 0.0
    
    alpha_rad = angle_of_attack(u_air_b_mps, w_air_b_mps)
    beta_rad = angle_of_sideslip(v_air_b_mps, true_airspeed_mps)
    C_w2b = wind_to_body_dcm(alpha_rad, beta_rad)

    phi_rad, theta_rad, psi_rad = b2n_dcm_to_euler(C_b2n)

    # Wind shear tensor mapping
    grad_W_n = np.array([
        [0.0, 0.0, -dW_N_dh],
        [0.0, 0.0, -dW_E_dh],
        [0.0, 0.0, -dW_D_dh]
    ], dtype=np.float64)
    grad_W_b = C_n2b @ grad_W_n @ C_b2n

    p_wind_b_rps = -grad_W_b[1, 2]
    q_wind_b_rps =  grad_W_b[0, 2]
    r_wind_b_rps = -grad_W_b[0, 1]

    return (q_b2e, C_e2b, C_n2b, C_b2n, C_e2n, C_w2b, 
            u_air_b_mps, v_air_b_mps, w_air_b_mps, true_airspeed_mps, qbar_kgpms2, Mach, 
            alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, 
            lat_rad, long_rad, h_m, p_wind_b_rps, q_wind_b_rps, r_wind_b_rps,
            W_N_mps, W_E_mps, W_D_mps, rho_kgpm3)

@njit
def compute_dynamics(x, dx, earth_model, mass_props, mass_props_dot, forces, moments, 
                     omega_ib_b_rps, C_e2b, C_e2n, C_n2b, m_fuel_dot_kgps, lat_rad, h_m):
    """
    Executes the integration and rigid-body dynamics natively in C.
    """
    u_b_mps, v_b_mps, w_b_mps = x[StateIdx.U_B_MPS : StateIdx.W_B_MPS + 1]
    p_b_rps, q_b_rps, r_b_rps = omega_ib_b_rps
    q_b2e = x[StateIdx.Q0 : StateIdx.Q3 + 1]
    x_e_m, y_e_m, z_e_m = x[StateIdx.X_E_M : StateIdx.Z_E_M + 1]

    m_total_kg = mass_props[0]
    Jxx_b_kgm2, Jyy_b_kgm2, Jzz_b_kgm2, Jxy_b_kgm2, Jxz_b_kgm2, Jyz_b_kgm2 = mass_props[1:7]
    Jxx_dot, Jyy_dot, Jzz_dot, Jxy_dot, Jxz_dot, Jyz_dot = mass_props_dot

    Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2 = forces
    l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2 = moments

    g_e_mps2 = earth_model.get_gravity_ecef(x_e_m, y_e_m, z_e_m)
    g_b_mps2 = C_e2b @ np.array(g_e_mps2, dtype=np.float64)
    
    omega_ie_e_rps = earth_model.get_earth_rate_ecef()
    omega_ie_b_rps = C_e2b @ np.array(omega_ie_e_rps, dtype=np.float64)
    omega_cor_b_rps = omega_ib_b_rps + omega_ie_b_rps

    # Velocity Equations
    dx[StateIdx.U_B_MPS] = (Fx_b_kgmps2 / m_total_kg) + g_b_mps2[0] - (omega_cor_b_rps[1]*w_b_mps - omega_cor_b_rps[2]*v_b_mps)
    dx[StateIdx.V_B_MPS] = (Fy_b_kgmps2 / m_total_kg) + g_b_mps2[1] - (omega_cor_b_rps[2]*u_b_mps - omega_cor_b_rps[0]*w_b_mps)
    dx[StateIdx.W_B_MPS] = (Fz_b_kgmps2 / m_total_kg) + g_b_mps2[2] - (omega_cor_b_rps[0]*v_b_mps - omega_cor_b_rps[1]*u_b_mps)

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

    inv_det = 1.0 / (Jxx_b_kgm2 * C11 - Jxy_b_kgm2 * C12 - Jxz_b_kgm2 * C13)

    # Final state derivatives for angular velocity
    dx[StateIdx.P_B_RPS] = (C11 * l_tot + C12 * m_tot + C13 * n_tot) * inv_det
    dx[StateIdx.Q_B_RPS] = (C12 * l_tot + C22 * m_tot + C23 * n_tot) * inv_det
    dx[StateIdx.R_B_RPS] = (C13 * l_tot + C23 * m_tot + C33 * n_tot) * inv_det

    # Quaternion rates depend on Earth-Relative Body Rates
    omega_eb_b_rps = omega_ib_b_rps - omega_ie_b_rps
    dx[StateIdx.Q0 : StateIdx.Q3 + 1] = quaternion_derivative(q_b2e, omega_eb_b_rps, 1.0)

    # Navigation (Cartesian Velocity Integration)
    dx[StateIdx.X_E_M : StateIdx.Z_E_M + 1] = C_e2b.T @ np.array([u_b_mps, v_b_mps, w_b_mps], dtype=np.float64)
    dx[StateIdx.M_FUEL_KG] = -m_fuel_dot_kgps

    # Earth rates
    v_n_mps = C_e2n @ dx[StateIdx.X_E_M : StateIdx.Z_E_M + 1]
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    den_wgs84 = math.sqrt(1.0 - (earth_model.e * sin_lat)**2)
    RN_m = earth_model.a / den_wgs84
    RM_m = (earth_model.a * (1.0 - earth_model.e**2)) / (den_wgs84**3)

    omega_en_n_rps = np.array([v_n_mps[1] / (RN_m + h_m), -v_n_mps[0] / (RM_m + h_m), -v_n_mps[1] * math.tan(lat_rad) / (RN_m + h_m)])
    omega_ie_n_rps = np.array([earth_model.omega_rps * cos_lat, 0.0, -earth_model.omega_rps * sin_lat])
    
    omega_in_b_rps = C_n2b @ (omega_ie_n_rps + omega_en_n_rps)
    omega_nb_b_rps = omega_ib_b_rps - omega_in_b_rps

    return dx, omega_nb_b_rps

# ==========================================
# 2. PYTHON OBJECT MANAGER
# ==========================================

class eom_solver:
    def __init__(self, earth_model, wind_model, atmo_model, vehicle, control_model):
        self.earth_model = earth_model     # Jitclass
        self.wind_model = wind_model       # Jitclass
        self.atmo_model = atmo_model       # Jitclass
        self.vehicle = vehicle             # Standard Python Class
        self.control_model = control_model # Standard Dictionary
    
    def solve_eom(self, t, x, dx, auxillary_data, x_trim_ref):
        p_b_rps, q_b_rps, r_b_rps = x[StateIdxSlices.ROT_SLICE]
        dela_ach_rad, dele_ach_rad, delr_ach_rad, delt_ach_pct = x[StateIdxSlices.ACT_SLICE]
        
        # --- 1. COMPILED KINEMATICS ---
        # Pass arrays and jitclasses directly into the C-compiled engine
        (q_b2e, C_e2b, C_n2b, C_b2n, C_e2n, C_w2b, 
         u_air_b_mps, v_air_b_mps, w_air_b_mps, true_airspeed_mps, qbar_kgpms2, Mach, 
         alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, 
         lat_rad, long_rad, h_m, p_wind_b_rps, q_wind_b_rps, r_wind_b_rps,
         W_N_mps, W_E_mps, W_D_mps, rho_kgpm3) = compute_kinematics(
             x, self.earth_model, self.wind_model, self.atmo_model
        )

        # --- 2. UNCOMPILED PYTHON LOGIC (Vehicle, Dictionaries, Routing) ---
        self.vehicle.set_gnc_inputs(t, self.control_model, self.atmo_model, lat_rad, long_rad, h_m, 
                                    alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, 
                                    p_b_rps, q_b_rps, r_b_rps, true_airspeed_mps, rho_kgpm3, x_trim_ref)
        
        speedbrake = self.control_model.get("speedbrake", False)
        delsb_deg = 0.0

        # Air-Relative Rates
        p_air_b_rps = p_b_rps - p_wind_b_rps
        q_air_b_rps = q_b_rps - q_wind_b_rps
        r_air_b_rps = r_b_rps - r_wind_b_rps

        # Actuation & Aerodynamics 
        dela_cmd_rad, dele_cmd_rad, delr_cmd_rad, delt_cmd_pct = self._route_controls(t, x, dela_ach_rad, dele_ach_rad, delr_ach_rad, delt_ach_pct, x_trim_ref)

        if self.control_model.get("type") in ["no_lag", "time_history"]:
            dela_ach_rad, dele_ach_rad, delr_ach_rad, delt_ach_pct = dela_cmd_rad, dele_cmd_rad, delr_cmd_rad, delt_cmd_pct
            x[StateIdxSlices.ACT_SLICE] = [dela_ach_rad, dele_ach_rad, delr_ach_rad, delt_ach_pct]

        Fx_b, Fy_b, Fz_b, l_b, m_b, n_b = self.vehicle.get_forces_and_moments(
            alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps,
            p_air_b_rps, q_air_b_rps, r_air_b_rps, dele_ach_rad, dela_ach_rad, delr_ach_rad, 
            delsb_deg, delt_ach_pct, C_w2b, speedbrake, h_m
        )

        m_total_kg = self.vehicle.m_dry_kg + x[StateIdx.M_FUEL_KG]
        mass_props = np.array([m_total_kg, *self.vehicle.get_mass_properties(m_total_kg)], dtype=np.float64)
        
        m_fuel_dot_kgps = self.vehicle.get_engine_burn_rate(delt_ach_pct)
        mass_props_dot = self._compute_inertia_derivatives(m_total_kg, m_fuel_dot_kgps)

        # --- 3. COMPILED DYNAMICS & INTEGRATION ---
        forces = np.array([Fx_b, Fy_b, Fz_b], dtype=np.float64)
        moments = np.array([l_b, m_b, n_b], dtype=np.float64)
        omega_ib_b_rps = np.array([p_b_rps, q_b_rps, r_b_rps], dtype=np.float64)

        dx, omega_nb_b_rps = compute_dynamics(
            x, dx, self.earth_model, mass_props, mass_props_dot, forces, moments, 
            omega_ib_b_rps, C_e2b, C_e2n, C_n2b, m_fuel_dot_kgps, lat_rad, h_m
        )

        # --- 4. PYTHON ACTUATOR KINEMATICS & AUX DATA ---
        dx[StateIdx.DELA_ACH_RAD] = self.vehicle.aileron_kinematics(dela_cmd_rad, dela_ach_rad)
        dx[StateIdx.DELE_ACH_RAD] = self.vehicle.elevator_kinematics(dele_cmd_rad, dele_ach_rad)
        dx[StateIdx.DELR_ACH_RAD] = self.vehicle.rudder_kinematics(delr_cmd_rad, delr_ach_rad)
        dx[StateIdx.DELT_ACH_PCT] = self.vehicle.throttle_kinematics(delt_cmd_pct, delt_ach_pct)

        auxillary_data[AuxIdxSlices.CMD_SLICE] = [dela_cmd_rad, dele_cmd_rad, delr_cmd_rad, delt_cmd_pct]
        auxillary_data[AuxIdxSlices.NAV_RATE_SLICE] = omega_nb_b_rps
        auxillary_data[AuxIdxSlices.FORCE_SLICE] = forces
        auxillary_data[AuxIdxSlices.MOMENT_SLICE] = moments
        auxillary_data[AuxIdxSlices.WIND_SLICE] = [W_N_mps, W_E_mps, W_D_mps]
        if x_trim_ref is not None: auxillary_data[AuxIdxSlices.TRIM_SLICE] = x_trim_ref[TrimStateIdxSlices.ACT_TRIM_SLICE]
        else: auxillary_data[AuxIdxSlices.TRIM_SLICE] = 0, 0, 0, 0

        return dx, auxillary_data

    def _route_controls(self, t, x, dela_ach_rad, dele_ach_rad, delr_ach_rad, delt_ach_pct, x_trim_ref):
        if self.control_model.get("trim_flag"):
            return dela_ach_rad, dele_ach_rad, delr_ach_rad, delt_ach_pct
        elif self.control_model.get("linearization_flag"):
            return self.control_model['dela_cmd_rad'], self.control_model['dele_cmd_rad'], self.control_model['delr_cmd_rad'], self.control_model['delt_cmd_pct']
        else:
            return self.vehicle.get_sas_commands(t, x, self.control_model, x_trim_ref)

    def _compute_inertia_derivatives(self, m_total_kg, m_fuel_dot_kgps):
        if m_fuel_dot_kgps == 0:
            return np.zeros(6, dtype=np.float64)
            
        dm_kg = 1.0
        m_plus = np.clip(m_total_kg + dm_kg, self.vehicle.m_dry_kg, self.vehicle.m_wet_kg)
        m_minus = np.clip(m_total_kg - dm_kg, self.vehicle.m_dry_kg, self.vehicle.m_wet_kg)
        dm_diff = m_plus - m_minus

        if dm_diff > 0:
            J_plus = self.vehicle.get_mass_properties(m_plus)
            J_minus = self.vehicle.get_mass_properties(m_minus)
            dJ_dm = [(p - m) / dm_diff for p, m in zip(J_plus, J_minus)]
        else:
            dJ_dm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return np.array([dJ * -m_fuel_dot_kgps for dJ in dJ_dm], dtype=np.float64)
    
    def post_process(self, x, t_s, auxillary_data, job_name, description, integrator):
        # Flatten state rows immediately
        u_b_mps, v_b_mps, w_b_mps = x[StateIdx.U_B_MPS, :], x[StateIdx.V_B_MPS, :], x[StateIdx.W_B_MPS, :]
        p_b_rps, q_b_rps, r_b_rps = x[StateIdx.P_B_RPS, :], x[StateIdx.Q_B_RPS, :], x[StateIdx.R_B_RPS, :]
        q_b2e = x[StateIdxSlices.QUAT_SLICE, :]
        x_e_m, y_e_m, z_e_m = x[StateIdx.X_E_M, :], x[StateIdx.Y_E_M, :], x[StateIdx.Z_E_M, :]
        m_fuel_kg=x[StateIdx.M_FUEL_KG, :]
        
        nt = len(t_s)
        lat_rad, long_rad, h_m = np.zeros(nt), np.zeros(nt), np.zeros(nt)
        
        # Geodetic recovery over time array
        vectorized_ecef_to_geodetic = np.vectorize(self.earth_model.ecef_to_geodetic)
        lat_rad, long_rad, h_m = vectorized_ecef_to_geodetic(x_e_m, y_e_m, z_e_m)
        
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
        
        W_n_mps = auxillary_data[AuxIdxSlices.WIND_SLICE, :]
        # Use einsum for (nt, 3, 3) @ (3, nt) -> (3, nt)
        W_b_mps = np.einsum('nij, jn -> in', C_n2b, W_n_mps)

        # Air-Relative Translational States
        u_air_b_mps = u_b_mps - W_b_mps[0]
        v_air_b_mps = v_b_mps - W_b_mps[1]
        w_air_b_mps = w_b_mps - W_b_mps[2]

        # Atmosphere & Air Data
        rho_kgpm3, c_snd_mps, p_kgpms2, T_K = self.atmo_model.get_atmo_properties_vectorized(h_m)
        
        true_airspeed_mps = np.sqrt(u_air_b_mps**2 + v_air_b_mps**2 + w_air_b_mps**2)
        mach = np.divide(true_airspeed_mps, c_snd_mps, out=np.zeros_like(true_airspeed_mps), where=c_snd_mps != 0)
        
        qbar_kgpms2 = 0.5 * rho_kgpm3 * true_airspeed_mps**2
        
        alpha_rad = np.arctan2(w_air_b_mps, u_air_b_mps)
        vt_ratio = np.divide(v_air_b_mps, true_airspeed_mps, out=np.zeros_like(v_air_b_mps), where=true_airspeed_mps != 0)
        beta_rad = np.arcsin(np.clip(vt_ratio, -1.0, 1.0))
        
        # Determine gravity magnitude from Earth model
        g_vecs = np.array([self.earth_model.get_gravity_ecef(x, y, z) for x, y, z in zip(x_e_m, y_e_m, z_e_m)])
        g_mag_mps2 = np.linalg.norm(g_vecs, axis=1)

        # Calculate Euler angles directly from the DCM
        # These will produce 'raw' angles with jump discontinuities at the branch cuts
        phi_rad   = np.arctan2(C_b2n[:, 2, 1], C_b2n[:, 2, 2])
        theta_rad = np.arcsin(np.clip(-C_b2n[:, 2, 0], -1.0, 1.0))
        psi_rad   = np.arctan2(C_b2n[:, 1, 0], C_b2n[:, 0, 0])

        # Apply global unwrapping to smooth the trajectories
        # phi_rad   = np.unwrap(phi_rad)
        # psi_rad   = np.unwrap(psi_rad)
        # theta_rad = np.unwrap(theta_rad)
        
        # Nav Velocities
        vel_b = np.stack([u_b_mps, v_b_mps, w_b_mps], axis=1) # (nt, 3)
        vel_n = np.einsum('nij, nj -> ni', C_b2n, vel_b) # (nt, 3)
        u_n_mps, v_n_mps, w_n_mps = vel_n[:, 0], vel_n[:, 1], vel_n[:, 2]
        
        # Extract Nav-Relative Body Rates
        p_nb_rps = auxillary_data[AuxIdx.P_NB_RPS, :]
        q_nb_rps = auxillary_data[AuxIdx.Q_NB_RPS, :]
        r_nb_rps = auxillary_data[AuxIdx.R_NB_RPS, :]

        # Euler Rates Kinematics (Must map from Nav-Relative Rates)
        phi_dot_rps, theta_dot_rps, psi_dot_rps = euler_rates_vectorized(phi_rad, theta_rad, p_nb_rps, q_nb_rps, r_nb_rps)
        
        # Load factors
        Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2 = auxillary_data[AuxIdxSlices.FORCE_SLICE, :]
        vehicle_weight = (self.vehicle.m_dry_kg + m_fuel_kg) * self.earth_model.g0
        n_x = Fx_b_kgmps2 / vehicle_weight
        n_y = Fy_b_kgmps2 / vehicle_weight
        n_z = -Fz_b_kgmps2 / vehicle_weight

        return SimData(
            job_name=job_name,
            description=description,
            integrator=integrator,
            vehicle=self.vehicle,
            earth_model=self.earth_model,
            wind_model=self.wind_model,
    
            t_s=t_s.flatten(),
            u_b_mps=u_b_mps, v_b_mps=v_b_mps, w_b_mps=w_b_mps,
            p_b_rps=p_b_rps, q_b_rps=q_b_rps, r_b_rps=r_b_rps,
            q0=q_b2e[0,:], q1=q_b2e[1,:], q2=q_b2e[2,:], q3=q_b2e[3,:],
            x_e_m=x_e_m, y_e_m=y_e_m, z_e_m=z_e_m,
            lat_rad=lat_rad, long_rad=long_rad, h_m=h_m,
            m_fuel_kg=m_fuel_kg,
            
            phi_rad=phi_rad, theta_rad=theta_rad, psi_rad=psi_rad,
            phi_dot_rps=phi_dot_rps, theta_dot_rps=theta_dot_rps, psi_dot_rps=psi_dot_rps,
            p_nb_rps=p_nb_rps, q_nb_rps=q_nb_rps, r_nb_rps=r_nb_rps,
            
            cs_mps=c_snd_mps, rho_kgpm3=rho_kgpm3, p_kgpms2=p_kgpms2, T_K=T_K,
            
            mach=mach, alpha_rad=alpha_rad, beta_rad=beta_rad, 
            true_airspeed_mps=true_airspeed_mps,
            qbar_kgpms2=qbar_kgpms2, g_mag_mps2=g_mag_mps2,
            
            u_n_mps=u_n_mps, v_n_mps=v_n_mps, w_n_mps=w_n_mps,
            
            Fx_b_kgmps2=Fx_b_kgmps2, Fy_b_kgmps2=Fy_b_kgmps2, Fz_b_kgmps2=Fz_b_kgmps2,
            l_b_kgm2ps2=auxillary_data[AuxIdx.L_B_KGM2PS2, :], m_b_kgm2ps2=auxillary_data[AuxIdx.M_B_KGM2PS2, :], n_b_kgm2ps2=auxillary_data[AuxIdx.N_B_KGM2PS2, :],
            n_x=n_x, n_y=n_y, n_z=n_z,
            
            dela_ach_rad=x[StateIdx.DELA_ACH_RAD, :], dele_ach_rad=x[StateIdx.DELE_ACH_RAD, :],
            delr_ach_rad=x[StateIdx.DELR_ACH_RAD, :], delt_ach_pct=x[StateIdx.DELT_ACH_PCT, :],
            
            dela_cmd_rad=auxillary_data[AuxIdx.DELA_CMD_RAD, :], dele_cmd_rad=auxillary_data[AuxIdx.DELE_CMD_RAD, :],
            delr_cmd_rad=auxillary_data[AuxIdx.DELR_CMD_RAD, :], delt_cmd_pct=auxillary_data[AuxIdx.DELT_CMD_PCT, :],
            
            dela_trim_rad=auxillary_data[AuxIdx.DELA_TRIM_RAD, :], dele_trim_rad=auxillary_data[AuxIdx.DELE_TRIM_RAD, :],
            delr_trim_rad=auxillary_data[AuxIdx.DELR_TRIM_RAD, :], delt_trim_pct=auxillary_data[AuxIdx.DELT_TRIM_PCT, :],
            
            W_N_mps=W_n_mps[0, :], W_E_mps=W_n_mps[1, :], W_D_mps=W_n_mps[2, :]
        )