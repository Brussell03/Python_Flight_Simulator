from dataclasses import dataclass
import numpy as np

from src.utils.constants import R2D

@dataclass
class SimData:
    t_s: np.ndarray                 # simulation time steps in seconds
    u_b_mps: np.ndarray             # axial velocity of CM wrt inertial CS resolved in aircraft body fixed CS [m/s]
    v_b_mps: np.ndarray             # lateral velocity of CM wrt inertial CS resolved in aircraft body fixed CS [m/s]
    w_b_mps: np.ndarray             # normal velocity of CM wrt inertial CS resolved in aircraft body fixed CS [m/s]
    p_b_rps: np.ndarray             # roll angular velocity of body fixed CS with respect to inertial CS [rad/s]
    q_b_rps: np.ndarray             # pitch angular velocity of body fixed CS with respect to inertial CS [rad/s]
    r_b_rps: np.ndarray             # yaw angular velocity of body fixed CS with respect to inertial CS [rad/s]
    q0: np.ndarray                  # quaternion scalar component representation of body wrt local navigation frame
    q1: np.ndarray                  # quaternion vector i-axis component representation of body wrt local navigation frame
    q2: np.ndarray                  # quaternion vector j-axis component representation of body wrt local navigation frame
    q3: np.ndarray                  # quaternion vector k-axis component representation of body wrt local navigation frame
    lat_rad: np.ndarray             # geodetic latitude of aircraft CM resolved in WGS84 ellipsoid framework [rad]
    long_rad: np.ndarray            # longitude of aircraft CM resolved in WGS84 ellipsoid framework [rad]
    h_m: np.ndarray                 # height of aircraft CM above the WGS84 reference ellipsoid [m]
    m_fuel_kg: np.ndarray           # current mass of usable fuel [kg]
    
    phi_rad: np.ndarray             # Euler roll angle mapping local navigation frame to body fixed CS [rad]
    theta_rad: np.ndarray           # Euler pitch angle mapping local navigation frame to body fixed CS [rad]
    psi_rad: np.ndarray             # Euler yaw (heading) angle mapping local navigation frame to body fixed CS [rad]
    phi_dot_rps: np.ndarray         # time derivative of Euler roll angle (kinematic roll rate) [rad/s]
    theta_dot_rps: np.ndarray       # time derivative of Euler pitch angle (kinematic pitch rate) [rad/s]
    psi_dot_rps: np.ndarray         # time derivative of Euler yaw angle (kinematic heading rate) [rad/s]
    p_nb_rps: np.ndarray            # roll angular velocity of body fixed CS relative to local navigation frame, resolved in body CS [rad/s]
    q_nb_rps: np.ndarray            # pitch angular velocity of body fixed CS relative to local navigation frame, resolved in body CS [rad/s]
    r_nb_rps: np.ndarray            # yaw angular velocity of body fixed CS relative to local navigation frame, resolved in body CS [rad/s]
    
    cs_mps: np.ndarray              # local speed of sound in the atmosphere [m/s]
    rho_kgpm3: np.ndarray           # local atmospheric air density [kg/m^3]
    p_kgpms2: np.ndarray            # local atmospheric ambient static pressure [N/m^2 or kg/(m*s^2)]
    T_K: np.ndarray                 # local atmospheric ambient absolute temperature [K]
    mach: np.ndarray                # Mach number (ratio of true airspeed to local speed of sound) [dimensionless]
    alpha_rad: np.ndarray           # aerodynamic angle of attack [rad]
    beta_rad: np.ndarray            # aerodynamic angle of sideslip [rad]
    true_airspeed_mps: np.ndarray   # magnitude of vehicle velocity vector relative to the local air mass [m/s]
    g_mag_mps2: np.ndarray          # local gravitational acceleration magnitude derived from position-dependent gravity model [m/s^2]
    
    u_n_mps: np.ndarray             # North component of ground velocity vector resolved in local navigation (NED) frame [m/s]
    v_n_mps: np.ndarray             # East component of ground velocity vector resolved in local navigation (NED) frame [m/s]
    w_n_mps: np.ndarray             # Down component of ground velocity vector resolved in local navigation (NED) frame [m/s]
    
    Fx_b_kgmps2: np.ndarray         # net external non-gravitational force along body fixed x-axis (aerodynamic + propulsive) [N]
    Fy_b_kgmps2: np.ndarray         # net external non-gravitational force along body fixed y-axis (aerodynamic + propulsive) [N]
    Fz_b_kgmps2: np.ndarray         # net external non-gravitational force along body fixed z-axis (aerodynamic + propulsive) [N]
    l_b_kgm2ps2: np.ndarray         # net external aerodynamic and propulsive rolling moment about body fixed x-axis [N*m]
    m_b_kgm2ps2: np.ndarray         # net external aerodynamic and propulsive pitching moment about body fixed y-axis [N*m]
    n_b_kgm2ps2: np.ndarray         # net external aerodynamic and propulsive yawing moment about body fixed z-axis [N*m]
    
    dela_ach_deg: np.ndarray        # actual achieved aileron surface deflection angle [deg]
    dele_ach_deg: np.ndarray        # actual achieved elevator surface deflection angle [deg]
    delr_ach_deg: np.ndarray        # actual achieved rudder surface deflection angle [deg]
    dela_cmd_deg: np.ndarray        # commanded aileron surface deflection angle from flight control system [deg]
    dele_cmd_deg: np.ndarray        # commanded elevator surface deflection angle from flight control system [deg]
    delr_cmd_deg: np.ndarray        # commanded rudder surface deflection angle from flight control system [deg]
    delt_percent: np.ndarray        # engine throttle lever position command [0.0 to 100.0%]
    
    # Optional Wind Parameters (Default to None to avoid breaking previous initializations)
    W_N_mps: np.ndarray = None      # North wind component resolved in local navigation (NED) frame [m/s]
    W_E_mps: np.ndarray = None      # East wind component resolved in local navigation (NED) frame [m/s]
    W_D_mps: np.ndarray = None      # Down wind component resolved in local navigation (NED) frame [m/s]
    
    def __post_init__(self):
        """Attempts to calculate missing telemetry data using established kinematic relationships."""
        
        # 1. Euler Angles (Derived from Quaternions)
        has_quat = not (np.isnan(self.q0).all() and np.isnan(self.q1).all() and np.isnan(self.q2).all() and np.isnan(self.q3).all())
        if has_quat:
            # Fallback to identity quaternion for partial data
            q0_s = np.nan_to_num(self.q0, nan=1.0)
            q1_s = np.nan_to_num(self.q1, nan=0.0)
            q2_s = np.nan_to_num(self.q2, nan=0.0)
            q3_s = np.nan_to_num(self.q3, nan=0.0)
            
            phi_calc = np.arctan2(2 * (q0_s * q1_s + q2_s * q3_s), 1 - 2 * (q1_s**2 + q2_s**2))
            theta_calc = np.arcsin(np.clip(2 * (q0_s * q2_s - q3_s * q1_s), -1.0, 1.0))
            psi_calc = np.arctan2(2 * (q0_s * q3_s + q1_s * q2_s), 1 - 2 * (q2_s**2 + q3_s**2))
            
            self.phi_rad = np.where(np.isnan(self.phi_rad), phi_calc, self.phi_rad)
            self.theta_rad = np.where(np.isnan(self.theta_rad), theta_calc, self.theta_rad)
            self.psi_rad = np.where(np.isnan(self.psi_rad), psi_calc, self.psi_rad)

        # 2. Frame Transformations (Unconditional generation of DCM for wind mapping)
        phi_safe = np.nan_to_num(self.phi_rad, nan=0.0)
        theta_safe = np.nan_to_num(self.theta_rad, nan=0.0)
        psi_safe = np.nan_to_num(self.psi_rad, nan=0.0)

        c_phi, s_phi = np.cos(phi_safe), np.sin(phi_safe)
        c_theta, s_theta = np.cos(theta_safe), np.sin(theta_safe)
        c_psi, s_psi = np.cos(psi_safe), np.sin(psi_safe)

        # Direction Cosine Matrix (Body -> Nav) components
        R11 = c_theta * c_psi
        R12 = s_phi * s_theta * c_psi - c_phi * s_psi
        R13 = c_phi * s_theta * c_psi + s_phi * s_psi
        R21 = c_theta * s_psi
        R22 = s_phi * s_theta * s_psi + c_phi * c_psi
        R23 = c_phi * s_theta * s_psi - s_phi * c_psi
        R31 = -s_theta
        R32 = s_phi * c_theta
        R33 = c_phi * c_theta
        
        has_nav_vel = not (np.isnan(self.u_n_mps).all() and np.isnan(self.v_n_mps).all() and np.isnan(self.w_n_mps).all())
        has_body_vel = not (np.isnan(self.u_b_mps).all() and np.isnan(self.v_b_mps).all() and np.isnan(self.w_b_mps).all())
        
        # Calculate Body Velocities (Nav -> Body via DCM Transpose)
        if has_nav_vel:
            un = np.nan_to_num(self.u_n_mps, nan=0.0)
            vn = np.nan_to_num(self.v_n_mps, nan=0.0)
            wn = np.nan_to_num(self.w_n_mps, nan=0.0)
            
            u_b_calc = R11 * un + R21 * vn + R31 * wn
            v_b_calc = R12 * un + R22 * vn + R32 * wn
            w_b_calc = R13 * un + R23 * vn + R33 * wn
            
            self.u_b_mps = np.where(np.isnan(self.u_b_mps), u_b_calc, self.u_b_mps)
            self.v_b_mps = np.where(np.isnan(self.v_b_mps), v_b_calc, self.v_b_mps)
            self.w_b_mps = np.where(np.isnan(self.w_b_mps), w_b_calc, self.w_b_mps)
        
        # Calculate Nav Velocities (Body -> Nav via DCM)
        if has_body_vel:
            ub = np.nan_to_num(self.u_b_mps, nan=0.0)
            vb = np.nan_to_num(self.v_b_mps, nan=0.0)
            wb = np.nan_to_num(self.w_b_mps, nan=0.0)
            
            u_n_calc = R11 * ub + R12 * vb + R13 * wb
            v_n_calc = R21 * ub + R22 * vb + R23 * wb
            w_n_calc = R31 * ub + R32 * vb + R33 * wb
            
            self.u_n_mps = np.where(np.isnan(self.u_n_mps), u_n_calc, self.u_n_mps)
            self.v_n_mps = np.where(np.isnan(self.v_n_mps), v_n_calc, self.v_n_mps)
            self.w_n_mps = np.where(np.isnan(self.w_n_mps), w_n_calc, self.w_n_mps)
        
        # 3. Aerodynamic Data Reconstruction
        ub_safe = np.nan_to_num(self.u_b_mps, nan=0.0)
        vb_safe = np.nan_to_num(self.v_b_mps, nan=0.0)
        wb_safe = np.nan_to_num(self.w_b_mps, nan=0.0)

        # Safeguard wind inputs
        wn_safe = np.nan_to_num(self.W_N_mps, nan=0.0) if self.W_N_mps is not None else np.zeros_like(ub_safe)
        we_safe = np.nan_to_num(self.W_E_mps, nan=0.0) if self.W_E_mps is not None else np.zeros_like(ub_safe)
        wd_safe = np.nan_to_num(self.W_D_mps, nan=0.0) if self.W_D_mps is not None else np.zeros_like(ub_safe)

        # Transform Wind from NED to Body Frame via C_n2b
        W_u = R11 * wn_safe + R21 * we_safe + R31 * wd_safe
        W_v = R12 * wn_safe + R22 * we_safe + R32 * wd_safe
        W_w = R13 * wn_safe + R23 * we_safe + R33 * wd_safe

        # Isolate Air-Relative Velocities
        u_air = ub_safe - W_u
        v_air = vb_safe - W_v
        w_air = wb_safe - W_w
        
        vt_calc = np.sqrt(u_air**2 + v_air**2 + w_air**2)
        
        if has_body_vel or has_nav_vel:
            self.true_airspeed_mps = np.where(np.isnan(self.true_airspeed_mps), vt_calc, self.true_airspeed_mps)
            
            vt_safe = np.where(vt_calc == 0, 1e-9, vt_calc)
            
            alpha_calc = np.arctan2(w_air, u_air)
            self.alpha_rad = np.where(np.isnan(self.alpha_rad), alpha_calc, self.alpha_rad)
            
            beta_calc = np.arcsin(np.clip(v_air / vt_safe, -1.0, 1.0))
            self.beta_rad = np.where(np.isnan(self.beta_rad), beta_calc, self.beta_rad)
            
            if not np.isnan(self.cs_mps).all():
                cs_safe = np.where(np.isnan(self.cs_mps) | (self.cs_mps == 0), 1e-9, self.cs_mps)
                mach_calc = self.true_airspeed_mps / cs_safe
                self.mach = np.where(np.isnan(self.mach), mach_calc, self.mach)

        # if has_nav_vel or has_body_vel:
        #     # Zero-fill missing Euler components to prevent NaN propagation in the DCM
        #     phi_safe = np.nan_to_num(self.phi_rad, nan=0.0)
        #     theta_safe = np.nan_to_num(self.theta_rad, nan=0.0)
        #     psi_safe = np.nan_to_num(self.psi_rad, nan=0.0)

        #     c_phi, s_phi = np.cos(phi_safe), np.sin(phi_safe)
        #     c_theta, s_theta = np.cos(theta_safe), np.sin(theta_safe)
        #     c_psi, s_psi = np.cos(psi_safe), np.sin(psi_safe)

        #     # Direction Cosine Matrix (Body -> Nav) components
        #     R11 = c_theta * c_psi
        #     R12 = s_phi * s_theta * c_psi - c_phi * s_psi
        #     R13 = c_phi * s_theta * c_psi + s_phi * s_psi
        #     R21 = c_theta * s_psi
        #     R22 = s_phi * s_theta * s_psi + c_phi * c_psi
        #     R23 = c_phi * s_theta * s_psi - s_phi * c_psi
        #     R31 = -s_theta
        #     R32 = s_phi * c_theta
        #     R33 = c_phi * c_theta

        #     # Calculate Body Velocities (Nav -> Body via DCM Transpose)
        #     if has_nav_vel:
        #         un = np.nan_to_num(self.u_n_mps, nan=0.0)
        #         vn = np.nan_to_num(self.v_n_mps, nan=0.0)
        #         wn = np.nan_to_num(self.w_n_mps, nan=0.0)
                
        #         u_b_calc = R11 * un + R21 * vn + R31 * wn
        #         v_b_calc = R12 * un + R22 * vn + R32 * wn
        #         w_b_calc = R13 * un + R23 * vn + R33 * wn
                
        #         self.u_b_mps = np.where(np.isnan(self.u_b_mps), u_b_calc, self.u_b_mps)
        #         self.v_b_mps = np.where(np.isnan(self.v_b_mps), v_b_calc, self.v_b_mps)
        #         self.w_b_mps = np.where(np.isnan(self.w_b_mps), w_b_calc, self.w_b_mps)

        #     # Calculate Nav Velocities (Body -> Nav via DCM)
        #     if has_body_vel:
        #         ub = np.nan_to_num(self.u_b_mps, nan=0.0)
        #         vb = np.nan_to_num(self.v_b_mps, nan=0.0)
        #         wb = np.nan_to_num(self.w_b_mps, nan=0.0)
                
        #         u_n_calc = R11 * ub + R12 * vb + R13 * wb
        #         v_n_calc = R21 * ub + R22 * vb + R23 * wb
        #         w_n_calc = R31 * ub + R32 * vb + R33 * wb
                
        #         self.u_n_mps = np.where(np.isnan(self.u_n_mps), u_n_calc, self.u_n_mps)
        #         self.v_n_mps = np.where(np.isnan(self.v_n_mps), v_n_calc, self.v_n_mps)
        #         self.w_n_mps = np.where(np.isnan(self.w_n_mps), w_n_calc, self.w_n_mps)

        # # 3. Aerodynamic Data Reconstruction
        # # Ensure safe arrays for aerodynamic angular math so `arctan2` does not process NaNs
        # ub_safe = np.nan_to_num(self.u_b_mps, nan=0.0)
        # vb_safe = np.nan_to_num(self.v_b_mps, nan=0.0)
        # wb_safe = np.nan_to_num(self.w_b_mps, nan=0.0)
        
        # vt_calc = np.sqrt(ub_safe**2 + vb_safe**2 + wb_safe**2)
        
        # if has_body_vel or has_nav_vel:
        #     self.true_airspeed_mps = np.where(np.isnan(self.true_airspeed_mps), vt_calc, self.true_airspeed_mps)
            
        #     vt_safe = np.where(vt_calc == 0, 1e-9, vt_calc)
            
        #     # Corrected: np.arctan2(y, x)
        #     alpha_calc = np.arctan2(wb_safe, ub_safe)
        #     self.alpha_rad = np.where(np.isnan(self.alpha_rad), alpha_calc, self.alpha_rad)
            
        #     beta_calc = np.arcsin(np.clip(vb_safe / vt_safe, -1.0, 1.0))
        #     self.beta_rad = np.where(np.isnan(self.beta_rad), beta_calc, self.beta_rad)
            
        #     if not np.isnan(self.cs_mps).all():
        #         cs_safe = np.where(np.isnan(self.cs_mps) | (self.cs_mps == 0), 1e-9, self.cs_mps)
        #         mach_calc = self.true_airspeed_mps / cs_safe
        #         self.mach = np.where(np.isnan(self.mach), mach_calc, self.mach)
    
    @property
    def lat_deg(self) -> np.ndarray:
        return self.lat_rad * R2D
    
    @property
    def long_deg(self) -> np.ndarray:
        return self.long_rad * R2D
    
    @property
    def qbar_kgpms2(self) -> np.ndarray:
        '''Returns the dynamic pressure.'''
        return 0.5 * self.rho_kgpm3 * self.true_airspeed_mps**2