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