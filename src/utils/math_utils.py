import math
import numpy as np
from numba import njit

@njit
def quat_to_dcm(q0, q1, q2, q3):
    """Converts unit quaternion to Direction Cosine Matrix (Body to Nav)."""
    C11 = q0**2 + q1**2 - q2**2 - q3**2
    C12 = 2 * (q1*q2 - q0*q3)
    C13 = 2 * (q1*q3 + q0*q2)
    C21 = 2 * (q1*q2 + q0*q3)
    C22 = q0**2 - q1**2 + q2**2 - q3**2
    C23 = 2 * (q2*q3 - q0*q1)
    C31 = 2 * (q1*q3 - q0*q2)
    C32 = 2 * (q2*q3 + q0*q1)
    C33 = q0**2 - q1**2 - q2**2 + q3**2
    return np.array([
        [C11, C12, C13],
        [C21, C22, C23],
        [C31, C32, C33]
    ], dtype=np.float64)

def quat_to_dcm_vectorized(q0, q1, q2, q3):
    """Converts unit quaternions (nt,) to Direction Cosine Matrices (3, 3, nt)."""
    # These operations automatically broadcast across the (nt,) array
    C11 = q0**2 + q1**2 - q2**2 - q3**2
    C12 = 2 * (q1*q2 - q0*q3)
    C13 = 2 * (q1*q3 + q0*q2)
    C21 = 2 * (q1*q2 + q0*q3)
    C22 = q0**2 - q1**2 + q2**2 - q3**2
    C23 = 2 * (q2*q3 - q0*q1)
    C31 = 2 * (q1*q3 - q0*q2)
    C32 = 2 * (q2*q3 + q0*q1)
    C33 = q0**2 - q1**2 - q2**2 + q3**2
    
    # 1. Stack into (3, 3, nt)
    # 2. Transpose axes to (nt, 3, 3) 
    #    (new_axis_0 = old_axis_2, new_axis_1 = old_axis_0, new_axis_2 = old_axis_1)
    return np.stack([
        np.stack([C11, C12, C13]),
        np.stack([C21, C22, C23]),
        np.stack([C31, C32, C33])
    ]).transpose(2, 0, 1)

@njit
def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """
    Robustly converts a DCM to a Quaternion [q0, q1, q2, q3] directly 
    matching the convention of quat_body_to_nav with strict sign consistency.
    """
    trace = np.trace(C)
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([
            0.25 / s,
            (C[2, 1] - C[1, 2]) * s,
            (C[0, 2] - C[2, 0]) * s,
            (C[1, 0] - C[0, 1]) * s
        ], dtype=np.float64)
    
    if C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = 2.0 * math.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        return np.array([
            (C[2, 1] - C[1, 2]) / s,
            0.25 * s,
            (C[0, 1] + C[1, 0]) / s,
            (C[0, 2] + C[2, 0]) / s
        ], dtype=np.float64)
    
    if C[1, 1] > C[2, 2]:
        s = 2.0 * math.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        return np.array([
            (C[0, 2] - C[2, 0]) / s,
            (C[0, 1] + C[1, 0]) / s,
            0.25 * s,
            (C[1, 2] + C[2, 1]) / s
        ], dtype=np.float64)
    
    s = 2.0 * math.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
    return np.array([
        (C[1, 0] - C[0, 1]) / s,
        (C[0, 2] + C[2, 0]) / s,
        (C[1, 2] + C[2, 1]) / s,
        0.25 * s
    ], dtype=np.float64)

@njit
def wind_to_body_dcm(alpha_rad, beta_rad):
    """Computes Wind to Body DCM."""
    sa, ca = math.sin(alpha_rad), math.cos(alpha_rad)
    sb, cb = math.sin(beta_rad), math.cos(beta_rad)
    return np.array([
        [ca*cb, -ca*sb, -sa],
        [sb,     cb,     0 ],
        [sa*cb, -sa*sb,  ca]
    ], dtype=np.float64)

@njit
def ecef_to_ned_dcm(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Generates the DCM from ECEF to local NED frame."""
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    
    return np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [-sin_lon,            cos_lon,            0.0    ],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ], dtype=np.float64)

@njit
def quaternion_derivative(q: np.ndarray, omega_rps: np.ndarray, k_quat: float = 1.0) -> np.ndarray:
    """
    Computes quaternion derivative with Baumgarte stabilization.
    q: [q0, q1, q2, q3]
    omega_rps: [p, q, r]
    """
    q0, q1, q2, q3 = q
    p, q_rate, r = omega_rps
    
    q_norm_sq = q0**2 + q1**2 + q2**2 + q3**2
    q_err = 1.0 - q_norm_sq
    
    dq0 = -0.5 * (p*q1 + q_rate*q2 + r*q3) + k_quat * q_err * q0
    dq1 =  0.5 * (p*q0 - q_rate*q3 + r*q2) + k_quat * q_err * q1
    dq2 =  0.5 * (q_rate*q0 + p*q3 - r*q1) + k_quat * q_err * q2
    dq3 =  0.5 * (r*q0 - p*q2 + q_rate*q1) + k_quat * q_err * q3
    
    return np.array([dq0, dq1, dq2, dq3], dtype=np.float64)

@njit
def euler_rates_vectorized(phi_rad: np.ndarray, theta_rad: np.ndarray, p_nb_rps: np.ndarray, q_nb_rps: np.ndarray, r_nb_rps: np.ndarray) -> np.ndarray:
    phi_dot_rps   = p_nb_rps + (q_nb_rps * np.sin(phi_rad) + r_nb_rps * np.cos(phi_rad)) * np.tan(theta_rad)
    theta_dot_rps = q_nb_rps * np.cos(phi_rad) - r_nb_rps * np.sin(phi_rad)
    psi_dot_rps   = (q_nb_rps * np.sin(phi_rad) + r_nb_rps * np.cos(phi_rad)) / np.cos(theta_rad)
    
    return np.stack((phi_dot_rps, theta_dot_rps, psi_dot_rps))

@njit
def body_to_ned_dcm(phi_rad: float, theta_rad: float, psi_rad: float) -> np.ndarray:
    sph, cph = math.sin(phi_rad), math.cos(phi_rad)
    sth, cth = math.sin(theta_rad), math.cos(theta_rad)
    sps, cps = math.sin(psi_rad), math.cos(psi_rad)

    C_b2n = np.array([
        [cps*cth, cps*sth*sph - sps*cph, cps*sth*cph + sps*sph],
        [sps*cth, sps*sth*sph + cps*cph, sps*sth*cph - cps*sph],
        [-sth,    cth*sph,               cth*cph]
    ], dtype=np.float64)
    
    return C_b2n

@njit
def b2n_dcm_to_euler(C_b2n: np.ndarray) -> np.ndarray:
    phi_rad   = np.arctan2(C_b2n[2, 1], C_b2n[2, 2])
    theta_rad = math.asin(min(max(-C_b2n[2, 0], -1.0), 1.0))
    psi_rad   = np.arctan2(C_b2n[1, 0], C_b2n[0, 0])
    
    return np.array([phi_rad, theta_rad, psi_rad], dtype=np.float64)

@njit
def flight_path_angle(w_n_mps: float, true_airspeed_mps: float):
    if true_airspeed_mps == 0.0:
        return 0.0
    
    ratio = min(max(w_n_mps / true_airspeed_mps, -1.0), 1.0)
    return -math.asin(ratio)

@njit
def body_velocities(true_airspeed_mps: float, alpha_rad: float, beta_rad: float) -> np.ndarray:
    u_b_mps = true_airspeed_mps * math.cos(alpha_rad) * math.cos(beta_rad)
    v_b_mps = true_airspeed_mps * math.sin(beta_rad)
    w_b_mps = true_airspeed_mps * math.sin(alpha_rad) * math.cos(beta_rad)
    
    return np.array([u_b_mps, v_b_mps, w_b_mps], dtype=np.float64)

@njit
def true_airspeed(u_b_mps: float, v_b_mps: float, w_b_mps: float):
    return math.sqrt(u_b_mps**2 + v_b_mps**2 + w_b_mps**2)

@njit
def euler_to_quat(phi_rad: float, theta_rad: float, psi_rad: float) -> np.ndarray:
    q0 = math.cos(psi_rad/2)*math.cos(theta_rad/2)*math.cos(phi_rad/2) + math.sin(psi_rad/2)*math.sin(theta_rad/2)*math.sin(phi_rad/2)
    q1 = math.cos(psi_rad/2)*math.cos(theta_rad/2)*math.sin(phi_rad/2) - math.sin(psi_rad/2)*math.sin(theta_rad/2)*math.cos(phi_rad/2)
    q2 = math.cos(psi_rad/2)*math.sin(theta_rad/2)*math.cos(phi_rad/2) + math.sin(psi_rad/2)*math.cos(theta_rad/2)*math.sin(phi_rad/2)
    q3 = math.sin(psi_rad/2)*math.cos(theta_rad/2)*math.cos(phi_rad/2) - math.cos(psi_rad/2)*math.sin(theta_rad/2)*math.sin(phi_rad/2)
    return np.array([q0, q1, q2, q3], dtype=np.float64)

@njit
def dynamic_pressure(rho_kgpm3: float, true_airspeed_mps: float):
    return 0.5 * rho_kgpm3 * true_airspeed_mps**2

@njit
def angle_of_attack(u_air_b_mps: float, w_air_b_mps: float):
    return math.atan2(w_air_b_mps, u_air_b_mps)

@njit
def angle_of_sideslip(v_air_b_mps: float, true_airspeed_mps: float):
    return math.asin(v_air_b_mps / true_airspeed_mps) if true_airspeed_mps > 0 else 0.0