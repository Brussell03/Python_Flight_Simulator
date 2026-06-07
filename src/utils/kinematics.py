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