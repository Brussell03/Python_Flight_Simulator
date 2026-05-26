import math
import numpy as np

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
    ])

def wind_to_body_dcm(alpha_rad, beta_rad):
    """Computes Wind to Body DCM."""
    sa, ca = math.sin(alpha_rad), math.cos(alpha_rad)
    sb, cb = math.sin(beta_rad), math.cos(beta_rad)
    return np.array([
        [ca*cb, -ca*sb, -sa],
        [sb,     cb,     0 ],
        [sa*cb, -sa*sb,  ca]
    ])