import math
import numpy as np
from numba import njit, float64, int32
from numba.experimental import jitclass

# Define specifications for the Class Attributes
spec = [
    ('a', float64),
    ('b', float64),
    ('omega_rps', float64),
    ('mu', float64),
    ('j2', float64),
    ('g0', float64),
    ('gravity_type', int32),  # 0: constant, 1: inverse_square, 2: J2
    ('e_sq', float64),
    ('e', float64),
]

@jitclass(spec)
class EarthModel:
    def __init__(self, a_m=6378137.0, b_m=6356752.314245, omega_rps=7.2921151467e-5, mu=3.986004418e14, j2=1.08262668e-3, g0=9.80665,
                 gravity_type=2):
        self.a = a_m                               # Equatorial radius
        self.b = b_m                               # Polar radius
        self.omega_rps = omega_rps                 # Rotation rate
        self.mu = mu                               # Gravitational parameter
        self.j2 = j2                               # J2 Perturbation
        self.g0 = g0                               # Gravity at MSL
        self.gravity_type=gravity_type             # Gravity implementation
        
        # Derived parameters
        self.e_sq = (self.a**2 - self.b**2) / self.a**2 if self.a != 0 else 0.0
        self.e = math.sqrt(self.e_sq)

    def ecef_to_geodetic(self, x: float, y: float, z: float):
        """Bowring method for ECEF to Geodetic. Handles spherical (e_sq=0) automatically."""
        if self.e_sq == 0.0:
            p = math.sqrt(x**2 + y**2)
            r = math.sqrt(p**2 + z**2)
            lat = math.atan2(z, p)
            lon = math.atan2(y, x)
            return lat, lon, r - self.a

        ep2 = (self.a**2 - self.b**2) / self.b**2
        p = math.sqrt(x**2 + y**2)
        th = math.atan2(self.a * z, self.b * p)
        
        lon = math.atan2(y, x)
        lat = math.atan2(z + ep2 * self.b * math.sin(th)**3, p - self.e_sq * self.a * math.cos(th)**3)
        
        N = self.a / math.sqrt(1.0 - self.e_sq * math.sin(lat)**2)
        alt = p / math.cos(lat) - N
        
        return lat, lon, alt

    def get_gravity_ecef(self, x: float, y: float, z: float) -> np.ndarray:
        """Effective gravity in ECEF (Mass attraction + Centripetal)."""
        r_sq = x**2 + y**2 + z**2
        if r_sq == 0: return np.zeros(3)
        r = math.sqrt(r_sq)
        
        # Centripetal term (applies to all rotating models)
        omega_sq = self.omega_rps**2
        gx_cent = omega_sq * x
        gy_cent = omega_sq * y
        gz_cent = 0.0

        if self.gravity_type == 0: # 'constant'
            g_mag = self.g0 
            gx_mass = -g_mag * (x / r)
            gy_mass = -g_mag * (y / r)
            gz_mass = -g_mag * (z / r)
            
        elif self.gravity_type == 1: # 'inverse_square'
            factor = (self.mu / r_sq) / r
            gx_mass = -x * factor
            gy_mass = -y * factor
            gz_mass = -z * factor
            
        else: # 'J2'
            factor = (self.mu / r_sq) / r
            j2_term = 1.5 * self.j2 * (self.a**2 / r_sq)
            z_term = 5.0 * (z**2 / r_sq)
            gx_mass = -x * factor * (1.0 + j2_term * (1.0 - z_term))
            gy_mass = -y * factor * (1.0 + j2_term * (1.0 - z_term))
            gz_mass = -z * factor * (1.0 + j2_term * (3.0 - z_term))
            
        return np.array([gx_mass + gx_cent, gy_mass + gy_cent, gz_mass + gz_cent], dtype=np.float64)

    def get_earth_rate_ecef(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.omega_rps], dtype=np.float64)