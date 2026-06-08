import math
import numpy as np
from numba import njit, float64, int32
from numba.experimental import jitclass
from src.utils.constants import FT2M

# Define specifications for the Class Attributes
spec = [
    ('wind_type', int32),  # 0: constant, 1: polynomial
    ('dir_rad', float64),
    ('offset_m', float64),
    ('slope_mps', float64),
]

@jitclass(spec)
class WindModel:
    def __init__(self, wind_type, dir_rad, offset_m, slope_mps):
        self.wind_type = wind_type
        self.dir_rad = dir_rad
        self.offset_m = offset_m
        self.slope_mps = slope_mps

    def get_velocity(self, h_m: float):
        wind_speed_mps = 0
        
        # Constant Model
        if self.wind_type == 0:
            wind_speed_mps = self.offset_m
        
        # Linear Model: wind = slope * h + offset
        elif self.wind_type == 1:
            wind_speed_mps = self.slope_mps * h_m + self.offset_m
        
        return wind_speed_mps * np.cos(self.dir_rad), wind_speed_mps * np.sin(self.dir_rad), 0
    
    def get_shear(self, h_m: float):
        shear_mag_s1 = 0.0
        
        # Constant Model
        if self.wind_type == 0:
            shear_mag_s1 = 0.0
        
        # Linear Model: wind = slope * h + offset
        elif self.wind_type == 1:
            shear_mag_s1 = self.slope_mps
        
        # Decompose into NED components
        dw_n = shear_mag_s1 * np.cos(self.dir_rad)
        dw_e = shear_mag_s1 * np.sin(self.dir_rad)
        dw_d = 0.0
        
        return dw_n, dw_e, dw_d

@njit
def compute_wind_vectorized(model, h_array):
    # Pre-allocate arrays for N, E, D
    n_samples = len(h_array)
    w_n = np.zeros(n_samples)
    w_e = np.zeros(n_samples)
    w_d = np.zeros(n_samples)
    
    # Vectorized loop
    for i in range(n_samples):
        h = h_array[i]
        mag = 0.0
        
        # Calculate Magnitude
        if model.wind_type == 0:
            mag = model.offset_m
        elif model.wind_type == 1:
            mag = model.slope_mps * h + model.offset_m
            
        # Decompose into components based on dir_rad
        # Assuming 0 rad is North, East is 90 deg (pi/2)
        w_n[i] = mag * np.cos(model.dir_rad)
        w_e[i] = mag * np.sin(model.dir_rad)
        w_d[i] = 0.0  # Or add vertical wind component
        
    return w_n, w_e, w_d