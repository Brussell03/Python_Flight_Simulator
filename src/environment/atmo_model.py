import numpy as np
from numba import float64
from numba.experimental import jitclass
from src.utils.interpolators import fastInterp1

# 1. Define specifications for 1D float64 arrays
spec = [
    ('alt_m', float64[:]),
    ('rho_kgpm3', float64[:]),
    ('c_mps', float64[:]),
    ('p_Npm2', float64[:]),
    ('T_K', float64[:])
]

@jitclass(spec)
class AtmoModel:
    def __init__(self, alt_m, rho_kgpm3, c_mps, p_Npm2, T_K):
        # Force contiguous memory layout to prevent Numba type failures
        self.alt_m = alt_m
        self.rho_kgpm3 = rho_kgpm3
        self.c_mps = c_mps
        self.p_Npm2 = p_Npm2
        self.T_K = T_K

    def get_properties(self, h_m: float):
        """
        Encapsulates the interpolation logic.
        Returns: (rho_kgpm3, c_mps, p_Npm2, T_K)
        """
        rho = fastInterp1(self.alt_m, self.rho_kgpm3, h_m)
        c = fastInterp1(self.alt_m, self.c_mps, h_m)
        p = fastInterp1(self.alt_m, self.p_Npm2, h_m)
        T = fastInterp1(self.alt_m, self.T_K, h_m)
        
        return rho, c, p, T
    
    def get_density_and_soundspeed(self, h_m: float):
        rho = fastInterp1(self.alt_m, self.rho_kgpm3, h_m)
        c = fastInterp1(self.alt_m, self.c_mps, h_m)
        return rho, c
    
    def get_density(self, h_m: float):
        rho = fastInterp1(self.alt_m, self.rho_kgpm3, h_m)
        return rho
    
    def get_soundspeed(self, h_m: float):
        c = fastInterp1(self.alt_m, self.c_mps, h_m)
        return c
    
    def get_atmo_properties_vectorized(self, h_m_array):
        """
        Vectorized wrapper for atmo_model properties.
        """
        n = len(h_m_array)
        # Pre-allocate output arrays
        rho_arr = np.zeros(n)
        c_arr = np.zeros(n)
        p_arr = np.zeros(n)
        T_arr = np.zeros(n)
        
        # Iterate using the compiled jitclass method
        for i in range(n):
            rho_arr[i], c_arr[i], p_arr[i], T_arr[i] = self.get_properties(h_m_array[i])
            
        return rho_arr, c_arr, p_arr, T_arr