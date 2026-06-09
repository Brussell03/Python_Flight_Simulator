import numpy as np

from models.dave_vehicle import DAVEVehicle
from src.utils.unit_conversion import UnitConverter

class Brick(DAVEVehicle):
    def __init__(self):
        super().__init__(name="Brick (Damping)", short_name="Brick", aero_dml_path="models/brick/brick_aero.dml", inertia_dml_path="models/brick/brick_inertia.dml")
        
    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake, h_m):
        
        # Set inputs
        self.set_var_val(self.true_airspeed_ref, true_airspeed_mps)
        self.set_var_val(self.p_b_ref, p_b_rps)
        self.set_var_val(self.q_b_ref, q_b_rps)
        self.set_var_val(self.r_b_ref, r_b_rps)
        
        Cl = self.get_si_val(self.Cl_def)
        Cm = self.get_si_val(self.Cm_def)
        Cn = self.get_si_val(self.Cn_def)
        
        # Dimensionalize Body-Frame Moments
        l_b = qbar_kgpms2 * self.S_ref_m2 * self.b_m * Cl
        m_b = qbar_kgpms2 * self.S_ref_m2 * self.c_m * Cm
        n_b = qbar_kgpms2 * self.S_ref_m2 * self.b_m * Cn
        
        return 0, 0 , 0, l_b, m_b, n_b

class DraglessBrick(DAVEVehicle):
    def __init__(self):
        super().__init__(name="Brick (No Damping)", short_name="Brick", aero_dml_path="models/brick/brick_aero.dml", inertia_dml_path="models/brick/brick_inertia.dml")
    
    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake, h_m):
        return 0, 0, 0, 0, 0, 0