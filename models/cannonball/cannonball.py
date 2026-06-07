import numpy as np

from models.dave_vehicle import DAVEVehicle

class DraglessCannonball(DAVEVehicle):
    def __init__(self):
        super().__init__(name="Cannonball", short_name="Cannonball", aero_dml_path="models/cannonball/cannonball_aero.dml", inertia_dml_path="models/cannonball/cannonball_inertia.dml")
        self.b_m = np.sqrt((4.0 * self.S_ref_m2) / np.pi)
        self.c_m = np.sqrt((4.0 * self.S_ref_m2) / np.pi)
    
    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        return 0, 0, 0, 0, 0, 0

class Cannonball(DAVEVehicle):
    def __init__(self):
        super().__init__(name="Cannonball", short_name="Cannonball", aero_dml_path="models/cannonball/cannonball_aero.dml", inertia_dml_path="models/cannonball/cannonball_inertia.dml")
        self.b_m = np.sqrt((4.0 * self.S_ref_m2) / np.pi)
        self.c_m = np.sqrt((4.0 * self.S_ref_m2) / np.pi)