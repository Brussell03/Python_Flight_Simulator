import numpy as np
import pyJanus
from models.vehicle_base import Vehicle
from src.utils.unit_conversion import UnitConverter

class DAVEVehicle(Vehicle):
    def __init__(self, name, short_name, aero_dml_path="models/spheres/cannonball_aero.dml", inertia_dml_path="models/spheres/cannonball_inertia.dml"):
        super().__init__()
        self._vehicle_name = name
        self._short_name = short_name
        
        # ==========================================
        # 1. Inertia System Initialization
        # ==========================================
        self.inertia_sys = pyJanus.Janus(inertia_dml_path)
        
        self.mass_def = self.inertia_sys.get_variabledef("XMASS")
        
        self.Jxx_def = self.inertia_sys.get_variabledef("XIXX")
        self.Jyy_def = self.inertia_sys.get_variabledef("XIYY")
        self.Jzz_def = self.inertia_sys.get_variabledef("XIZZ")
        self.Jxz_def = self.inertia_sys.get_variabledef("XIZX")
        self.Jxy_def = self.inertia_sys.get_variabledef("XIXY")
        self.Jyz_def = self.inertia_sys.get_variabledef("XIYZ")
        
        self.x_cg_b_def = self.inertia_sys.get_variabledef("DXCG")
        self.y_cg_b_def = self.inertia_sys.get_variabledef("DYCG")
        self.z_cg_b_def = self.inertia_sys.get_variabledef("DZCG")
        
        # ==========================================
        # 2. Aero System Initialization
        # ==========================================
        self.aero_sys = pyJanus.Janus(aero_dml_path)
        
        self.CL_def = self.aero_sys.get_variabledef("CL")
        self.CD_def = self.aero_sys.get_variabledef("CD")
        self.CY_def = self.aero_sys.get_variabledef("CY")
        self.Cl_def = self.aero_sys.get_variabledef("Cl")
        self.Cm_def = self.aero_sys.get_variabledef("Cm")
        self.Cn_def = self.aero_sys.get_variabledef("Cn")
        
        # Static Reference Geometry
        S_ref_def = self.aero_sys.get_variabledef("SWING")
        self.S_ref_m2 = UnitConverter.to_si(S_ref_def.get_value(), str(S_ref_def.units))
        self.b_m = np.sqrt((4.0 * self.S_ref_m2) / np.pi)
        self.c_m = np.sqrt((4.0 * self.S_ref_m2) / np.pi)
        
        # print(UnitConverter.to_si(self.mass_def.get_value(), str(self.mass_def.units)))
        # print(self.S_ref_m2)
        # print(self.b_m)
        # print(UnitConverter.to_si(self.Jxx_def.get_value(), str(self.Jxx_def.units)))
        # print(UnitConverter.to_si(self.Jyy_def.get_value(), str(self.Jyy_def.units)))
        # print(UnitConverter.to_si(self.Jzz_def.get_value(), str(self.Jzz_def.units)))
        # print(UnitConverter.to_si(self.Jxz_def.get_value(), str(self.Jxz_def.units)))
        # print(UnitConverter.to_si(self.CL_def.get_value(), str(self.CL_def.units)))
        # print(UnitConverter.to_si(self.CD_def.get_value(), str(self.CD_def.units)))
        # print(UnitConverter.to_si(self.CY_def.get_value(), str(self.CY_def.units)))
        # print(UnitConverter.to_si(self.Cl_def.get_value(), str(self.Cl_def.units)))
        # print(UnitConverter.to_si(self.Cm_def.get_value(), str(self.Cm_def.units)))
        # print(UnitConverter.to_si(self.Cn_def.get_value(), str(self.Cn_def.units)))

    # ==========================================
    # Base Class Properties
    # ==========================================
    @property
    def vehicle_name(self) -> str:
        return self._vehicle_name
    
    @property
    def short_name(self) -> str:
        return self._short_name
    
    @property
    def m_dry_kg(self) -> float:
        return UnitConverter.to_si(self.mass_def.get_value(), str(self.mass_def.units))
    
    @property
    def m_wet_kg(self) -> float:
        return UnitConverter.to_si(self.mass_def.get_value(), str(self.mass_def.units))

    # ==========================================
    # Dynamics and Aero Methods
    # ==========================================
    def get_mass_properties(self, m_total_kg):
        Jxx = UnitConverter.to_si(self.Jxx_def.get_value(), str(self.Jxx_def.units))
        Jyy = UnitConverter.to_si(self.Jyy_def.get_value(), str(self.Jyy_def.units))
        Jzz = UnitConverter.to_si(self.Jzz_def.get_value(), str(self.Jzz_def.units))
        Jxz = UnitConverter.to_si(self.Jxz_def.get_value(), str(self.Jxz_def.units))
        return [Jxx, Jyy, Jzz, Jxz]

    def get_aero_coeffs(self, alpha, mach, **kwargs):
        # Explicit bypass: aero mapping is handled via Janus inside get_forces_and_moments
        pass

    def get_engine_burn_rate(self, throttle_perc):
        return 0.0

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        
        CL = UnitConverter.to_si(self.CL_def.get_value(), str(self.CL_def.units))
        CD = UnitConverter.to_si(self.CD_def.get_value(), str(self.CD_def.units))
        CY = UnitConverter.to_si(self.CY_def.get_value(), str(self.CY_def.units))
        
        Cl = UnitConverter.to_si(self.Cl_def.get_value(), str(self.Cl_def.units))
        Cm = UnitConverter.to_si(self.Cm_def.get_value(), str(self.Cm_def.units))
        Cn = UnitConverter.to_si(self.Cn_def.get_value(), str(self.Cn_def.units))
        
        # Dimensionalize Wind-Frame Forces
        D_N = qbar_kgpms2 * self.S_ref_m2 * CD
        Y_N = qbar_kgpms2 * self.S_ref_m2 * CY
        L_N = qbar_kgpms2 * self.S_ref_m2 * CL
        
        # Rotate Wind to Body-Frame Forces
        F_wind_N = np.array([-D_N, Y_N, -L_N])
        F_body_N = C_w2b @ F_wind_N
        Fx_b, Fy_b, Fz_b = F_body_N[0], F_body_N[1], F_body_N[2]
        
        # Dimensionalize Body-Frame Moments
        l_b = qbar_kgpms2 * self.S_ref_m2 * self.b_m * Cl
        m_b = qbar_kgpms2 * self.S_ref_m2 * self.c_m * Cm
        n_b = qbar_kgpms2 * self.S_ref_m2 * self.b_m * Cn
        
        return Fx_b, Fy_b, Fz_b, l_b, m_b, n_b

    # ==========================================
    # Pass-through Kinematics
    # ==========================================
    def aileron_kinematics(self, dela_cmd_deg, dela_ach_deg_old):
        return 0.0

    def elevator_kinematics(self, dele_cmd_deg, dele_ach_deg_old):
        return 0.0

    def rudder_kinematics(self, delr_cmd_deg, delr_ach_deg_old):
        return 0.0

    def get_sas_commands(self, t, x, cmod, u_trim):
        return 0.0, 0.0, 0.0

class DraglessCannonball(DAVEVehicle):
    def __init__(self):
        super().__init__(name="Cannonball", short_name="Cannonball", aero_dml_path="models/spheres/cannonball_aero.dml", inertia_dml_path="models/spheres/cannonball_inertia.dml")
    
    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        return 0, 0, 0, 0, 0, 0

class Cannonball(DAVEVehicle):
    def __init__(self):
        super().__init__(name="Cannonball", short_name="Cannonball", aero_dml_path="models/spheres/cannonball_aero.dml", inertia_dml_path="models/spheres/cannonball_inertia.dml")