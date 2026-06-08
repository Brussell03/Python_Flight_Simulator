import numpy as np
import pyJanus
from models.vehicle_base import Vehicle
from src.utils.unit_conversion import UnitConverter

class DAVEVehicle(Vehicle):
    def get_var_def(self, system, var_name):
        """Attempts to fetch a variabledef, returning None if missing."""
        try:
            var_def = system.get_variabledef(var_name)
        except:
            var_def = None
        return var_def

    def get_si_val(self, var_def):
        if var_def is not None:
            return UnitConverter.to_si(var_def.get_value(), str(var_def.units))
        return 0

    def set_var_val(self, var_def, value):
        if var_def is not None:
            var_def.set_value(UnitConverter.from_si(value, str(var_def.units)))
        
    def __init__(self, name, short_name, aero_dml_path="models/cannonball/cannonball_aero.dml", inertia_dml_path="models/cannonball/cannonball_inertia.dml"):
        super().__init__()
        self._vehicle_name = name
        self._short_name = short_name
        
        # ==========================================
        # 1. Inertia System Initialization
        # ==========================================
        self.inertia_sys = pyJanus.Janus(inertia_dml_path)
        
        self.mass_def = self.get_var_def(self.inertia_sys, "XMASS")
        
        self.Jxx_def = self.get_var_def(self.inertia_sys, "XIXX")
        self.Jyy_def = self.get_var_def(self.inertia_sys, "XIYY")
        self.Jzz_def = self.get_var_def(self.inertia_sys, "XIZZ")
        self.Jxz_def = self.get_var_def(self.inertia_sys, "XIZX")
        self.Jxy_def = self.get_var_def(self.inertia_sys, "XIXY")
        self.Jyz_def = self.get_var_def(self.inertia_sys, "XIYZ")
        
        self.x_cg_b_def = self.get_var_def(self.inertia_sys, "DXCG")
        self.y_cg_b_def = self.get_var_def(self.inertia_sys, "DYCG")
        self.z_cg_b_def = self.get_var_def(self.inertia_sys, "DZCG")
        
        # ==========================================
        # 2. Aero System Initialization
        # ==========================================
        self.aero_sys = pyJanus.Janus(aero_dml_path)
        
        # Constants
        self.S_ref_def = self.get_var_def(self.aero_sys, "SWING")
        self.S_ref_m2 = self.get_si_val(self.S_ref_def)
        
        self.b_ref = self.get_var_def(self.aero_sys, "BSPAN")
        self.b_m = self.get_si_val(self.b_ref)
        
        self.c_ref = self.get_var_def(self.aero_sys, "CBAR")
        self.c_m = self.get_si_val(self.c_ref)
        
        self.Clp_def = self.get_var_def(self.aero_sys, "CLP_DAMPING")
        self.Clp_prad = self.get_si_val(self.Clp_def)
        
        self.Clr_def = self.get_var_def(self.aero_sys, "CLR_DAMPING")
        self.Clr_prad = self.get_si_val(self.Clr_def)
        
        self.Cmq_def = self.get_var_def(self.aero_sys, "CMQ_DAMPING")
        self.Cmq_prad = self.get_si_val(self.Cmq_def)
        
        self.Cnp_def = self.get_var_def(self.aero_sys, "CNP_DAMPING")
        self.Cnp_prad = self.get_si_val(self.Cnp_def)
        
        self.Cnr_def = self.get_var_def(self.aero_sys, "CNR_DAMPING")
        self.Cnr_prad = self.get_si_val(self.Cnr_def)
        
        # Inputs
        self.true_airspeed_ref = self.get_var_def(self.aero_sys, "VRW")
        self.p_b_ref = self.get_var_def(self.aero_sys, "PB")
        self.q_b_ref = self.get_var_def(self.aero_sys, "QB")
        self.r_b_ref = self.get_var_def(self.aero_sys, "RB")
        
        # Outputs
        self.CL_def = self.get_var_def(self.aero_sys, "CL")
        self.CD_def = self.get_var_def(self.aero_sys, "CD")
        self.CY_def = self.get_var_def(self.aero_sys, "CY")
        self.Cl_def = self.get_var_def(self.aero_sys, "Cl")
        self.Cm_def = self.get_var_def(self.aero_sys, "Cm")
        self.Cn_def = self.get_var_def(self.aero_sys, "Cn")
        
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
        return self.get_si_val(self.mass_def)
    
    @property
    def m_wet_kg(self) -> float:
        return self.get_si_val(self.mass_def)

    # ==========================================
    # Dynamics and Aero Methods
    # ==========================================
    def get_mass_properties(self, m_total_kg):
        Jxx = self.get_si_val(self.Jxx_def)
        Jyy = self.get_si_val(self.Jyy_def)
        Jzz = self.get_si_val(self.Jzz_def)
        Jxz = self.get_si_val(self.Jxz_def)
        Jxy = self.get_si_val(self.Jxy_def)
        Jyz = self.get_si_val(self.Jyz_def)
        return [Jxx, Jyy, Jzz, Jxy, Jxz, Jyz]

    def get_aero_coeffs(self, alpha, mach, **kwargs):
        # Explicit bypass: aero mapping is handled via Janus inside get_forces_and_moments
        pass

    def get_engine_burn_rate(self, throttle_perc):
        return 0.0

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        
        # Set inputs
        self.set_var_val(self.true_airspeed_ref, true_airspeed_mps)
        self.set_var_val(self.p_b_ref, p_b_rps)
        self.set_var_val(self.q_b_ref, q_b_rps)
        self.set_var_val(self.r_b_ref, r_b_rps)
        
        # Get outputs
        CL = self.get_si_val(self.CL_def)
        CD = self.get_si_val(self.CD_def)
        CY = self.get_si_val(self.CY_def)
        
        Cl = self.get_si_val(self.Cl_def)
        Cm = self.get_si_val(self.Cm_def)
        Cn = self.get_si_val(self.Cn_def)
        
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