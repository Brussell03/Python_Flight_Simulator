import numpy as np
from models.vehicle_base import Vehicle

class DMLCannonball(Vehicle):
    def __init__(self, aero_dml_path="models/spheres/cannonball_aero.dml", inertia_dml_path="models/spheres/cannonball_inertia.dml"):
        super().__init__()
        
        # Unit Conversion Constants
        SLUG_TO_KG = 14.5939029
        SLUGFT2_TO_KGM2 = 1.3558179
        FT_TO_M = 0.3048
        FT2_TO_M2 = 0.09290304
        
        # ==========================================
        # 1. Inertia System Initialization
        # ==========================================
        self.inertia_sys = janus.Janus()
        self.inertia_sys.xmlParse(inertia_dml_path)
        
        # Extract and convert mass/inertia to SI [kg, kg*m^2]
        self._m_dry_kg = self.inertia_sys.getVariable("XMASS").getValue() * SLUG_TO_KG
        self._m_wet_kg = self._m_dry_kg 
        
        self.Jxx = self.inertia_sys.getVariable("XIXX").getValue() * SLUGFT2_TO_KGM2
        self.Jyy = self.inertia_sys.getVariable("XIYY").getValue() * SLUGFT2_TO_KGM2
        self.Jzz = self.inertia_sys.getVariable("XIZZ").getValue() * SLUGFT2_TO_KGM2
        self.Jxz = self.inertia_sys.getVariable("XIXZ").getValue() * SLUGFT2_TO_KGM2
        
        # ==========================================
        # 2. Aero System Initialization
        # ==========================================
        self.aero_sys = janus.Janus()
        self.aero_sys.xmlParse(aero_dml_path)
        
        # Extract and convert reference geometry to SI [m, m^2]
        self.S_ref_m2 = self.aero_sys.getVariable("SWING").getValue() * FT2_TO_M2
        self.b_ref_m  = 0.1524 # 6 inches in meters (cannonball diam)
        self.cbar_ref_m = 0.1524
        
        # Cache Input Variables
        self.in_mach  = self.aero_sys.getVariable("mach")
        self.in_alpha = self.aero_sys.getVariable("alpha")
        self.in_beta  = self.aero_sys.getVariable("beta")
        
        # Cache Output Variables
        self.out_CL = self.aero_sys.getVariable("CL")
        self.out_CD = self.aero_sys.getVariable("CD")
        self.out_CY = self.aero_sys.getVariable("CY")
        self.out_Cl = self.aero_sys.getVariable("Cl")
        self.out_Cm = self.aero_sys.getVariable("Cm")
        self.out_Cn = self.aero_sys.getVariable("Cn")

    # ==========================================
    # Base Class Properties
    # ==========================================
    @property
    def vehicle_name(self) -> str:
        return "DML Cannonball"
    
    @property
    def short_name(self) -> str:
        return "DML_Cball"
    
    @property
    def m_dry_kg(self) -> float:
        return self._m_dry_kg
    
    @property
    def m_wet_kg(self) -> float:
        return self._m_wet_kg

    # ==========================================
    # Dynamics and Aero Methods
    # ==========================================
    def get_mass_properties(self, m_total_kg):
        return [self.Jxx, self.Jyy, self.Jzz, self.Jxz]

    def get_aero_coeffs(self, alpha, mach, **kwargs):
        # Explicit bypass: aero mapping is handled via Janus inside get_forces_and_moments
        pass

    def get_engine_burn_rate(self, throttle_perc):
        return 0.0

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        
        # 1. Update Janus Inputs
        self.in_alpha.setValue(alpha_rad)
        self.in_beta.setValue(beta_rad)
        self.in_mach.setValue(Mach)
        
        # 2. Execute DAVE-ML MathML/Tables
        self.aero_sys.update()
        
        # 3. Retrieve dimensionless coefficients
        CD = self.out_CD.getValue()
        CL = self.out_CL.getValue()
        CY = self.out_CY.getValue()
        
        Cl = self.out_Cl.getValue()
        Cm = self.out_Cm.getValue()
        Cn = self.out_Cn.getValue()
        
        # 4. Dimensionalize Wind-Frame Forces [Newtons]
        D_N = qbar_kgpms2 * self.S_ref_m2 * CD
        Y_N = qbar_kgpms2 * self.S_ref_m2 * CY
        L_N = qbar_kgpms2 * self.S_ref_m2 * CL
        
        F_wind_N = np.array([-D_N, Y_N, -L_N])
        
        # 5. Rotate to Body-Frame Forces [Newtons]
        F_body_N = C_w2b @ F_wind_N
        Fx_b, Fy_b, Fz_b = F_body_N[0], F_body_N[1], F_body_N[2]
        
        # 6. Dimensionalize Body-Frame Moments [Newton-meters]
        l_b = qbar_kgpms2 * self.S_ref_m2 * self.b_ref_m * Cl
        m_b = qbar_kgpms2 * self.S_ref_m2 * self.cbar_ref_m * Cm
        n_b = qbar_kgpms2 * self.S_ref_m2 * self.b_ref_m * Cn
        
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