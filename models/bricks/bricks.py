import numpy as np
from models.vehicle_base import Vehicle

class TumblingBrick(Vehicle):
    """
    Base class for the tumbling brick models, implementing geometric, mass, 
    and rotational aerodynamic damping properties.
    """
    def __init__(self, name, short_name, Clp, Cmq, Cnr):
        self._vehicle_name = name
        self._short_name = short_name
        
        # Constants
        in2m = 0.0254
        ft2m = 0.304878
        slug2kg = 14.5939

        # Mass Calculations
        m_brick_slug = 0.1554048
        self._m_kg = m_brick_slug * slug2kg
        
        # Moments of Inertia
        self.Jxx_kgm2 = slug2kg * (ft2m**2) * 0.00189422
        self.Jyy_kgm2 = slug2kg * (ft2m**2) * 0.00621102
        self.Jzz_kgm2 = slug2kg * (ft2m**2) * 0.00719467
        self.Jxz_kgm2 = 0.0

        # Reference Geometry
        self.length_m = 8 * in2m
        self.width_m = 4 * in2m
        self.A_ref_m2 = self.length_m * self.width_m
        
        self.b_m = 0.33333 * ft2m
        self.c_m = 0.66667 * ft2m
        
        # Aerodynamic Damping Coefficients
        self.Clp = Clp
        self.Clr = 0.0
        self.Cmq = Cmq
        self.Cnp = 0.0
        self.Cnr = Cnr

    # --- Property Implementations ---

    @property
    def vehicle_name(self) -> str:
        return self._vehicle_name
    
    @property
    def short_name(self) -> str:
        return self._short_name
    
    @property
    def m_dry_kg(self) -> float:
        return self._m_kg
    
    @property
    def m_wet_kg(self) -> float:
        return self._m_kg

    # --- Method Implementations ---

    def get_mass_properties(self, m_total_kg):
        """Returns the brick's constant inertia tensor components."""
        return self.Jxx_kgm2, self.Jyy_kgm2, self.Jzz_kgm2, self.Jxz_kgm2

    def get_aero_coeffs(self, alpha_deg, Mach, **kwargs):
        """Bricks lack lift and drag in this model; return zero padding."""
        return (0.0,) * 21

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        """
        Computes body-axis forces and rotational damping moments.
        Translational forces (Fx, Fy, Fz) are defined as zero for the brick models.
        """
        Fx_b = Fy_b = Fz_b = 0.0
        
        # Rotational Aerodynamic Damping 
        if true_airspeed_mps > 0:
            # Dimensionless angular rates
            p_hat = (p_b_rps * self.b_m) / (2.0 * true_airspeed_mps)
            q_hat = (q_b_rps * self.c_m) / (2.0 * true_airspeed_mps)
            r_hat = (r_b_rps * self.b_m) / (2.0 * true_airspeed_mps)
            
            # Non-dimensional moments
            Cl = self.Clp * p_hat + self.Clr * r_hat
            Cm = self.Cmq * q_hat
            Cn = self.Cnp * p_hat + self.Cnr * r_hat
            
            # Dimensionalize back into body moments
            l_b_kgm2ps2 = Cl * qbar_kgpms2 * self.A_ref_m2 * self.b_m
            m_b_kgm2ps2 = Cm * qbar_kgpms2 * self.A_ref_m2 * self.c_m
            n_b_kgm2ps2 = Cn * qbar_kgpms2 * self.A_ref_m2 * self.b_m
        else:
            l_b_kgm2ps2 = m_b_kgm2ps2 = n_b_kgm2ps2 = 0.0

        return Fx_b, Fy_b, Fz_b, l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2

    # --- 6DOF Control Loop Compatibility Interface (Stubs) ---
    def get_engine_burn_rate(self, throttle_perc): return 0.0
    def get_sas_commands(self, t, x, cmod, u_trim): return 0.0, 0.0, 0.0
    def aileron_kinematics(self, dela_cmd_deg, dela_ach_deg): return 0.0
    def elevator_kinematics(self, dele_cmd_deg, dele_ach_deg): return 0.0
    def rudder_kinematics(self, delr_cmd_deg, delr_ach_deg): return 0.0


# =============================================================================
# Specific Vehicle Implementations
# =============================================================================

class NASAAtmos02Brick(TumblingBrick):
    def __init__(self):
        super().__init__(
            name="Tumbling Brick (No Damping or Drag)",
            short_name="Atmos02",
            Clp=0.0, 
            Cmq=0.0, 
            Cnr=0.0
        )

class NASAAtmos03Brick(TumblingBrick):
    def __init__(self):
        super().__init__(
            name="Tumbling Brick (with Aerodynamic Damping)",
            short_name="Atmos03",
            Clp=-1.0, 
            Cmq=-1.0, 
            Cnr=-1.0
        )