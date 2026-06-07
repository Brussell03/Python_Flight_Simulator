import math
import numpy as np
from models.vehicle_base import Vehicle
from src.utils.constants import IN2M, R2D

class BallisticSphere(Vehicle):
    """
    Base class for all spherical objects, implementing common geometric,
    mass, and aerodynamic properties.
    """
    def __init__(self, name, short_name, r_sphere_in, rho_kgpm3):
        self._vehicle_name = name
        self._short_name = short_name
        
        # Unit Conversions
        
        self.r_sphere_m = r_sphere_in * IN2M
        
        # Reference Geometry
        self.b_m = self.r_sphere_m
        self.c_m = self.r_sphere_m
        self.A_ref_m2 = math.pi * (self.r_sphere_m ** 2)
        
        # Physical Calculations
        self.vol_sphere_m3 = (4.0 / 3.0) * math.pi * (self.r_sphere_m ** 3)
        self._m_sphere_kg = rho_kgpm3 * self.vol_sphere_m3
        self.J_sphere_kgm2 = 0.4 * self._m_sphere_kg * (self.r_sphere_m ** 2)
        
        # Mass Bounds (Constant for rigid spheres)
        self._m_dry_kg = self._m_sphere_kg
        self._m_wet_kg = self._m_sphere_kg
        
        # Approximate baseline drag coefficient
        self.CD_approx = 0.5
        self.Vterm_mps = math.sqrt((2.0 * self._m_sphere_kg * 9.81) / (1.2 * self.CD_approx * self.A_ref_m2))
    
    @property
    def vehicle_name(self) -> str:
        return self._vehicle_name
    
    @property
    def short_name(self) -> str:
        return self._short_name
    
    @property
    def m_dry_kg(self) -> float:
        return self._m_sphere_kg
    
    @property
    def m_wet_kg(self) -> float:
        return self._m_sphere_kg

    def get_mass_properties(self, m_total_kg):
        """
        Returns the sphere's constant inertia tensor components.
        Matches the layout: Jxx, Jyy, Jzz, Jxz
        """
        return self.J_sphere_kgm2, self.J_sphere_kgm2, self.J_sphere_kgm2, 0.0

    def get_aero_coeffs(self, alpha_deg, Mach, **kwargs):
        """
        Calculates Mach-dependent drag coefficient for a sphere.
        Returns a 21-element tuple to preserve interface symmetry with X15.
        """
        if Mach <= 0.722:
            cd = 0.45 * (Mach ** 2) + 0.424
        else:
            cd = 2.1 * math.exp(-1.16 * (Mach + 0.35)) - 8.9 * math.exp(-2.2 * (Mach + 0.35)) + 0.92
            
        # Unused aero terms are explicitly padded with 0.0
        return (0.0, 0.0, cd, 0.0, 0.0, 0.0,  # Lift, Drag, Pitching
                0.0, 0.0, 0.0, 0.0, 0.0,       # Sideforce
                0.0, 0.0, 0.0, 0.0, 0.0,       # Rolling Moment
                0.0, 0.0, 0.0, 0.0, 0.0)       # Yawing Moment

    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_deg, dela_ach_deg, 
                               delr_ach_deg, delsb_deg, throttle_perc, C_w2b, speedbrake):
        """
        Computes dimensional forces and resolves them into the body frame using the DCM matrix.
        """
        coeffs = self.get_aero_coeffs(alpha_rad * R2D, Mach)
        cd = coeffs[2]
        
        # Dimensional Wind-Axis Forces (Spheres lack lift/sideforce)
        drag_kgmps2 = cd * qbar_kgpms2 * self.A_ref_m2
        
        # Transform Wind Forces to Body Frame via C_w2b Direction Cosine Matrix
        Fx_b_kgmps2 = -C_w2b[0, 0] * drag_kgmps2
        Fy_b_kgmps2 = -C_w2b[1, 0] * drag_kgmps2
        Fz_b_kgmps2 = -C_w2b[2, 0] * drag_kgmps2
        
        # Spheres exhibit no aerodynamic or control moments in this model
        return Fx_b_kgmps2, Fy_b_kgmps2, Fz_b_kgmps2, 0.0, 0.0, 0.0

    # --- 6DOF Control Loop Compatibility Interface (Stubs) ---
    def get_engine_burn_rate(self, throttle_perc): return 0.0
    def get_sas_commands(self, t, x, cmod, u_trim): return 0.0, 0.0, 0.0
    
    def aileron_kinematics(self, dela_cmd_deg, dela_ach_deg): return 0.0
    def elevator_kinematics(self, dele_cmd_deg, dele_ach_deg): return 0.0
    def rudder_kinematics(self, delr_cmd_deg, delr_ach_deg): return 0.0


# =============================================================================
# Specific Vehicle Implementations
# =============================================================================

class Musketball50cal(BallisticSphere):
    def __init__(self):
        super().__init__(name="50 Cal Lead Ball", short_name="50cal", r_sphere_in=0.495, rho_kgpm3=11300.0)

class Carronade12lb(BallisticSphere):
    def __init__(self):
        super().__init__(name="Carronade 12 lb (5.4 kg) Cannonball", short_name="12lb", r_sphere_in=4.40, rho_kgpm3=7000.0)

class Blueberry(BallisticSphere):
    def __init__(self):
        super().__init__(name="A Blueberry", short_name="Blueberry", r_sphere_in=0.3, rho_kgpm3=786.0)

class BowlingBall(BallisticSphere):
    def __init__(self):
        super().__init__(name="Bowling Ball", short_name="Bowling", r_sphere_in=4.40, rho_kgpm3=1500.0)

class TsarCannonball(BallisticSphere):
    def __init__(self):
        super().__init__(name="Tsar Cannonball", short_name="Tsar", r_sphere_in=35.0, rho_kgpm3=7000.0)

class NASAAtmos01Sphere(BallisticSphere):
    def __init__(self):
        super().__init__(name="NASA Atmos01 1-Slug Cannonball", short_name="Atmos01", r_sphere_in=3.0, rho_kgpm3=7868.36)