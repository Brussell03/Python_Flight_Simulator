from abc import ABC, abstractmethod

class Vehicle(ABC):
    
    @property
    @abstractmethod
    def vehicle_name(self) -> str:
        """Return the full vehicle name."""
        pass
    
    @property
    @abstractmethod
    def short_name(self) -> str:
        """Return the shortened vehicle name."""
        pass
    
    @property
    @abstractmethod
    def m_dry_kg(self) -> float:
        """Return the mass of the vehicle with no fuel in kilograms."""
        pass
    
    @property
    @abstractmethod
    def m_wet_kg(self) -> float:
        """Return the mass of the vehicle with full fuel in kilograms."""
        pass

    @abstractmethod
    def get_mass_properties(self, m_total_kg):
        """Returns inertia properties."""
        pass
    
    @abstractmethod
    def get_forces_and_moments(self, alpha_rad, beta_rad, Mach, qbar_kgpms2, true_airspeed_mps, 
                               p_b_rps, q_b_rps, r_b_rps, dele_ach_rad, dela_ach_rad, 
                               delr_ach_rad, delsb_deg, throttle_perc, C_w2b, speedbrake, h_m):
        pass
    
    @abstractmethod
    def get_engine_burn_rate(self, throttle_perc):
        pass
    
    @abstractmethod
    def set_gnc_inputs(self, t_s, cmod, amod, lat_rad, long_rad, h_m, alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad, p_b_rps, q_b_rps, r_b_rps, true_airspeed_mps, rho_kgpm3, x_trim):
        pass
    
    @abstractmethod
    def get_sas_commands(self, t, x, cmod, x_trim):
        pass
    
    @abstractmethod
    def aileron_kinematics(self, dela_cmd_rad, dela_ach_rad):
        pass
    
    @abstractmethod
    def elevator_kinematics(self, dele_cmd_rad, dele_ach_rad):
        pass
    
    @abstractmethod
    def rudder_kinematics(self, delr_cmd_rad, delr_ach_rad):
        pass
    
    @abstractmethod
    def throttle_kinematics(self, delt_cmd_pct, delt_ach_pct):
        return 0.0