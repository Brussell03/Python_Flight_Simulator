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
    def get_aero_coeffs(self, alpha, mach, **kwargs):
        """Returns a dict of aero coefficients."""
        pass
    
    @abstractmethod
    def get_engine_burn_rate(self, throttle_perc):
        pass
    
    @abstractmethod
    def get_sas_commands(self, t, x, cmod, u_trim):
        pass
    
    @abstractmethod
    def aileron_kinematics(self, dela_cmd_deg, dela_ach_deg):
        pass
    
    @abstractmethod
    def elevator_kinematics(self, dele_cmd_deg, dele_ach_deg):
        pass
    
    @abstractmethod
    def rudder_kinematics(self, delr_cmd_deg, delr_ach_deg):
        pass