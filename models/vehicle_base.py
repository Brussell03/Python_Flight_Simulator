from abc import ABC, abstractmethod

class Vehicle(ABC):

    @abstractmethod
    def get_mass_properties(self, m_total_kg):
        """Returns inertia properties."""
        pass

    @abstractmethod
    def get_aero_coeffs(self, alpha, mach, **kwargs):
        """Returns a dict of aero coefficients."""
        pass