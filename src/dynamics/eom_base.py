from abc import ABC, abstractmethod

class eom(ABC):

    @abstractmethod
    def solve_eom(t, x, dx, auxillary_data, u_trim, vehicle, amod, cmod):
        """Returns the time derivative of each state in x (RHS of governing equations)."""
        pass

    @abstractmethod
    def post_process(self, x, t_s, amod, auxillary_data, **kwargs):
        """Returns an array of all simulation data."""
        pass