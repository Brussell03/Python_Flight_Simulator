import math

class UnitConverter:
    """
    Handles bi-directional unit conversion between the SI units used by the 
    eom_unified solver (kg, m, rad) and the arbitrary units defined in DAVE-ML.
    """
    # Base multipliers to reach SI
    TO_SI = {
        'slug': 14.5939029,
        'lbm': 0.45359237,
        'ft': 0.3048,
        'ft2': 0.09290304,
        'ft^2': 0.09290304,
        'ft_s' : 0.3048,
        'slugft2': 1.35581795,
        'slugft^2': 1.35581795,
        'lbf': 4.4482216153,
        'ftlbf': 1.3558179483,
        'deg': math.pi / 180.0,
        '_deg': 180.0 / math.pi,
        'rad': 1.0,
        '_rad': 1.0,
        'rad_s': 1.0,
        'nmi_h': 0.51444444,
        'nd': 1.0,
        'pct': 1.0,
        'frac': 1.0,
        'none': 1.0,
        '': 1.0
    }

    @staticmethod
    def _clean_unit(unit_str: str) -> str:
        """Strips formatting characters to ensure robust dictionary matching."""
        if not unit_str:
            return ''
        # Convert to lowercase and strip spaces, hyphens, and asterisks
        return unit_str.lower().replace(' ', '').replace('-', '').replace('*', '')

    @classmethod
    def to_si(cls, value: float, unit_str: str, log: bool = False) -> float:
        """Converts a value from DAVE-ML units into SI units."""
        clean_unit = cls._clean_unit(unit_str)
        
        if clean_unit not in cls.TO_SI:
            raise ValueError(f"UnitConverter missing definition for DAVE-ML unit: '{unit_str}' (Parsed as '{clean_unit}')")
        
        if log:
            print(f"unit_str: {unit_str} converted by {cls.TO_SI[clean_unit]}")
        return value * cls.TO_SI[clean_unit]

    @classmethod
    def from_si(cls, value: float, unit_str: str, log: bool = False) -> float:
        """Converts an SI value from the EOM solver into the target DAVE-ML unit."""
        clean_unit = cls._clean_unit(unit_str)
        
        if clean_unit not in cls.TO_SI:
            raise ValueError(f"UnitConverter missing definition for DAVE-ML unit: '{unit_str}' (Parsed as '{clean_unit}')")
        
        if log:
            print(f"unit_str: {unit_str} converted by {cls.TO_SI[clean_unit]}")
        return value / cls.TO_SI[clean_unit]