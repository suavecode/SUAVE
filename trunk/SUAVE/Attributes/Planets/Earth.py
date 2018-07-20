## @ingroup Attributes-Planets
# Earth.py
# 
# Created:  Unk, 2013, J. Sinsay
# Modified: Apr, 2015, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Planet import Planet
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Earth Constant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Planets
class Earth(Planet):
    """Holds constants for Earth
    
    Assumptions:
    None
    
    Source:
    None
    """
    def __defaults__(self):
        """This sets the default values.

        Assumptions:
        None

        Source:
        Values commonly available

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """         
        self.tag = 'Earth'
        self.mass              = 5.98e24 *Units['kg']     # [kg]
        self.mean_radius       = 6.371e6 *Units['m']      # [m]
        self.sea_level_gravity = 9.80665 *Units['m/s^2']  # [m/s^2] 
        self.HitchHikersGuide  = 'MostlyHarmless'