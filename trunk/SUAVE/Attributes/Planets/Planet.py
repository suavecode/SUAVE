## @ingroup Attributes-Planets
# Planet.py
# 
# Created:  Unk, 2013, J. Sinsay
# Modified: Apr, 2015, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Constants import Constant

# ----------------------------------------------------------------------
#  Planet Constant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Planets
class Planet(Constant):
    """Holds constants for a planet
    
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
        None

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """           
        self.mass              = 0.0  # kg
        self.mean_radius       = 0.0  # m
        self.sea_level_gravity = 0.0  # m/s^2   
