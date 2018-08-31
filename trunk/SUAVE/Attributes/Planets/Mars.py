## @ingroup Attributes-Planets
# Mars.py
# 
# Created:  Unk. 2013, J. Sinsay
# Modified: Apr, 2015, A. Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Planet import Planet

# ----------------------------------------------------------------------
#  Mars Constant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Planets
class Mars(Planet):
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
        self.mass              = 639e21 # kg
        self.mean_radius       = 339e4  # m
        self.sea_level_gravity = 3.711  # m/s^2   

