## @ingroup Attributes-Planets
# Earth.py
# 
# Created:  Unk, 2013, J. Sinsay
# Modified: Apr, 2015, E. Botero
#           Sep, 2018, W. Maier

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Planet import Planet
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
        self.mass              = 5.98e24  # kg
        self.mean_radius       = 6.371e6  # m
        self.sea_level_gravity = 9.80665  # m/s^2   
        self.HitchHikersGuide  = 'MostlyHarmless'

    def compute_gravity(self, H=0.0):
        """Compute the gravitational acceleration at altitude
            
        Assumptions:      

        Source:

        Inputs:
        H     [m] (Altitude)

        Outputs:
        g     [m/(s^2)] (Gravity)

        Properties Used:
        None
        """          
        # Unpack
        g0  = self.sea_level_gravity
        Re  = self.mean_radius
        Alt = H*Units['m']
        
        # Calculate gravity
        gh = g0*(Re/(Re+H))**2.0
        
        return gh