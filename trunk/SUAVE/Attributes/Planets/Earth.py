# Constants.py: Physical constants and helepr functions
# 
# Created By:       J. Sinsay
# Updated:          M. Colonno   04/09/2013
#                   T. Lukaczyk  06/23/2013

""" SUAVE Data Class for Constants """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
from Planet import Planet
from SUAVE.Attributes.Constants import Composition
     
# ----------------------------------------------------------------------
#  Earth
# ----------------------------------------------------------------------
     
class Earth(Planet):
    """ Physical constants specific to Earth and Earth's atmosphere """
    def __defaults__(self):
        self.tag = 'Earth'
        self.mass              = 5.98e24  # kg
        self.mean_radius       = 6.371e3  # km
        self.sea_level_gravity = 9.80665  # m/s^2   
        self.HitchHikersGuide  = 'MostlyHarmless'


