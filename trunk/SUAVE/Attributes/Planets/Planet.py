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
from SUAVE.Attributes.Constants import Constant, Composition

# ----------------------------------------------------------------------
#  Planet
# ----------------------------------------------------------------------
     
class Planet(Constant):
    """ Physical constants of big space rocks """
    def __defaults__(self):
        self.mass              = 0.0  # kg
        self.mean_radius       = 0.0  # m
        self.sea_level_gravity = 0.0  # m/s^2   
