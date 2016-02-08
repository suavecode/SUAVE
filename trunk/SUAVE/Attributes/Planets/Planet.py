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
     
class Planet(Constant):
    """ Physical constants of big space rocks """
    def __defaults__(self):
        self.mass              = 0.0  # kg
        self.mean_radius       = 0.0  # m
        self.sea_level_gravity = 0.0  # m/s^2   
