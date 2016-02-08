# Earth.py
# 
# Created:  Unk, 2013, J. Sinsay
# Modified: Apr, 2015, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Planet import Planet
     
# ----------------------------------------------------------------------
#  Earth Constant Class
# ----------------------------------------------------------------------
 
class Earth(Planet):
    """ Physical constants specific to Earth"""
    def __defaults__(self):
        self.tag = 'Earth'
        self.mass              = 5.98e24  # kg
        self.mean_radius       = 6.371e6  # m
        self.sea_level_gravity = 9.80665  # m/s^2   
        self.HitchHikersGuide  = 'MostlyHarmless'


