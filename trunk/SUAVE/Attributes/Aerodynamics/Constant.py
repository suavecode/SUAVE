""" Constant.py: A constant lift and drag aerodynamic model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Aerodynamics import Aerodynamics

# ----------------------------------------------------------------------
#  Runway
# ----------------------------------------------------------------------
    
class Constant(Aerodynamics):
    """ SUAVE.Attributes.Aerodyanmics.Constant: a constant lift and drag model """

    def __defaults__(self):
        self.tag = 'Constant Properties'
        self.CD = 1.0
        self.CL = 1.0
        self.S = 1.0            # reference area (m^2)

    def __call__(self,alpha,segment):
        return self.CD, self.CL
