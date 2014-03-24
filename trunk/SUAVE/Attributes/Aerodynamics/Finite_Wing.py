""" Finite_Wing.py: A simple finite wing aerodynamic model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Aerodynamics import Aerodynamics

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
    
class Finite_Wing(Aerodynamics):
    """ SUAVE.Attributes.Aerodyanmics.FiniteWing: a simple finite wing lift and drag model """
    
    def __defaults__(self):
        self.tag = 'Finite Wing'
        self.S = 1.0                                        # reference area (m^2)
        self.AR = 0.0                                       # aspect ratio
        self.e = 1.0                                        # Oswald factor
        self.CD0 = 0.0                                      # CD at zero lift
        self.CL0 = 0.0                                      # CL at alpha = 0.0
        self.dCLdalpha = 2*np.pi                            # dCL/dalpha

    def __call__(self,alpha,segment):

        CL = self.CL0 + self.dCLdalpha*alpha                # linear lift vs. alpha
        CD = self.CD0 + (CL**2)/(np.pi*self.AR*self.e)      # parbolic drag

        return CD, CL
