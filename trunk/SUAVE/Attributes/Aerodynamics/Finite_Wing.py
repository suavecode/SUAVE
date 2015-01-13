""" Finite_Wing.py: A simple finite wing aerodynamic model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Attributes.Aerodynamics import Aerodynamics
from SUAVE.Core import Data


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

    def __call__(self,conditions):
        # unpack
        q             = conditions.freestream.dynamic_pressure
        Sref          = self.S       
        
        alpha=conditions.aerodynamics.angle_of_attack
        
        CL = self.CL0 + self.dCLdalpha*alpha                # linear lift vs. alpha
        CD = self.CD0 + (CL**2)/(np.pi*self.AR*self.e)      # parbolic drag
        
        # pack results
        results = Data()
        results.lift_coefficient = CL
        results.drag_coefficient = CD
        
        N = q.shape[0]
        L = np.zeros([N,3])
        D = np.zeros([N,3])
        
        L[:,2] = ( -CL * q * Sref )[:,0]
        D[:,0] = ( -CD * q * Sref )[:,0]
        
        results.lift_force_vector = L
        results.drag_force_vector = D        

        return results
