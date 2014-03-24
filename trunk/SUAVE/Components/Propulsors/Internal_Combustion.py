""" Internal_Combustion.py: internal combustion (piston-cylinder) engine class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data
from Propulsor import Propulsor
import numpy as np

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Internal_Combustion(Propulsor):   ### Over-simplified but functional; needs additional work (MC)
    
    def __defaults__(self):
        self.tag = 'Internal Combustion Engine'
        self.D = 0.0                    # prop diameter (m)
        self.F_min_static = 0.0         # static thrust corresponding to min throttle
        self.F_max_static = 0.0         # static thrust corresponding to max throttle
        self.mdot_min_static = 0.0      # mass flow corresponding to min throttle
        self.mdot_max_static = 0.0      # mass flow corresponding to max throttle

    def __call__(self,eta,segment):

        F = self.F_min_static + (self.F_max_static - self.F_min_static)*eta
        mdot = self.mdot_min_static  + (self.mdot_max_static - self.mdot_min_static)*eta

        CF = F/(segment.q*np.pi*(self.D/2)**2)
        Isp = F/(mdot*segment.g0)

        return CF, Isp, 0.0