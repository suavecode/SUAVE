""" Internal_Combustion.py: internal combustion (piston-cylinder) engine class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
# from SUAVE.Methods.Power import RunFuelCell
from Propulsor import Propulsor
import copy
import numpy as np

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Motor_FC(Propulsor):   #take simple engine module and power it with a fuel cell
    def __defaults__(self):
        self.tag = 'Electric Motor with Fuel Cell'
        self.D = 0.0                    # prop diameter (m)
        self.F_min_static = 0.0         # static thrust corresponding to min throttle
        self.F_max_static = 0.0         # static thrust corresponding to max throttle
        self.mdot_min_static = 0.0      # mass flow corresponding to min throttle
        self.mdot_max_static = 0.0      # mass flow corresponding to max throttle
        self.propellant = SUAVE.Attributes.Propellants.Jet_A1()
        self.e = 0.8
        
    def sizing(self,state):
        mach = state.M
        temp = state.T
        gamma = 1.4
        R = 286.9
        a = np.sqrt(gamma*R*temp)
        U = mach*a
        P = self.F_max_static*U
        spec_energy = self.propellant.specific_energy
        self.mdot_max_static = P/self.e/self.propellant.specific_energy

    def __call__(self,eta,segment):
        

        F = self.F_min_static + (self.F_max_static - self.F_min_static)*eta
        mdot = self.mdot_min_static + (self.mdot_max_static - self.mdot_min_static)*eta
   
        CF = F/(segment.q*np.pi*(self.D/2)**2)

        Isp = F/(mdot*segment.g0)
        #eta_Pe=F*segment.V/Preqfc
        eta_Pe=0
        return CF, Isp, eta_Pe