# Nozzle.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pure_Loss_Set import Pure_Loss_Set

class Nozzle(Pure_Loss_Set):
    
    def __defaults__(self):
        pass
    
    def compute(self):
        self.compute_flow()
        
    def compute_static(self,exhaust_velocity,exhaust_temperature,ambient_pressure):
        
        gamma = self.inputs.working_fluid.gamma
        R     = self.inputs.working_fluid.R
        
        Tt    = self.outputs.total_temperature
        Pt    = self.outputs.total_pressure
        ht    = self.outputs.total_enthalpy
        
        ex_M = exhaust_velocity/np.sqrt(gamma*R*exhaust_temperature)
        
        R     = np.ones(np.shape(ht))*R
        P     = np.ones(np.shape(ht))
        M     = np.ones(np.shape(ht))
        
        P[ex_M<1.] = ambient_pressure[ex_M<1.]
        M[ex_M<1.] = np.sqrt(((Pt[ex_M<1.]/P[ex_M<1.])**((gamma[ex_M<1.]-1.)/gamma[ex_M<1.]) - 1.)*2./(gamma[ex_M<1.]-1.))
        
        #M[ex_M>=1.] = 1. pre-set to 1.
        P[ex_M>=1.] = Pt[ex_M>=1.]/(1.+(gamma[ex_M>=1.]-1.)/2.*M[ex_M>=1.]*M[ex_M>=1.])**(gamma[ex_M>=1.]/(gamma[ex_M>=1.]-1.))
            
        T = Tt/(1.+(gamma-1.)/2.*M*M)
        h = ht*T/Tt
        u = np.sqrt(2.*(ht-h))
        rho = P/(R*T)
        
        self.outputs.static_density = rho
        self.outputs.flow_speed     = u
        
        
    def size(self,mdot,exhaust_velocity,exhaust_temperature,ambient_pressure,flow_ratio = 1.):
        
        self.compute_static(exhaust_velocity, exhaust_temperature, ambient_pressure)
        rho = self.outputs.static_density
        u   = self.outputs.flow_speed
        
        # the bypass ratio can be passed as flow_ratio for fan nozzles
        A   = flow_ratio*mdot/(rho*u)
        
        self.exit_area = A
        
        