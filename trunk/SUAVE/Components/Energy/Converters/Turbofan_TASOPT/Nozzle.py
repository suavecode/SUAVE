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
        
    def size(self,mdot,exhaust_velocity,exhaust_temperature,ambient_pressure,flow_ratio = 1.):
        
        
        
        gamma = self.inputs.working_fluid.gamma
        R     = self.inputs.working_fluid.R
        
        Tt    = self.outputs.total_temperature
        Pt    = self.outputs.total_pressure
        ht    = self.outputs.total_enthalpy
        
        ex_M = exhaust_velocity/np.sqrt(gamma*R*exhaust_temperature)
        
        if ex_M < 1.:
            P = ambient_pressure
            M = np.sqrt(((Pt/P)**((gamma-1.)/gamma) - 1.)*2./(gamma-1.))
        else:
            M = 1.
            P = Pt/(1.+(gamma-1.)/2.*M*M)**(gamma/(gamma-1.))
            
        T = Tt/(1.+(gamma-1.)/2.*M*M)
        h = ht*T/Tt
        u = np.sqrt(2.*(ht-h))
        rho = P/(R*T)
        
        # the bypass ratio can be passed as flow_ratio for fan nozzles
        A   = flow_ratio*mdot/(rho*u)
        
        self.exit_area = A
        
        