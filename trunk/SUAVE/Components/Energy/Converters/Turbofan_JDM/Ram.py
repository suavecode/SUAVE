# Ram.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Ram Component
# ----------------------------------------------------------------------

class Ram(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Ram
        a Ram class that is used to convert static properties into
        stagnation properties
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #set the deafult values
        self.tag = 'Ram'
        self.g_c = 1.0
        self.outputs.stagnation_temperature  = 1.0
        self.outputs.stagnation_pressure     = 1.0
        self.inputs.working_fluid = Data()

    def compute(self,conditions):
        
        #unpack from conditions
        T0 = conditions.freestream.temperature
        M0 = conditions.freestream.mach_number
        
        #unpack from inputs
        g_c = self.g_c
        pi_d_max = self.pi_d_max
        working_fluid          = self.inputs.working_fluid
        gamma_c = working_fluid.gamma
        R_c     = working_fluid.R
        
    
        tau_r = 1.0 + (gamma_c - 1.0)*0.5*(M0**2.0)
    
        pi_r = tau_r**(gamma_c/(gamma_c-1.0))
    
        eta_r = 1.0
    
        if(M0>1.0):
            eta_r = 1.0 - 0.075*(M0 - 1.0)**1.35
    
    
    
        pi_d = pi_d_max*eta_r        
        
        #pack computed outputs
        
        
        
        #pack the values into conditions
        self.outputs.pi_r            = pi_r
        self.outputs.tau_r           = tau_r
        self.outputs.eta_r           = eta_r
    
    
    
    __call__ = compute