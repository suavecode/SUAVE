# Expansion_Nozzle.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units

# package imports
import numpy as np

from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Propulsion.fm_id import fm_id

# ----------------------------------------------------------------------
#  Expansion Nozzle Component
# ----------------------------------------------------------------------

class Expansion_Nozzle_Fan(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #set the defaults
        self.tag = 'Nozzle'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.inputs.stagnation_temperature   = 0.
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 0.
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
    
    
    
    def compute(self,conditions):
        
        #unpack the values
        gamma_c    = self.inputs.gamma_c
        pi_r       = self.inputs.pi_r
        pi_d       = self.inputs.pi_d
        pi_f       = self.inputs.pi_f
        pi_fn      = self.inputs.pi_fn
        tau_r      = self.inputs.tau_r
        tau_f      = self.inputs.tau_f

        
        P0_P19 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_f*pi_fn)
        
        if(P0_P19>1.0):
            P0_P19 = 1.0         
        
        pt19_p19 = P0_P19*pi_r*pi_d*pi_f*pi_fn

        M19 = np.sqrt(2.0/(gamma_c-1.0)*((pt19_p19)**((gamma_c-1.0)/gamma_c)-1.0))
        
        T19_T0 = tau_r*tau_f/(pt19_p19**((gamma_c-1.0)/gamma_c))
        
        V19_a0 = M19*np.sqrt(T19_T0)
        
        #pack computed quantities into outputs
        self.outputs.P0_P19  = P0_P19
        self.outputs.T19_T0  = T19_T0
        self.outputs.V19_a0  = V19_a0
        self.outputs.M19     = M19
        
    

    __call__ = compute