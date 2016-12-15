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

class Expansion_Nozzle_Core(Energy_Component):
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
        gamma_t    = self.inputs.gamma_t
        pi_r       = self.inputs.pi_r
        pi_d       = self.inputs.pi_d
        pi_cL      = self.inputs.pi_cL
        pi_cH      = self.inputs.pi_cH
        pi_b       = self.inputs.pi_b
        pi_tL      = self.inputs.pi_tL
        pi_tH      = self.inputs.pi_tH
        pi_n       = self.inputs.pi_n
        pi_f       = self.inputs.pi_f
        tau_lamda  = self.inputs.tau_lambda
        tau_tH     = self.inputs.tau_tH
        tau_tL     = self.inputs.tau_tL
        R_c        = self.inputs.R_c
        R_t        = self.inputs.R_t
        c_pc       = self.inputs.c_pc
        c_pt       = self.inputs.c_pt

        
        P0_P9 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tL*pi_tH*pi_n*pi_f)
            
        if(P0_P9>1.0):
            P0_P9 = 1.0                 
        

        pt9_p9 = P0_P9*(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_n*pi_f)
        
        M9 = np.sqrt(2.0/(gamma_t-1.0)*((pt9_p9)**((gamma_t-1.0)/gamma_t)-1.0))
        
        T9_T0 = tau_lamda*tau_tL*tau_tH*c_pc/(pt9_p9**((gamma_t-1.0)/gamma_t)*c_pt)
        
        V9_a0 = M9*np.sqrt(gamma_t*R_t*T9_T0/(gamma_c*R_c))
        
        #pack computed quantities into outputs
        self.outputs.P0_P9  = P0_P9
        self.outputs.T9_T0  = T9_T0
        self.outputs.V9_a0  = V9_a0
        self.outputs.M9     = M9

        
    

    __call__ = compute