# Fan.py
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

# ----------------------------------------------------------------------
#  Fan Component
# ----------------------------------------------------------------------

class Fan(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Fan
        a Fan component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #set the default values
        self.tag ='Fan'
        self.polytropic_efficiency          = 1.0
        self.pressure_ratio                 = 1.0
        self.inputs.stagnation_temperature  = 0.
        self.inputs.stagnation_pressure     = 0.
        self.outputs.stagnation_temperature = 0.
        self.outputs.stagnation_pressure    = 0.
        self.outputs.stagnation_enthalpy    = 0.
    
    
    
    def compute(self,conditions):
        
        #unpack the values
        
        #unpack from conditions

        
        # unpack from inputs
        working_fluid          = self.inputs.working_fluid
        gamma_c = working_fluid.gamma        
        e_f = self.polytropic_efficiency
        pi_f = self.pressure_ratio
        
        # compute values
        
        tau_f = pi_f**((gamma_c-1.0)/(gamma_c*e_f))
        eta_f = (pi_f**((gamma_c-1.0)/gamma_c) - 1.0)/(tau_f -1.0)         
        
        #pack the computed quantities into outputs
        self.outputs.tau_f  = tau_f
        self.outputs.eta_f  = eta_f
    

    __call__ = compute
