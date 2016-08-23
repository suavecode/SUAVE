# Compressor.py
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
#  Compressor Component
# ----------------------------------------------------------------------

class Compressor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Compressor
        a compressor component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #set the default values
        self.tag                             = 'Compressor'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.inputs.stagnation_temperature   = 0.
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 0.
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
    

    def compute(self,conditions):
        
        #unpack the values
        
        # unpack from inputs
        working_fluid          = self.inputs.working_fluid
        gamma_c = working_fluid.gamma        
        e = self.polytropic_efficiency
        pi = self.pressure_ratio
    
        # compute values
    
        tau = pi**((gamma_c-1.0)/(gamma_c*e))
        eta = (pi**((gamma_c-1.0)/gamma_c) - 1.0)/(tau -1.0)         
    
        #pack the computed quantities into outputs
        self.outputs.tau  = tau
        self.outputs.eta  = eta
    
    
    __call__ = compute
