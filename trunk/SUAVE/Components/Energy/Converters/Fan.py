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
        gamma     = conditions.freestream.isentropic_expansion_factor
        Cp        = conditions.freestream.specific_heat_at_constant_pressure
        
        #unpack from inputs
        Tt_in     = self.inputs.stagnation_temperature
        Pt_in     = self.inputs.stagnation_pressure
        
        #unpack from self
        pid       = self.pressure_ratio
        etapold   = self.polytropic_efficiency
        
        #method to compute the fan properties
        
        #Compute the output stagnation quantities 
        ht_in     = Cp*Tt_in
        
        Pt_out    = Pt_in*pid
        Tt_out    = Tt_in*pid**((gamma-1)/(gamma*etapold))
        ht_out    = Cp*Tt_out    
        
        #computing the wok done by the fan (for matching with turbine)
        work_done = ht_out- ht_in
        
        #pack the computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.work_done               = work_done
    

    __call__ = compute
