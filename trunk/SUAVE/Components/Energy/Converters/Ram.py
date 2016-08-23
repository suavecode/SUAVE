# Ram.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald

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
        self.outputs.stagnation_temperature  = 1.0
        self.outputs.stagnation_pressure     = 1.0
        self.inputs.working_fluid = Data()

    def compute(self,conditions):
        
        #unpack from conditions
        Po = conditions.freestream.pressure
        To = conditions.freestream.temperature
        M = conditions.freestream.mach_number
        
        #unpack from inputs
        working_fluid          = self.inputs.working_fluid


        #method to compute the ram properties
        
        #computing the working fluid properties
        gamma                  = 1.4
        Cp                     = 1.4*287.87/(1.4-1)
        R                      = 287.87
        
        ao                     =  np.sqrt(Cp/(Cp-R)*R*To)
        
        #Compute the stagnation quantities from the input static quantities
        stagnation_temperature = To*(1+((gamma-1)/2 *M*M))
        stagnation_pressure    = Po* ((1+(gamma-1)/2 *M*M )**3.5 )
        
        
        
        #pack computed outputs
        
        #pack the values into conditions
        self.outputs.stagnation_temperature              = stagnation_temperature
        self.outputs.stagnation_pressure                 = stagnation_pressure
        self.outputs.isentropic_expansion_factor         = gamma
        self.outputs.specific_heat_at_constant_pressure  = Cp
        self.outputs.universal_gas_constant              = R
        
        #pack the values into outputs
        conditions.freestream.stagnation_temperature               = stagnation_temperature
        conditions.freestream.stagnation_pressure                  = stagnation_pressure
        conditions.freestream.isentropic_expansion_factor          = gamma
        conditions.freestream.specific_heat_at_constant_pressure   = Cp
        conditions.freestream.universal_gas_constant               = R
        conditions.freestream.speed_of_sound                       = ao
    
    
    
    __call__ = compute

