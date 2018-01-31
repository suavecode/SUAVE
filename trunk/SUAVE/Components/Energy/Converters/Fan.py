## @ingroup Components-Energy-Converters
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
## @ingroup Components-Energy-Converters
class Fan(Energy_Component):
    """This is a fan component typically used in a turbofan.
    Calling this class calls the compute function.
    
    Assumptions:
    Pressure ratio and efficiency do not change with varying conditions.

    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    """
    
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """         
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
        """ This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        Constant polytropic efficiency and pressure ratio

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
        self.inputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]

        Outputs:
        self.outputs.
          stagnation_temperature              [K]  
          stagnation_pressure                 [Pa]
          stagnation_enthalpy                 [J/kg]
          work_done                           [J/kg]

        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
        """          
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
