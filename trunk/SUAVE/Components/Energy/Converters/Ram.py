## @ingroup Components-Energy-Converters
# Ram.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald
#           Jan 2018, W. Maier

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
## @ingroup Components-Energy-Converters
class Ram(Energy_Component):
    """This represent the compression of incoming air flow.
    Calling this class calls the compute function.

    Assumptions:
    None

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
        #set the deafult values
        self.tag = 'Ram'
        self.outputs.stagnation_temperature  = 1.0
        self.outputs.stagnation_pressure     = 1.0
        self.inputs.working_fluid = Data()

    def compute(self,conditions):
        """ This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        None

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          pressure
          temperature
          mach_number
        self.inputs.working_fluid

        Outputs:
        self.outputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          gas_specific_constant               [J/(kg K)]
        conditions.freestream.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          gas_specific_constant               [J/(kg K)]
          speed_of_sound                      [m/s]

        Properties Used:
        None
        """
        #unpack from conditions
        Po = conditions.freestream.pressure
        To = conditions.freestream.temperature
        M  = conditions.freestream.mach_number

        #unpack from inputs
        working_fluid          = self.inputs.working_fluid

        #method to compute the ram properties

        #computing the working fluid properties
        gamma                  = working_fluid.compute_gamma(To,Po) 
        Cp                     = working_fluid.compute_cp(To,Po)
        R                      = working_fluid.gas_specific_constant

        #Compute the stagnation quantities from the input static quantities
        stagnation_temperature = To*(1.+((gamma-1.)/2.*M*M))
        stagnation_pressure    = Po*((1.+(gamma-1.)/2.*M*M )**(gamma/(gamma-1.)))

        #pack computed outputs
        #pack the values into conditions
        self.outputs.stagnation_temperature              = stagnation_temperature
        self.outputs.stagnation_pressure                 = stagnation_pressure
        self.outputs.isentropic_expansion_factor         = gamma
        self.outputs.specific_heat_at_constant_pressure  = Cp
        self.outputs.gas_specific_constant               = R

        #pack the values into outputs
        conditions.freestream.stagnation_temperature               = stagnation_temperature
        conditions.freestream.stagnation_pressure                  = stagnation_pressure
        conditions.freestream.isentropic_expansion_factor          = gamma
        conditions.freestream.specific_heat_at_constant_pressure   = Cp
        conditions.freestream.gas_specific_constant                = R

    __call__ = compute