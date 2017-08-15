## @ingroup Components-Energy-Converters
# Shaft_Power_Off_Take.py
#
# Created:  Jun 2016, L. Kulik

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Shaft Power component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Shaft_Power_Off_Take(Energy_Component):
    """This is a component representing the power draw from the shaft.
    
    Assumptions:
    None

    Source:
    None
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
        self.power_draw = 0.0
        self.reference_temperature = 288.15
        self.reference_pressure = 1.01325 * 10 ** 5

    def compute(self, state):
        """ This computes the work done from the power draw.

        Assumptions:
        None

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        self.inputs.
          mdhc                  [-] Compressor nondimensional mass flow
          reference_temperature [K]
          reference_pressure    [Pa]

        Outputs:
        self.outputs.
          power                 [W]
          work_done             [J/kg] (if power draw is not zero)

        Properties Used:
        self.power_draw         [W]
        """  
        if self.power_draw == 0.0:
            self.outputs.work_done = np.array([0.0])

        else:

            mdhc = self.inputs.mdhc
            Tref = self.reference_temperature
            Pref = self.reference_pressure

            total_temperature_reference = self.inputs.total_temperature_reference
            total_pressure_reference    = self.inputs.total_pressure_reference

            self.outputs.power = self.power_draw

            mdot_core = mdhc * np.sqrt(Tref / total_temperature_reference) * (total_pressure_reference / Pref)

            self.outputs.work_done = self.outputs.power / mdot_core

            self.outputs.work_done[mdot_core == 0] = 0

    __call__ = compute
