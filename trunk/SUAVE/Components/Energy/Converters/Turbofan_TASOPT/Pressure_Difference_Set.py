# Pressure_Difference_Set.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

class Pressure_Difference_Set(Energy_Component):
    """Class used to determine flow properties with a specified pressure change."""
    
    def __defaults__(self):
        
        self.tag = 'Pressure_Difference_Set'
        self.pressure_ratio = 1.
        self.polytropic_efficiency = 1.
        self.inputs.working_fluid = Data()
        
    def compute_flow(self):
            
        pi  = self.pressure_ratio
        Tti = self.inputs.total_temperature
        Pti = self.inputs.total_pressure
        Hti = self.inputs.total_enthalpy
        eta = self.polytropic_efficiency
        
        cp  = self.inputs.working_fluid.specific_heat
        gamma = self.inputs.working_fluid.gamma
        
        Ttf = Tti*pi**((gamma-1.)/(gamma*eta))
        Ptf = Pti*pi
        Htf = cp*Ttf
        
        self.outputs.total_temperature = Ttf
        self.outputs.total_pressure    = Ptf
        self.outputs.total_enthalpy    = Htf