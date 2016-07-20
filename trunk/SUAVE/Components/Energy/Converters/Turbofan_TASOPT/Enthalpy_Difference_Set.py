# Enthalpy_Difference_Set.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

class Enthalpy_Difference_Set(Energy_Component):
    """Class used to determine flow properties with a specified enthalpy change."""
    
    def __defaults__(self):
        
        self.tag = 'Enthalpy_Different_Set'
        #self.P0  = 1.
        #self.T0  = 1.
        #self.Dh  = 1.
        self.polytropic_efficiency = 1.
        self.inputs.working_fluid = Data()
        
    def compute_flow(self):
        
        Dh  = self.inputs.delta_enthalpy    
        Tti = self.inputs.total_temperature
        Pti = self.inputs.total_pressure
        Hti = self.inputs.total_enthalpy
        
        cp  = self.inputs.working_fluid.specific_heat
        gamma = self.inputs.working_fluid.gamma
        
        Ttf = Tti + Dh/cp
        Ptf = Pti*(Ttf/Tti)**(gamma/((gamma-1.)*self.polytropic_efficiency))
        Htf = Hti + Dh
        
        self.outputs.total_temperature = Ttf
        self.outputs.total_pressure    = Ptf
        self.outputs.total_enthalpy    = Htf