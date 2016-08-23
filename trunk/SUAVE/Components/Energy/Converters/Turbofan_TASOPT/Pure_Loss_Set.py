# Pure_Loss_Set.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

class Pure_Loss_Set(Energy_Component):
    """Class used to determine flow properties with a pure loss."""
    
    def __defaults__(self):
        
        self.tag = 'Enthalpy_Different_Set'
        self.inputs.working_fluid = Data()
        self.pressure_ratio = 1.
        
    def compute_flow(self):
         
        Tti = self.inputs.total_temperature
        Pti = self.inputs.total_pressure
        Hti = self.inputs.total_enthalpy
        pi  = self.pressure_ratio
        
        Ptf = Pti*pi
        
        self.outputs.total_temperature = Tti
        self.outputs.total_pressure    = Ptf
        self.outputs.total_enthalpy    = Hti
        