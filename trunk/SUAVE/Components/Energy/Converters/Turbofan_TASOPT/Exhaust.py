# Exhaust.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Difference_Set import Pressure_Difference_Set

class Exhaust(Pressure_Difference_Set):
    """Exhaust computations based on TASOPT model"""
    
    def __defaults__(self):
        self.design_polytropic_efficiency = 1.
        self.efficiency_map = None
        self.speed_map      = None
        self.speed_change_by_pressure_ratio = 0.
        self.speed_change_by_mass_flow      = 0.
       
    def compute(self):
        
        pi = self.pessure_ratio
        self.compute_flow()
        
        Hti = self.inputs.total_enthalpy
        Htf = self.outputs.total_enthalpy
        
        self.outputs.flow_speed = np.sqrt(2.*(Hti-Htf))