# Turbine.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT import Enthalpy_Difference_Set

class Turbine(Enthalpy_Difference_Set):
    """Turbine computations based on TASOPT model"""
    
    def __defaults__(self):
        self.design_polytropic_efficiency = 1.
        self.speed_change_by_pressure_ratio = 0.
        self.speed_change_by_mass_flow      = 0.
       
    def compute(self):
        
        #self.polytopic_efficiency = self.design_polytropic_efficiency
        self.compute_flow()