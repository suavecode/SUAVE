# Compressor.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Difference_Set import Pressure_Difference_Set

class Compressor(Pressure_Difference_Set):
    """Turbine computations based on TASOPT model"""
    
    def __defaults__(self):
        self.design_polytropic_efficiency = 1.
        self.design_pressure_ratio        = 1.5
        self.efficiency_map = None
        self.speed_map      = None
        self.speed_change_by_pressure_ratio = 0.
        self.speed_change_by_mass_flow      = 0.
       
    def compute(self):
        
        self.compute_flow()
        
    def compute_performance(self):
        
        # This will change the efficiency
        pi = self.pressure_ratio
        mdotc = self.corrected_mass_flow
        self.polytropic_efficiency = self.efficiency_map.compute_efficiency(pi,mdotc)
        N, dN_pi, dN_mf           = self.speed_map.compute_speed(pi,mdotc)
        self.corrected_speed      = N
        self.speed_change_by_pressure_ratio = dN_pi
        self.speed_change_by_mass_flow      = dN_mf

    def set_design_condition(self):
        
        self.polytropic_efficiency = self.design_polytropic_efficiency
        self.pressure_ratio        = self.design_pressure_ratio