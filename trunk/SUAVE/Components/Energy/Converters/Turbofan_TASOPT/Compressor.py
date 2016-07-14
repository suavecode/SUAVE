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
        self.efficiency_map = None
        self.speed_map      = None
        self.speed_change_by_pressure_ratio = 0.
        self.speed_change_by_mass_flow      = 0.
       
    def compute_design(self):
        
        self.polytopic_efficiency = self.design_polytropic_efficiency
        self.compute_flow()
        
    def compute_offdesign(self):
        
        pi = self.pessure_ratio
        mdotc = self.corrected_mass_flow
        self.polytopic_efficiency = self.efficiency_map(pi,mdotc)
        N, dN_pi, dN_mf           = self.speed_map(pi,mdotc)
        self.corrected_speed      = N
        self.speed_change_by_pressure_ratio = dN_pi
        self.speed_change_by_mass_flow      = dN_mf
        
        self.compute_flow()