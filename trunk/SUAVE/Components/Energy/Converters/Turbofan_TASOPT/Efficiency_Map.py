# Efficiency_Map.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# Built from TASOPT maps, based on Anil's TASOPT code

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data

class Efficiency_Map(Data):
    
    def __defaults__(self):
        self.polytropic_efficiency = .9
        self.c              = 3.
        self.C              = 2.5
        
    def compute_speed(self):

        eta_0 = self.polytropic_efficiency
        c  = self.c
        C  = self.C
        pi = self.pressure_ratio
        pD = self.design_pressure_ratio
        mD = self.design_mass_flow
        md = self.inputs.mass_flow
        
        mb = md/mD
        pb = (pi-1.)/(pD-1.)

        
        eta_offdesign = eta_0*(1. - C*(np.abs(pb/mb-1.)**c))
    
        return eta_offdesign  