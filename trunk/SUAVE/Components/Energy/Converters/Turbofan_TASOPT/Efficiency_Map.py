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
        self.C              = 0.1
        
    def compute_efficiency(self,pi,md):

        eta_0 = self.design_polytropic_efficiency
        c  = self.c
        C  = self.C
        #pi  = self.pressure_ratio
        piD = self.design_pressure_ratio
        mD  = self.design_mass_flow
        #md  = self.mass_flow
        
        mb = md/mD
        pb = (pi-1.)/(piD-1.)

        
        eta_offdesign = eta_0*(1. - C*(np.abs(pb/mb-1.)**c))
    
        return eta_offdesign