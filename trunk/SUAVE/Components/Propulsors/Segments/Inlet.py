""" Inlet.py: Inlet Propulsor Segment """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from Segment import Segment

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Inlet(Segment):   

    def __defaults__(self):

        self.tag = 'Inlet'

    def __call__(self,thermo,power):

        """  Inlet(): populates final p_t and T_t values based on initial p_t, T_t values and efficiencies
    
         Inputs:    self.pt_ratio = stagnation pressure ratio of inlet          (float)     (required)
                    self.Tt_ratio = stagnation temperature ratio of inlet       (float)     (required)

         Outputs:   thermo.pt[f] = stagnation pressure at compressor outlet     (float)     (required)
                    thermo.Tt[f] = stagnation temperature at compressor outlet  (float)     (required)
                    thermo.ht[f] = stagnation enthalpy at compressor outlet     (float)     (required)

        """

        if self.active:
            pt[self.f] = pt[self.i]*self.pt_ratio
            Tt[self.f] = Tt[self.i]*self.Tt_ratio
            Tt[self.f] = Tt[self.f]*self.cp[self.f]
        
        return