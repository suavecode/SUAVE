""" Compressor.py: Compressor Segment of a Propulsor """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from Segment import Segment

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Compressor(Segment):   

    """ A Compressor Segment of a Propulsor """

    def __defaults__(self):

        self.tag = 'Compressor'

    def __call__(self,thermo,power):

        """  Compressor(): populates final p_t and T_t values based on initial p_t, T_t values and efficiencies
    
         Inputs:    self.pt[0] = stagnation pressure at compressor inlet        (float)     (required)
                    self.Tt[0] = stagnation temperature at compressor inlet     (float)     (required)
                    self.eta = compressor efficiency                            (float)     (required if eta_polytropic not defined)
                    self.eta_polytropic = compressor polytropic efficiency      (float)     (required if eta not defined)

         Outputs:   self.pt[1] = stagnation pressure at compressor outlet       (float)     (required)
                    self.Tt[1] = stagnation temperature at compressor outlet    (float)     (required)
                    self.eta = compressor efficiency                            (float)     (if eta_polytropic is defined)
                    self.eta_polytropic = compressor polytropic efficiency      (float)     (if eta is defined)                                                                                             

        """

        # unpack
        i = self.i; f = self.f;
        g = (thermo.gamma[i] + thermo.gamma[f])/2

        # apply polytropic efficiency, if defined by user
        if self.eta_polytropic is not None:
            if len(self.eta_polytropic) == 1:
                self.Tt_ratio = self.pt_ratio**((g-1)/(g*self.eta_polytropic))
                self.eta = (self.pt_ratio**((g-1)/g) - 1)/(self.Tt_ratio - 1)
            else:
                print "Error in Compressor: polytropic efficiency must be a single defined value"
        else:
            # error checking on eta
            if self.eta is not None:
                if len(self.eta) == 1:
                    Tt_ratio_ideal = self.pt_ratio**((g-1)/g)
                    self.Tt_ratio = 1 + (Tt_ratio_ideal - 1)/self.eta
                    self.eta_polytropic = (g-1)*np.log(self.pt_ratio)/(g*np.log(self.Tt_ratio))
                else:
                    print "Error in Compressor: efficiency must be a single defined value"
            else:
                print "Error in Compressor: efficiency must be a single defined value"

        # compute outlet conditions 
        thermo.pt[f] = thermo.pt[i]*self.pt_ratio
        thermo.Tt[f] = thermo.Tt[i]*self.Tt_ratio
        thermo.ht[f] = thermo.Tt[f]*thermo.cp[f]

        return