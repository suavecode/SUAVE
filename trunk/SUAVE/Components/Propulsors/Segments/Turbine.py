""" Turbine.py: Turbine Segment of a Propulsor """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from Segment import Segment

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Turbine(Segment):   

    """ A Turbine Segment of a Propulsor """

    def __defaults__(self):

        self.tag = 'Turbine'

    def __call__(self, Wdot=None):

        """  Turbine(): populates final p_t and T_t values based on initial p_t, T_t values and efficiencies
    
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
        g = self.gamma

        # apply polytropic efficiency, if defined by user
        if self.eta_polytropic is not None:
            if len(self.eta_polytropic) == 1:
                self.Tt_ratio = self.pt_ratio**((g-1)*self.eta_polytropic/g)
                self.eta = (self.pt_ratio**((g-1)/g) - 1)/(self.Tt_ratio - 1)
            else:
                print "Error in Turbine: polytropic efficiency must be a single defined value"
        else:
            # error checking on eta
            if self.eta is not None:
                if len(self.eta) == 1:
                    Tt_ratio_ideal = self.pt_ratio**((g-1)/g)
                    self.Tt_ratio = 1 + (Tt_ratio_ideal - 1)*self.eta
                    self.eta_polytropic = g*np.log(self.Tt_ratio)/((g-1)*np.log(self.pt_ratio))
                else:
                    print "Error in Turbine: efficiency must be a single defined value"
            else:
                print "Error in Turbine: efficiency must be a single defined value"

        # compute outlet conditions 
        self.pt[1] = self.pt[0]*self.pt_ratio
        self.Tt[1] = self.Tt[0]*self.Tt_ratio
        self.ht = self.Tt*self.cp

        # compute shaft work done


        return Wdot