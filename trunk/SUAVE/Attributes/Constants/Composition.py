# Composition.py
# 
# Created: Mar 2014,     J. Sinsay
# Modified: Jan, 2016,  M. Vegh



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# initialized constants
from Constant import Constant

# exceptions/warnings
from warnings import warn

# ----------------------------------------------------------------------
#  Composition Constant Class
# ----------------------------------------------------------------------

class Composition(Constant):
    """ Composition base class for gas mixtures """
    def __defaults__(self):
        pass
    
    def __check__(self):
        
        # check that composition sums to 1.0
        total = 0.0
        for v in self.values():
            total += v
        other = 1.0 - total

        # set other if needed
        if other != 0.0: 
            if self.has_key('Other'):
                other += self.Other
            self.Other = other
            self.swap('Other',-1)
                
        # check for negative other
        if other < 0.0:
            warn('Composition adds to more than 1.0',Data_Warning)
