# Constants.py: Physical constants and helepr functions
# 
# Created By:       J. Sinsay
# Updated:          M. Colonno   04/09/2013
#                   T. Lukaczyk  06/23/2013

""" SUAVE Data Class for Constants """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# initialized constants
from Constant import Constant

# exceptions/warnings
from SUAVE.Core import Data_Warning
from warnings import warn

# ----------------------------------------------------------------------
#  Constants 
# ----------------------------------------------------------------------

class Composition(Constant):
    """ Constant Base Class """
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
