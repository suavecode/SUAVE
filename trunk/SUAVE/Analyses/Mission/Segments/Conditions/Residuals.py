# Residuals.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Conditions import Conditions

# ----------------------------------------------------------------------
#  Residuals
# ----------------------------------------------------------------------

class Residuals(Conditions):
    
    def __defaults__(self):
        self.tag = 'residuals'