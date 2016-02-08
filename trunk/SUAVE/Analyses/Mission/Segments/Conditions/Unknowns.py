# Unknowns.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Conditions import Conditions

# ----------------------------------------------------------------------
#  Unknowns
# ----------------------------------------------------------------------

class Unknowns(Conditions):
    
    def __defaults__(self):
        self.tag = 'unknowns'