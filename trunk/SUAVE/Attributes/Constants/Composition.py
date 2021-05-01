## @ingroup Attributes-Constants
# Composition.py
# 
# Created: Mar 2014,     J. Sinsay
# Modified: Jan, 2016,  M. Vegh



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# initialized constants
from .Constant import Constant

# exceptions/warnings
from warnings import warn

# ----------------------------------------------------------------------
#  Composition Constant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Constants
class Composition(Constant):
    """A container to store chemical compositions.
    
    Assumptions:
    None
    
    Source:
    None
    """
    def __defaults__(self):
        """This sets the default values (just a pass here).
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """            
        pass
    
    def __check__(self):
        """Checks that the composition values sum to 1.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        self.values       [-]
        self.Other        [-] (optional)
        """            
        # check that composition sums to 1.0
        total = 0.0
        for v in self.values():
            total += v
        other = 1.0 - total

        # set other if needed
        if other != 0.0: 
            if 'Other' in self:
                other += self.Other
            self.Other = other
            self.swap('Other',-1)
                
        # check for negative other
        if other < 0.0:
            warn('Composition adds to more than 1.0')
