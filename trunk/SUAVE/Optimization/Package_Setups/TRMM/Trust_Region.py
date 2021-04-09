## @ingroup Optimization-Package_Setups-TRMM
# Trust_Region.py
#
# Created:  Apr 2017, T. MacDonald
# Modified: Jun 2017, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Trust Region Class
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups-TRMM
class Trust_Region(Data):
    """A trust region class
    
    Assumptions:
    None
    
    Source:
    None
    """    
    
    def __defaults__(self):
        """This sets the default values.
    
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
        
        self.initial_size       = 0.05
        self.size               = 0.05
        self.minimum_size       = 1e-15
        self.contract_threshold = 0.25
        self.expand_threshold   = 0.75
        self.contraction_factor = 0.25
        self.expansion_factor   = 1.5
        
        
    def evaluate_function(self,f,gviol):
        """ Evaluates the function
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            self    
            f         []
            gviol     []
    
            Outputs:
            phi       []
    
            Properties Used:
            None
            """            
        phi = f + gviol**2
        return phi        