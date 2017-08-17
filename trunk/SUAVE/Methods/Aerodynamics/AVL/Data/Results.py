## @ingroup Methods-Aerodynamics-AVL-Data
# Results.py
# 
# Created:  Jan 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data

# ------------------------------------------------------------
#   Wing
# ------------------------------------------------------------

## @ingroup Methods-Aerodynamics-AVL-Data
class Results(Data):
    """ A data class defining aerodynamics and stability results 

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        None

    Outputs:
        None

    Properties Used:
        N/A
    """    
    
    def __defaults__(self):
        """ Defining data structure  and defaults for aerodynamics and stabilty results 
    
        Assumptions:
            None
            
        Source:
            None
    
        Inputs:
            None
    
        Outputs:
            None
    
        Properties Used:
            N/A
        """ 
        self.aerodynamics = Data()
        self.stability    = Data()
        
        self.stability.alpha_derivatives = Data()
        self.stability.beta_derivatives  = Data()
        