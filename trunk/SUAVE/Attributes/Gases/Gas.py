## @ingroup Attributes-Gases
# Gas.py: 

# Created:  Mar 2014, SUAVE Team
# Modified: Jan 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Gas Data Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Gases
class Gas(Data):
    """Base class for gases

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
        self.molecular_mass  = 0.0    
        self.gas_specific_constant              = 0.0              
        self.composition = Data()
        self.composition.gas = 1.0
