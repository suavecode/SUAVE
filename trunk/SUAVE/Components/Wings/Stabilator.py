## @ingroup Components-Wings
# Stabilator.py
#
# Created:  Feb 2014, T. Lukacyzk, T. Orra
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from .Wing import Wing

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

## @ingroup Components-Wings
class Stabilator(Wing):
    """ This class is used to define stabilators in SUAVE
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """ 

    def __defaults__(self):
        """This sets the default for stabilators in SUAVE.
        
        see Control_Surface().__defaults__ and Wing().__defaults__ for an explanation of attributes
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """ 
        self.tag = 'stabilator'
        
        #describe control surface-like behavior
        self.sign_duplicate        = 1.0
        self.hinge_fraction        = 0.25
        self.deflection            = 0.0      


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')