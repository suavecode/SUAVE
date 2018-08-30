## @ingroup Components-Wings
# Main_Wing.py
#
# Created:  Feb 2014, T. Lukacyzk, T. Orra
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave imports
from .Wing import Wing

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

## @ingroup Components-Wings
class Main_Wing(Wing):
    """This class is used to define main wings SUAVE

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
        """This sets the default for main wings in SUAVE.
    
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
        pass


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')