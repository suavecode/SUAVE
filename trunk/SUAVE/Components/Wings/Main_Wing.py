## @ingroup Components-Wings
# Main_Wing.py
#
# Created:  Feb 2014, T. Lukacyzk, T. Orra
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from .Wing import Wing
from SUAVE.Core import ContainerOrdered
from SUAVE.Components.Wings.Segment import Segment

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
        self.tag             = 'main_wing'
        self.Segments         = Segment_Container()
        self.max_per_vehicle = 3
        self.PGM_compulsory  = True
        
        
        
## @ingroup Components-Wings
class Segment_Container(ContainerOrdered):
    """ Container for wing segment
    
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

    def get_children(self):
        """ Returns the components that can go inside
        
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
        
        return [Segment]


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')