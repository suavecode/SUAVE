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
from SUAVE.Core import ContainerOrdered, Container
from SUAVE.Components.Wings.Segment import Segment

import numpy as np

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
        self.PGM_characteristics    = ['spans.projected','chords.root','non_dimensional_origin[0][0]','non_dimensional_origin[0][1]','non_dimensional_origin[0][2]']
        self.PGM_char_min_bounds    = [0,0,0,-np.inf,-np.inf]   
        self.PGM_char_max_bounds    = [np.inf,np.inf,np.inf,np.inf,np.inf]        
        
        
        
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
    
    
    def append(self,val):
        """Appends the value to the containers
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            self
    
            Outputs:
            N/A
    
            Properties Used:
            N/A
            """          
        
        
        # See if the component exists, if it does modify the name
        keys = self.keys()
        if val.tag in keys:
            string_of_keys = "".join(val.keys())
            n_comps = string_of_keys.count(val.tag)
            val.tag = val.tag + str(n_comps+1)    
            
        ContainerOrdered.append(self,val)


# ----------------------------------------------------------------------
#   Unit Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError('test failed, not implemented')