## @ingroup Components-Fuselages
# Segment.py
# 
# Created: Oct 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data
from SUAVE.Components import Lofted_Body
from SUAVE.Components import Component, Lofted_Body, Mass_Properties
# ------------------------------------------------------------ 
#  Wing Segments
# ------------------------------------------------------------

## @ingroup Components-Fuselages
class Segment(Lofted_Body.Segment):
    def __defaults__(self):
        """This sets the default for wing segments in SUAVE.

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
        self.tag                = 'segment'		
        self.origin	        = [0., 0. ,0.]		
        self.percent_x_location	= 0.0		
        self.percent_z_location	= 0.0
        self.height		= 0.0		
        self.width		= 0.0		
        self.length	        = 0.0		
        self.effective_diameter	= 0.0	
    
## @ingroup Components-Fuselages
class SegmentContainer(Lofted_Body.Segment.Container):
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
    
    pass