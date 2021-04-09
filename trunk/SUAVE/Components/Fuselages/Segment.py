## @ingroup Components-Fuselages
# Segment.py
# 
# Created:  Sep 2016, E. Botero (for wings)
# Modified: Jul 2017, M. Clarke
#           Aug 2018, T. St Francis (for fuselages)
#           Jan 2019, T. MacDonald
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Components import Lofted_Body

# ------------------------------------------------------------ 
#  Fuselage Segments
# ------------------------------------------------------------

## @ingroup Components-Fuselages
class Segment(Lofted_Body.Segment):
    def __defaults__(self):
        """This sets the defaults for fuselage segments in SUAVE.

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
        self.percent_x_location = 0.0      # Percent location along fuselage length.
        self.percent_y_location = 0.0       
        self.percent_z_location = 0.0      # Vertical translation of segment. Percent of length.
        self.height             = 0.0
        self.width              = 0.0
        self.length             = 0.0    
        self.effective_diameter = 0.0
        self.vsp_data           = Data()
        self.vsp_data.xsec_id   = ''       # OpenVSP XSec ID such as 'MWLKSGTGDD'
        self.vsp_data.shape     = ''
        
## @ingroup Components-Wings
class Segment_Container(Lofted_Body.Segment.Container):
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
        
        return []