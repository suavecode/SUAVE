## @ingroup Components-Wings
# Segment.py
# 
# Created:  Sep 2016, E. Botero
# Modified: Jul 2017, M. Clarke
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data, ContainerOrdered
from SUAVE.Components import Component, Lofted_Body
import numpy as np

# ------------------------------------------------------------ 
#  Wing Segments
# ------------------------------------------------------------

## @ingroup Components-Wings
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
        self.tag = 'segment'
        self.percent_span_location = 0.0
        self.twist                 = 0.0
        self.root_chord_percent    = 0.0
        self.dihedral_outboard     = 0.0
        self.thickness_to_chord    = 0.0
        self.sweeps                = Data()
        self.sweeps.quarter_chord  = 0.0
        self.sweeps.leading_edge   = None
        self.areas                 = Data()
        self.areas.reference       = 0.0
        self.areas.exposed         = 0.0
        self.areas.wetted          = 0.0
        self.Airfoil               = SUAVE.Core.ContainerOrdered()
        self.generative_design_minimum           = 2
        self.generative_design_max_per_vehicle   = 10
        self.generative_design_special_parent    = SUAVE.Components.Wings.Main_Wing
        self.generative_design_characteristics   = ['percent_span_location','twist','root_chord_percent','dihedral_outboard','sweeps.quarter_chord','thickness_to_chord']
        self.generative_design_char_min_bounds   = [0.,-np.pi/3,0.,-.1,-1.2,0.0001]   
        self.generative_design_char_max_bounds   = [1.,np.pi/3,np.inf,1.,1.2,0.5]        
        
        
    def append_airfoil(self,airfoil):
        """ Adds an airfoil to the segment

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
        # assert database type
        if not isinstance(airfoil,Data):
            raise Exception('input component must be of type Data()')

        # store data
        self.Airfoil.append(airfoil)

        
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
        
        return []
