## @defgroup Components-Energy-Nacelles Nacelles
# Nacelle.py
# 
# Created:  Jul 2021, M. Clarke 
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core import Data, Container, ContainerOrdered
from SUAVE.Components import Physical_Component, Lofted_Body
from SUAVE.Components.Nacelles.Segment import Segment_Container
import numpy as np


# ------------------------------------------------------------
#  Nacalle
# ------------------------------------------------------------

## @ingroup components-nacelles
class Nacelle(Lofted_Body):
    """ This is a nacelle for a generic aircraft.
    
    Assumptions:
    None
    
    Source:
    N/A
    """
    
    def __defaults__(self):
        """ This sets the default values for the component to function.
        
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
        
        self.tag                   = 'nacelle'
        self.origin                = [[0.0,0.0,0.0]]
        self.aerodynamic_center    = [0.0,0.0,0.0] 
         
        self.areas                 = Data()
        self.areas.front_projected = 0.0
        self.areas.side_projected  = 0.0
        self.areas.wetted          = 0.0
         
        self.diameter              = 0.0 
        self.lengths               = 0.0  
        
        self.x_rotation            = 0.0
        self.y_rotation            = 0.0
        self.z_rotation            = 0.0 
        
        self.flow_through          = True 
        self.differential_pressure = 0.0  
        self.naca_4_series_airfoil = False 
 
        # For VSP
        self.vsp_data                = Data()
        self.vsp_data.xsec_surf_id   = ''    # There is only one XSecSurf in each VSP geom.
        self.vsp_data.xsec_num       = None  # Number if XSecs in nacelle geom.
        
        self.Segments                = ContainerOrdered()
        
    def append_segment(self,segment):
        """ Adds a segment to the nacelle. 
    
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

        # Assert database type
        if not isinstance(segment,Data):
            raise Exception('input component must be of type Data()')

        # Store data
        self.Segments.append(segment)

        return
         
         

class Container(Physical_Component.Container):
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
        
        return [Nacelle]

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Nacelle.Container = Container         