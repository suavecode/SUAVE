## @defgroup Components-Energy-Nacelles Nacelles
# Nacelle.py
# 
# Created:  Jul 2021, M. Clarke 
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component  

# ------------------------------------------------------------
#  Nacalle
# ------------------------------------------------------------

## @ingroup components-nacelles
class Nacelle(Energy_Component):
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
 
        # For VSP
        self.vsp_data                = Data()
        self.vsp_data.xsec_surf_id   = ''    # There is only one XSecSurf in each VSP geom.
        self.vsp_data.xsec_num       = None  # Number if XSecs in fuselage geom.
         