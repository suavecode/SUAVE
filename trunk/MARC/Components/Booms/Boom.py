## @ingroup components-booms
# Boom.py
# 
# Created:  Mar 2023, M Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
 
from MARC.Core import Data, Container, ContainerOrdered
from MARC.Components import Physical_Component, Lofted_Body 
import numpy as np

# ------------------------------------------------------------
#  Boom
# ------------------------------------------------------------

## @ingroup components-booms
class Boom(Lofted_Body):
    """ This is a standard boom for a rotor.
    
    Assumptions:
    Conventional boom
    
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
        
        self.tag                                    = 'boom'
        self.origin                                 = [[0.0,0.0,0.0]]
        self.aerodynamic_center                     = [0.0,0.0,0.0]  
                 
        self.areas                                  = Data()
        self.areas.front_projected                  = 0.0
        self.areas.side_projected                   = 0.0
        self.areas.wetted                           = 0.0
                         
        self.effective_diameter                     = 0.0
        self.width                                  = 0.0 
                         
        self.heights                                = Data()
        self.heights.maximum                        = 0.0
        self.heights.at_quarter_length              = 0.0
        self.heights.at_three_quarters_length       = 0.0
        self.heights.at_wing_root_quarter_chord     = 0.0
        self.heights.at_vertical_root_quarter_chord = 0.0
        
        self.x_rotation                             = 0.0
        self.y_rotation                             = 0.0
        self.z_rotation                             = 0.0
             
        self.lengths                                = Data()
        self.lengths.nose                           = 0.0
        self.lengths.tail                           = 0.0
        self.lengths.total                          = 0.0
        self.lengths.cabin                          = 0.0
        self.lengths.fore_space                     = 0.0
        self.lengths.aft_space                      = 0.0
                 
        self.fineness                               = Data()
        self.fineness.nose                          = 0.0
        self.fineness.tail                          = 0.0
             
        self.differential_pressure                  = 0.0 
             
        self.non_dimensional_origin                 = [[0.0,0.0,0.0]]
        self.generative_design_minimum              = 0
        self.generative_design_max_per_vehicle      = 3
        self.generative_design_characteristics      = ['lengths.total','heights.maximum','width','fineness.nose','fineness.tail']
        self.generative_design_char_min_bounds      = [1.,0.001,0.001,0.001,0.001]   
        self.generative_design_char_max_bounds      = [np.inf,np.inf,np.inf,np.inf,np.inf]    
     
        # For VSP     
        self.vsp_data                               = Data()
        self.vsp_data.xsec_surf_id                  = ''    # There is only one XSecSurf in each VSP geom.
        self.vsp_data.xsec_num                      = None  # Number if XSecs in rotor_boom geom.
                        
        self.Segments                               = ContainerOrdered()
        
    def append_segment(self,segment):
        """ Adds a segment to the rotor_boom. 
    
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
        
        return [Boom]

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Boom.Container = Container
