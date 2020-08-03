## @ingroup components-fuselages
# Fuselage.py
# 
# Created:  Mar 2014, T. Lukacyzk
# Modified: Sep 2016, E. Botero
#           Jun 2017, M. Clarke
#           Aug 2018, T St. Francis
#           Oct 2018, T. MacDonald
#           Dec 2018, T. MacDonald 
#           Mar 2020, M. Clarke
#           May 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Container, ContainerOrdered
from SUAVE.Components import Physical_Component, Lofted_Body
from SUAVE.Components.Fuselages.Segment import Segment_Container
import numpy as np

# ------------------------------------------------------------
#  Fuselage
# ------------------------------------------------------------

## @ingroup components-fuselages
class Fuselage(Lofted_Body):
    """ This is a standard fuselage for a tube and wing aircraft.
    
    Assumptions:
    Conventional fuselage
    
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
        
        self.tag = 'fuselage'
        self.origin             = [[0.0,0.0,0.0]]
        self.aerodynamic_center = [0.0,0.0,0.0]
        self.max_per_vehicle = 4
        
        self.number_coach_seats = 0.0
        self.seats_abreast      = 0.0
        self.seat_pitch         = 0.0

        self.areas = Data()
        self.areas.front_projected = 0.0
        self.areas.side_projected  = 0.0
        self.areas.wetted          = 0.0
        
        self.effective_diameter = 0.0
        self.width              = 0.0
        
        self.heights = Data()
        self.heights.maximum                        = 0.0
        self.heights.at_quarter_length              = 0.0
        self.heights.at_three_quarters_length       = 0.0
        self.heights.at_wing_root_quarter_chord     = 0.0
        self.heights.at_vertical_root_quarter_chord = 0.0
        
        self.x_rotation  = 0.0
        self.y_rotation  = 0.0
        self.z_rotation  = 0.0
        
        self.lengths = Data()
        self.lengths.nose       = 0.0
        self.lengths.tail       = 0.0
        self.lengths.total      = 0.0
        self.lengths.cabin      = 0.0
        self.lengths.fore_space = 0.0
        self.lengths.aft_space  = 0.0
            
        self.fineness = Data()
        self.fineness.nose = 0.0
        self.fineness.tail = 0.0
        
        self.differential_pressure = 0.0
        
        # for BWB 
        self.aft_centerbody_area  = 0.0
        self.aft_centerbody_taper = 0.0
        self.cabin_area           = 0.0
        
        self.non_dimensional_origin = [[0.0,0.0,0.0]]
        self.generative_design_minimum         = 0
        self.generative_design_max_per_vehicle = 3
        self.generative_design_characteristics = ['lengths.total','heights.maximum','width','fineness.nose','fineness.tail']
        self.generative_design_char_min_bounds = [1.,0.001,0.001,0.001,0.001]   
        self.generative_design_char_max_bounds = [np.inf,np.inf,np.inf,np.inf,np.inf]    

        self.Fuel_Tanks = Physical_Component.Container()

        # For VSP
        self.vsp_data                = Data()
        self.vsp_data.xsec_surf_id   = ''    # There is only one XSecSurf in each VSP geom.
        self.vsp_data.xsec_num       = None  # Number if XSecs in fuselage geom.
        
        self.Segments                = ContainerOrdered()
        
    def append_segment(self,segment):
        """ Adds a segment to the fuselage. 
    
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
    
    def append_fuel_tank(self,fuel_tank):
        """ Adds a fuel tank to the fuselage 
    
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
        if not isinstance(fuel_tank,Data):
            raise Exception('input component must be of type Data()')
    
        # Store data
        self.Fuel_Tanks.append(fuel_tank)

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
        
        return [Fuselage]

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Fuselage.Container = Container
