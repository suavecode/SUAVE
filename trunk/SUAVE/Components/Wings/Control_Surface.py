## @ingroup Components-Wings
# Control_Surface.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Jun 2017, M. Clarke
#           Aug 2019, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Core import Data , Units
from SUAVE.Components import Component
from SUAVE.Components import Lofted_Body
import numpy as np 
    
# ------------------------------------------------------------
#  Control Surfaces
# ------------------------------------------------------------

## @ingroup Components-Wings
class Control_Surface(Component):
    def __defaults__(self):
        """This sets the default values of control surfaces defined in SUAVE. 
        sign_duplicate: 1.0 or -1.0 - the sign of the duplicate control on the mirror wing.
        Use 1.0 for a mirrored control surface, like an elevator. Use -1.0 for an aileron.
        The span fraction is given by the array shown below:  
        [abs. % span location at beginning of crtl surf, abs. % span location at end  of crtl surf]

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

        self.tag                   = 'control_surface' 
        self.function              = 'unspecified' # This is a string argument which defines the function of a control surface. Options are 'elevator','rudder','flap', 'aileron' and 'slat'
        self.span                  = 0.0
        self.span_fraction_start   = 0.0
        self.span_fraction_end     = 0.0
        self.chord_fraction        = 0.0  
        self.hinge_fraction        = 1.0
        self.deflection      = 0.0  
        self.configuration_type    = 'single_slotted'
        self.gain                  = 1.0
 
def append_ctrl_surf_to_wing_segments(wing):
    '''This function takes the control surfaces defined on a wing and appends them to wing segments
    Conditional statements are used to determine where the control surface bounds are in relation 
    to the wing segments. For example, If a control surface extends beyond a wing segment, the bounds 
    on the control surface span fraction are set to be the bounds of the wing section'''
    w_cs  = wing.control_surfaces  
    w_seg = wing.Segments
    
    # loop though the segments on the wing and clear existing control surfaces 
    for i , seg in enumerate(w_seg):    
        w_seg[i].control_surfaces = Data() 
          
    # loop throught the control surfaces on the wing 
    for cs in w_cs :
        sf    = np.zeros(2) # set a temporary data structure to store the span fraction bounds
        sf[0] = w_cs[cs].span_fraction_start
        sf[1] = w_cs[cs].span_fraction_end
        
        # loop though the segments on the wing
        for i , seg in enumerate(w_seg):
            
            append_CS = False
            s_sf = np.zeros(2) 
            if i == 0: # the first segment (root) cannot have any control surfaces 
                pass
            else: # the following block determines where the bounds of the control surface are in relation to the segment breaks
                # Case 1 
                if (sf[0] < w_seg[i-1].percent_span_location) and (sf[1] < w_seg[i].percent_span_location) and (sf[1] > w_seg[i-1].percent_span_location) :
                    s_sf = np.array([w_seg[i-1].percent_span_location,sf[1]])   
                    append_CS = True 
                
                # Case 2
                elif (sf[0] < w_seg[i-1].percent_span_location) and (sf[1] == w_seg[i].percent_span_location):
                    s_sf = np.array([w_seg[i-1].percent_span_location,w_seg[i].percent_span_location])       
                    append_CS = True 
                    
                # Case 3   
                elif (sf[0] < w_seg[i-1].percent_span_location) and (sf[1] > w_seg[i].percent_span_location):
                    s_sf = np.array([w_seg[i-1].percent_span_location,w_seg[i].percent_span_location])       
                    append_CS = True                 
                
                # Case 4 
                elif (sf[0] == w_seg[i-1].percent_span_location) and (sf[1] < w_seg[i].percent_span_location):
                    s_sf = np.array([w_seg[i-1].percent_span_location,sf[1]])   
                    append_CS = True 
                   
                # Case 5 
                elif (sf[0] == w_seg[i-1].percent_span_location) and (sf[1] == w_seg[i].percent_span_location): 
                    s_sf = np.array([w_seg[i-1].percent_span_location,w_seg[i].percent_span_location])       
                    append_CS = True
                    
                # Case 6 
                elif (sf[0] > w_seg[i-1].percent_span_location) and (sf[1] < w_seg[i].percent_span_location):
                    s_sf = np.array([sf[0],sf[1]])
                    append_CS = True
                    
                # Case 7 
                elif (sf[0] > w_seg[i-1].percent_span_location) and (sf[1] == w_seg[i].percent_span_location) :
                    s_sf = np.array([sf[0],w_seg[1].percent_span_location]) 
                    append_CS = True                    
                    
                # Case 8
                elif (sf[0] > w_seg[i-1].percent_span_location) and (sf[1] > w_seg[i].percent_span_location) and (sf[0] < w_seg[i].percent_span_location):
                    s_sf = np.array([sf[0],w_seg[1].percent_span_location]) 
                    append_CS = True
                    
                else: 
                    append_CS = False
                
                if append_CS == True:
                    # initialize the data structure for control surfaces , store results, and append to the correct segment 
                    control_surface = Control_Surface() 
                    control_surface.tag                   = w_cs[cs].tag
                    control_surface.function              = w_cs[cs].function
                    control_surface.span_fraction_start   = s_sf[0] 
                    control_surface.span_fraction_end     = s_sf[1]         
                    control_surface.chord_fraction        = w_cs[cs].chord_fraction
                    control_surface.hinge_fraction        = w_cs[cs].hinge_fraction
                    control_surface.deflection            = w_cs[cs].deflection / Units.deg
                    w_seg[i].append_control_surface(control_surface)        
                
    # returns an updated wing with control surfaces appended onto the wing segments                  
    return wing  
