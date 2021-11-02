## @ingroup Methods-Geometry-Two_Dimensional.Planform
# populate_control_sections 
#
# Created:  Jan 2015, T. Momose
# Modified: Jan 2016, E. Botero 
#           Jan 2020 M. Clarke
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#  Append Control Surfaces to Wing Segments
# ----------------------------------------------------------------------  
from SUAVE.Core import Data , Units 
import numpy as np 
from SUAVE.Components.Wings.Control_Surfaces import Aileron , Elevator , Slat , Flap , Rudder 

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def populate_control_sections(wing):
    """This function takes the control surfaces defined on a wing and appends them to wing segments
    Conditional statements are used to determine where the control surface bounds are in relation 
    to the wing segments. For example, If a control surface extends beyond a wing segment, the bounds 
    on the control surface span fraction are set to be the bounds of the wing section
    
    Assumptions: 
       None

    Source:
       None

    Inputs: 
      wing.control_surface.tag                  [unitless]
      wing.control_surface.span_fraction_start  [unitless]
      wing.control_surface.span_fraction_end    [unitless]
      wing.control_surface.chord_fraction       [unitless]
      wing.control_surface.hinge_fraction       [unitless]
      wing.control_surface.deflection           [degrees]
      
    Outputs: 
      wing.control_surface.tag                  [unitless]    
      wing.control_surface.span_fraction_start  [unitless]
      wing.control_surface.span_fraction_end    [unitless]
      wing.control_surface.chord_fraction       [unitless]
      wing.control_surface.hinge_fraction       [unitless]
      wing.control_surface.deflection           [degrees]
        

    Properties Used:
    N/A
    """
    w_cs  = wing.control_surfaces  
    w_seg = wing.Segments
    
    # loop though the segments on the wing and clear existing control surfaces 
    for i , seg in enumerate(w_seg):    
        w_seg[i].control_surfaces = Data() 
          
    # loop throught the control surfaces on the wing 
    for cs in w_cs :
        sf    = np.zeros(2) # set a temporary data structure to store the span fraction bounds
        sf[0] = cs.span_fraction_start
        sf[1] = cs.span_fraction_end
        
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
                    s_sf = np.array([sf[0],w_seg[i].percent_span_location]) 
                    append_CS = True                    
                    
                # Case 8
                elif (sf[0] > w_seg[i-1].percent_span_location) and (sf[1] > w_seg[i].percent_span_location) and (sf[0] < w_seg[i].percent_span_location):
                    s_sf = np.array([sf[0],w_seg[i].percent_span_location]) 
                    append_CS = True
                    
                else: 
                    append_CS = False
                
                if append_CS == True:
                    # initialize the data structure for control surfaces , store results, and append to the correct segment 
                    control_surface = type(cs)() # control_surface takes came type as cs (Slat, Aileron, Data(for VLM), etc)                                             
                    control_surface.tag                   = cs.tag 
                    control_surface.span                  = cs.span*(s_sf[1]-s_sf[0])/(cs.span_fraction_end-cs.span_fraction_start)
                    control_surface.span_fraction_start   = s_sf[0] 
                    control_surface.span_fraction_end     = s_sf[1]         
                    control_surface.hinge_fraction        = cs.hinge_fraction
                    control_surface.chord_fraction        = cs.chord_fraction
                    control_surface.sign_duplicate        = cs.sign_duplicate
                    control_surface.deflection            = cs.deflection
                    control_surface.configuration_type    = cs.configuration_type
                    control_surface.gain                  = cs.gain                    
                    
                    #for calls from make_VLM_wings
                    if 'cs_type' in cs.keys():
                        control_surface.cs_type           = cs.cs_type
                    
                    w_seg[i].control_surfaces.append(control_surface)        
                
    # returns an updated wing with control surfaces appended onto the wing segments                  
    return wing  

