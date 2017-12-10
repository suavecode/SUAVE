## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# parasite_drag_wing.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero       

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# local imports
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import compressible_mixed_flat_plate

# suave imports
from SUAVE.Core import Data

# package imports
import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Wing
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def parasite_drag_wing(state,settings,geometry):
    """Computes the parasite drag due to wings

    Assumptions:
    Basic fit

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    settings.wing_parasite_drag_form_factor      [Unitless]
    state.conditions.freestream.
      mach_number                                [Unitless]
      temperature                                [K]
      reynolds_number                            [Unitless]
    geometry.
      areas.reference                            [m^2]
      chords.mean_aerodynamic                    [m]
      thickness_to_chord                         [Unitless]
      sweeps.quarter_chord                       [radians]
      aspect_ratio                               [Unitless]
      spans.projected                            [m]
      areas.exposed                              [m^2]
      areas.affected                             [m^2]
      areas.wetted                               [m^2]
      transition_x_upper                         [Unitless]
      transition_x_lower                         [Unitless]
      
      
    Outputs:
    wing_parasite_drag                           [Unitless]

    Properties Used:
    N/A
    """
    
    # unpack inputs
    C = settings.wing_parasite_drag_form_factor
    freestream = state.conditions.freestream
    
    # conditions
    Mc  = freestream.mach_number
    Tc  = freestream.temperature    
    re  = freestream.reynolds_number     
    
    wing = geometry
    wing_parasite_drag = 0.0
    
    # Unpack wing
    Sref = wing.areas.reference
    
    if len(wing.Segments.keys())>0:# if wing has segments, compute and sum parasite drag of each segment
        symm                 = wing.symmetric
        semispan             = wing.spans.projected*0.5 * (2 - symm)
        root_chord           = wing.chords.root
        num_segments         = len(wing.Segments.keys())     
        
        total_wetted_area            = 0
        total_segment_parasite_drag  = 0 
        total_segment_k_w            = 0 
        total_segment_cf_w_u         = 0
        total_segment_cf_w_l         = 0 
        total_segment_k_comp_u       = 0
        total_k_reyn_l               = 0      
        for i_segs in xrange(num_segments): 
            
            if i_segs == num_segments-1:
                continue 
            else:   
                t_c_seg       = wing.thickness_to_chord
                span_seg      = semispan*(wing.Segments[i_segs+1].percent_span_location - wing.Segments[i_segs].percent_span_location )            
                sweep_seg     = wing.Segments[i_segs].sweeps.quarter_chord    
                xtu           = wing.transition_x_upper
                xtl           = wing.transition_x_lower                  
                #if i_segs == 0:
                    #fuse_width    = 2.88/2 # 2.88/2 for Concorde and  3.76/2 for B737
                    #chord_root   = root_chord*wing.Segments[i_segs].root_chord_percent
                    #chord_tip    = root_chord*wing.Segments[i_segs+1].root_chord_percent   
                    #wing_root     = chord_root - fuse_width*((chord_tip - chord_root)/span_seg)
                    #taper         = chord_tip/chord_root   
                    #mac_seg       = wing_root  * 2/3 * (( 1 + taper  + taper**2 )/( 1 + taper))  
                    #Sref_seg      = (span_seg-wing_root) *(wing_root+chord_tip)*0.5                    
                #else: 
                chord_root   = root_chord*wing.Segments[i_segs].root_chord_percent
                chord_tip    = root_chord*wing.Segments[i_segs+1].root_chord_percent
                taper         = chord_tip/chord_root   
                mac_seg       = chord_root * 2/3 * (( 1 + taper  + taper**2 )/( 1 + taper))
                Sref_seg      = span_seg *(chord_root+chord_tip)*0.5
                    

                # compute exposed area of segment
                S_exposed_seg  = Sref_seg*2.0 
                if wing.thickness_to_chord < 0.05:
                    Swet_seg = 2.003* S_exposed_seg
                else:
                    Swet_seg = (1.977 + 0.54*wing.thickness_to_chord )* S_exposed_seg     
                 
                # compute parasite drag coef., form factor, skin friction coef., compressibility factor and reynolds number for segments
                segment_parasite_drag , segment_k_w, segment_cf_w_u, segment_cf_w_l, segment_k_comp_u, k_reyn_l = compute_parasite_drag(re,mac_seg,Mc,Tc,xtu,xtl,sweep_seg,t_c_seg,Sref_seg,Swet_seg,C)    
                
                total_wetted_area            += Swet_seg
                total_segment_parasite_drag += segment_parasite_drag*Sref_seg   
                total_segment_k_w            += segment_k_w*Sref_seg 
                total_segment_cf_w_u         += segment_cf_w_u*Sref_seg 
                total_segment_cf_w_l         += segment_cf_w_l*Sref_seg 
                total_segment_k_comp_u       += segment_k_comp_u*Sref_seg 
                total_k_reyn_l               += k_reyn_l*Sref_seg  
                
            Swet              = total_wetted_area     
            wing.areas.wetted = total_wetted_area 
            wing_parasite_drag= total_segment_parasite_drag  / Sref
            k_w               = total_segment_k_w / Sref
            cf_w_u            = total_segment_cf_w_u  / Sref
            cf_w_l            = total_segment_cf_w_l / Sref
            k_comp_u          =  total_segment_k_comp_u  / Sref
            k_reyn_l          = total_k_reyn_l  / Sref

       
    else:   # if wing has no segments              
        # wing
        mac_w        = wing.chords.mean_aerodynamic
        t_c_w        = wing.thickness_to_chord
        sweep_w      = wing.sweeps.quarter_chord
        arw_w        = wing.aspect_ratio
        span_w       = wing.spans.projected
        S_exposed_w  = wing.areas.exposed # TODO: calculate by fuselage diameter (in Fidelity_Zero.initialize())
        xtu          = wing.transition_x_upper
        xtl          = wing.transition_x_lower
        
        # compute wetted area 
        if wing.has_key('areas'):
            if wing.areas.has_key('wetted'):
                Swet = wing.areas.wetted
            else:
                if wing.thickness_to_chord < 0.05:
                    Swet = 2.003* S_exposed_seg
                else:
                    Swet = (1.977 + 0.54*wing.thickness_to_chord )* S_exposed_seg
                wing.areas.wetted = Swet                            
        
        # compute parasite drag coef., form factor, skin friction coef., compressibility factor and reynolds number for wing
        wing_parasite_drag , k_w, cf_w_u, cf_w_l, k_comp_u, k_reyn_l = compute_parasite_drag(re,mac_w,Mc,Tc,xtu,xtl,sweep_w,t_c_w,Sref,Swet,C)             
    
    
    # dump data to conditions
    wing_result = Data(
        wetted_area               = Swet   , 
        reference_area            = Sref   , 
        parasite_drag_coefficient = wing_parasite_drag ,
        skin_friction_coefficient = (cf_w_u+cf_w_l)/2.   ,
        compressibility_factor    = k_comp_u ,
        reynolds_factor           = k_reyn_l , 
        form_factor               = k_w    ,
    )
    
    state.conditions.aerodynamics.drag_breakdown.parasite[wing.tag] = wing_result

    return wing_parasite_drag



def compute_parasite_drag(re,mac_w,Mc,Tc,xtu,xtl,sweep_w,t_c_w,Sref,Swet,C):
   
    # reynolds number
    Re_w = re*mac_w  
    
    # skin friction  coefficient, upper
    cf_w_u, k_comp_u, k_reyn_u = compressible_mixed_flat_plate(Re_w,Mc,Tc,xtu)
    
    # skin friction  coefficient, lower
    cf_w_l, k_comp_l, k_reyn_l = compressible_mixed_flat_plate(Re_w,Mc,Tc,xtl) 
    
    # correction for airfoils
    cos_sweep = np.cos(sweep_w)
    cos2      = cos_sweep*cos_sweep
    
    k_w = 1. + ( 2.* C * (t_c_w * cos2) ) / ( np.sqrt(1.- Mc*Mc * cos2) )  \
            + ( C*C * cos2 * t_c_w*t_c_w * (1. + 5.*(cos2)) ) \
            / (2.*(1.-(Mc*cos_sweep)**2.))             

    k_w[Mc >= 0.95] =  1. 

    # find the final result
    wing_parasite_drag = k_w * cf_w_u * Swet / Sref /2. + k_w * cf_w_l * Swet / Sref /2.


    return wing_parasite_drag , k_w, cf_w_u, cf_w_l, k_comp_u, k_reyn_l