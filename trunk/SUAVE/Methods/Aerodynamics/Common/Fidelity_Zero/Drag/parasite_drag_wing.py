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
    
    wing = geometry
    Sref = wing.areas.reference
    
    # wing
    mac_w        = wing.chords.mean_aerodynamic
    t_c_w        = wing.thickness_to_chord
    sweep_w      = wing.sweeps.quarter_chord
    arw_w        = wing.aspect_ratio
    span_w       = wing.spans.projected
    S_exposed_w  = wing.areas.exposed # TODO: calculate by fuselage diameter (in Fidelity_Zero.initialize())
    S_affected_w = wing.areas.affected  
    xtu          = wing.transition_x_upper
    xtl          = wing.transition_x_lower
    
    # compute wetted area 
    if wing.has_key('areas'):
        if wing.areas.has_key('wetted'):
            Swet = wing.areas.wetted
        else:
            Swet = 1. * (1.0+ 0.2*t_c_w) * S_exposed_w
            wing.areas.wetted = Swet
    else:
        Swet = 1. * (1.0+ 0.2*t_c_w) * S_exposed_w
        wing.areas.wetted = Swet        
    
    # conditions
    Mc  = freestream.mach_number
    Tc  = freestream.temperature    
    re  = freestream.reynolds_number

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