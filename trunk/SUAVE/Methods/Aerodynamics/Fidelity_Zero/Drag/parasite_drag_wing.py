# parasite_drag_wing.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero       

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# local imports
from compressible_mixed_flat_plate import compressible_mixed_flat_plate

# suave imports
from SUAVE.Analyses import Results

# package imports
import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Wing
# ----------------------------------------------------------------------

def parasite_drag_wing(state,settings,geometry):
    """ SUAVE.Methods.parasite_drag_wing(conditions,configuration,wing)
        computes the parastite drag associated with a wing 
        
        Inputs:
            conditions
            -freestream mach number
            -freestream density
            -freestream dynamic_viscosity
            -freestream temperature
            -freestream pressuve
            
            configuration
            -wing parasite drag form factor
            
            wing
            -S reference
            -mean aerodynamic chord
            -thickness to chord ratio
            -sweep
            -aspect ratio
            -span
            -S exposed
            -S affected
            -transition x
            
        Outputs:
            wing parasite drag coefficient with refernce area as the
            reference area of the input wing
        
        Assumptions:
        
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

    # find the final result
    wing_parasite_drag = k_w * cf_w_u * Swet / Sref /2. + k_w * cf_w_l * Swet / Sref /2.
    
    # dump data to conditions
    wing_result = Results(
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