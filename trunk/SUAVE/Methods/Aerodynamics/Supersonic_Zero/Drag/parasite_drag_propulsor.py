## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# parasite_drag_propulsor.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Nov 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import compressible_turbulent_flat_plate
from SUAVE.Core import Data

import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Propulsors
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def parasite_drag_propulsor(state,settings,geometry):
    """Computes the parasite drag due to the propulsor

    Assumptions:
    Basic fit

    Source:
    Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach) (subsonic)
    http://adg.stanford.edu/aa241/drag/BODYFORMFACTOR.HTML (supersonic)

    Inputs:
    state.conditions.freestream.
      mach_number                                [Unitless]
      temperature                                [K]
      reynolds_number                            [Unitless]
    geometry.      
      nacelle_diameter                           [m^2]
      areas.wetted                               [m^2]
      engine_length                              [m]
    state.conditions.aerodynamics.drag_breakdown.
      compressible.main_wing.divergence_mach     [Unitless]

    Outputs:
    propulsor_parasite_drag                      [Unitless]

    Properties Used:
    N/A
    """
    
    # unpack inputs
    
    conditions    = state.conditions
    configuration = settings
    propulsor     = geometry
        
    freestream = conditions.freestream
    
    Sref        = propulsor.nacelle_diameter**2 / 4 * np.pi
    Swet        = propulsor.areas.wetted
    
    l_prop  = propulsor.engine_length
    d_prop  = propulsor.nacelle_diameter
    
    # conditions
    freestream = conditions.freestream
    Mc = freestream.mach_number
    Tc = freestream.temperature    
    re = freestream.reynolds_number

    # reynolds number
    Re_prop = re*l_prop
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)


    k_prop = np.array([[0.0]]*len(Mc))
    # assume that the drag divergence mach number of the propulsor matches the main wing
    Mdiv = state.conditions.aerodynamics.drag_breakdown.compressible.main_wing.divergence_mach
    
    # form factor according to Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach)
    k_prop_sub = 1. + 0.35 / (float(l_prop)/float(d_prop)) 
    
    # for supersonic flow (http://adg.stanford.edu/aa241/drag/BODYFORMFACTOR.HTML)
    k_prop_sup = 1.
    
    sb_mask = (Mc <= Mdiv)
    tn_mask = ((Mc > Mdiv) & (Mc < 1.05))
    sp_mask = (Mc >= 1.05)
    
    k_prop[sb_mask] = k_prop_sub
    # basic interpolation for transonic
    k_prop[tn_mask] = (k_prop_sup-k_prop_sub)*(Mc[tn_mask]-Mdiv[tn_mask])/(1.05-Mdiv[tn_mask]) + k_prop_sub
    k_prop[sp_mask] = k_prop_sup
    
    # --------------------------------------------------------
    # find the final result    
    propulsor_parasite_drag = k_prop * cf_prop * Swet / Sref  
    # --------------------------------------------------------
    
    # dump data to conditions
    propulsor_result = Data(
        wetted_area               = Swet    , 
        reference_area            = Sref    , 
        parasite_drag_coefficient = propulsor_parasite_drag ,
        skin_friction_coefficient = cf_prop ,
        compressibility_factor    = k_comp  ,
        reynolds_factor           = k_reyn  , 
        form_factor               = k_prop  ,
    )
    state.conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag] = propulsor_result    
    
    return propulsor_parasite_drag