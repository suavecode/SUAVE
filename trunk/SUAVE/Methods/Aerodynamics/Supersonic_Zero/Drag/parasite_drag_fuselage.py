## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# parasite_drag_fuselage.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Nov 2016, T. MacDonald
#           Feb 2019, T. MacDonald
#           Jan 2020, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import compressible_turbulent_flat_plate
from SUAVE.Core import Data
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender

import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Fuselage
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def parasite_drag_fuselage(state,settings,geometry):
    """Computes the parasite drag due to the fuselage

    Assumptions:
    Basic fit

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/aircraftdesign.html (Stanford AA241 A/B Course Notes)

    Inputs:
    state.conditions.freestream.
      mach_number                                [Unitless]
      temperature                                [K]
      reynolds_number                            [Unitless]
    settings.fuselage_parasite_drag_form_factor  [Unitless]
    geometry.fuselage.       
      areas.front_projected                      [m^2]
      areas.wetted                               [m^2]
      lengths.total                              [m]
      effective_diameter                         [m]

    Outputs:
    fuselage_parasite_drag                       [Unitless]

    Properties Used:
    N/A
    """

    # unpack inputs
    configuration = settings
    form_factor   = configuration.fuselage_parasite_drag_form_factor
    low_cutoff    = configuration.fuselage_parasite_drag_begin_blend_mach
    high_cutoff   = configuration.fuselage_parasite_drag_end_blend_mach    
    fuselage      = geometry
    
    freestream  = state.conditions.freestream
    Sref        = fuselage.areas.front_projected
    Swet        = fuselage.areas.wetted
    
    l_fus  = fuselage.lengths.total
    d_fus  = fuselage.effective_diameter
    
    # conditions
    Mc  = freestream.mach_number
    Tc  = freestream.temperature    
    re  = freestream.reynolds_number

    # reynolds number
    Re_fus = re*(l_fus)
    
    # skin friction coefficient
    cf_fus, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_fus,Mc,Tc)
    
    # form factor for cylindrical bodies
    d_d = float(d_fus)/float(l_fus)
    
    D_low = np.array([[0.0]] * len(Mc))
    a_low = np.array([[0.0]] * len(Mc))
    du_max_u_low = np.array([[0.0]] * len(Mc))
    
    D_high = np.array([[0.0]] * len(Mc))
    a_high = np.array([[0.0]] * len(Mc))
    du_max_u_high = np.array([[0.0]] * len(Mc))    
    
    k_fus = np.array([[0.0]] * len(Mc))
    
    low_inds  = Mc < high_cutoff
    high_inds = Mc > low_cutoff
    
    D_low[low_inds] = np.sqrt(1 - (1-Mc[low_inds]**2) * d_d**2)
    a_low[low_inds] = 2 * (1-Mc[low_inds]**2) * (d_d**2) *(np.arctanh(D_low[low_inds])-D_low[low_inds]) / (D_low[low_inds]**3)
    du_max_u_low[low_inds] = a_low[low_inds] / ( (2-a_low[low_inds]) * (1-Mc[low_inds]**2)**0.5 )
    
    D_high[high_inds] = np.sqrt(1 - d_d**2)
    a_high[high_inds] = 2  * (d_d**2) *(np.arctanh(D_high[high_inds])-D_high[high_inds]) / (D_high[high_inds]**3)
    du_max_u_high[high_inds] = a_high[high_inds] / ( (2-a_high[high_inds]) )
    
    spline = Cubic_Spline_Blender(low_cutoff,high_cutoff)
    h00 = lambda M:spline.compute(M)
    
    du_max_u = du_max_u_low*(h00(Mc)) + du_max_u_high*(1-h00(Mc))    
    
    k_fus = (1 + form_factor*du_max_u)**2

    fuselage_parasite_drag = k_fus * cf_fus * Swet / Sref  
    
    # dump data to conditions
    fuselage_result = Data(
        wetted_area               = Swet   , 
        reference_area            = Sref   , 
        parasite_drag_coefficient = fuselage_parasite_drag ,
        skin_friction_coefficient = cf_fus ,
        compressibility_factor    = k_comp ,
        reynolds_factor           = k_reyn , 
        form_factor               = k_fus  ,
    )
    try:
        state.conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag] = fuselage_result
    except:
        print("Drag Polar Mode fuse parasite")
    
    return fuselage_parasite_drag