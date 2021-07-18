## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# parasite_drag_pylon.py
# 
# Created:  Jan 2014, T. Orra
# Modified: Jan 2016, E. Botero   

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# Suave imports
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Computes the pyloan parasite drag
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def parasite_drag_pylon(state,settings,geometry):
    """Computes the parasite drag due to pylons as a proportion of the propulsor drag

    Assumptions:
    Basic fit

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].
      form_factor                                                   [Unitless]
      compressibility_factor                                        [Unitless]
      skin_friction_coefficient                                     [Unitless]
      wetted_area                                                   [m^2]
      parasite_drag_coefficient                                     [Unitless]
      reynolds_number                                               [Unitless]
    geometry.reference_area                                         [m^2]
    geometry.nacelle.
      diameter                                                      [m]
      number_of_nacelles                                            [Unitless]

    Outputs:
    propulsor_parasite_drag                                         [Unitless]

    Properties Used:
    N/A
    """
    # unpack
    
    conditions = state.conditions
    configuration = settings
    
    pylon_factor        =  0.20 # 20% of propulsor drag
    n_nacelles          =  len(geometry.nacelle)  # number of propulsive system in vehicle (NOT # of ENGINES)
    pylon_parasite_drag = 0.00
    pylon_wetted_area   = 0.00
    pylon_cf            = 0.00
    pylon_compr_fact    = 0.00
    pylon_rey_fact      = 0.00
    pylon_FF            = 0.00

    # Estimating pylon drag
    for nacelle in geometry.nacelles:
        ref_area = nacelle.diameter**2 / 4 * np.pi
        pylon_parasite_drag += pylon_factor *  conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].parasite_drag_coefficient* (ref_area/geometry.reference_area * propulsor.number_of_engines)
        pylon_wetted_area   += pylon_factor *  conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].wetted_area * propulsor.number_of_nacelles
        pylon_cf            += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].skin_friction_coefficient
        pylon_compr_fact    += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].compressibility_factor
        pylon_rey_fact      += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].reynolds_factor
        pylon_FF            += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].form_factor
        
    pylon_cf            /= n_nacelles       
    pylon_compr_fact    /= n_nacelles
    pylon_rey_fact      /= n_nacelles
    pylon_FF            /= n_nacelles
    
    # dump data to conditions
    pylon_result = Data(
        wetted_area               = pylon_wetted_area   ,
        reference_area            = geometry.reference_area   ,
        parasite_drag_coefficient = pylon_parasite_drag ,
        skin_friction_coefficient = pylon_cf  ,
        compressibility_factor    = pylon_compr_fact   ,
        reynolds_factor           = pylon_rey_fact   ,
        form_factor               = pylon_FF   ,
    )
    conditions.aerodynamics.drag_breakdown.parasite['pylon'] = pylon_result 
 
    return pylon_parasite_drag