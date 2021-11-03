## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# miscellaneous_drag_aircraft.py
# 
# Created:  Aug 2014, T. Macdonald
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data

import numpy as np

# ----------------------------------------------------------------------
#  Miscellaneous Drag Aircraft
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def miscellaneous_drag_aircraft(state,settings,geometry):
    """Computes the miscellaneous drag associated with an aircraft

    Assumptions:
    Basic fit

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/aircraftdesign.html (Stanford AA241 A/B Course Notes)

    Inputs:
    configuration.trim_drag_correction_factor  [Unitless]
    geometry.nacelle.diameter                  [m]
    geometry.reference_area                    [m^2]
    geometry.wings['main_wing'].aspect_ratio   [Unitless]
    state.conditions.freestream.mach_number    [Unitless] (actual values are not used)

    Outputs:
    total_miscellaneous_drag                   [Unitless]

    Properties Used:
    N/A
    """

    # unpack inputs
    configuration = settings
     
    vehicle_reference_area = geometry.reference_area
    ones_1col              = state.conditions.freestream.mach_number *0.+1
        
    conditions = state.conditions
        
    # ------------------------------------------------------------------
    #   Control surface gap drag
    # ------------------------------------------------------------------
    #f_gaps_w=0.0002*(numpy.cos(sweep_w))**2*S_affected_w
    #f_gaps_h=0.0002*(numpy.cos(sweep_h))**2*S_affected_h
    #f_gaps_v=0.0002*(numpy.cos(sweep_v))**2*S_affected_v

    #f_gapst = f_gaps_w + f_gaps_h + f_gaps_v
    
    # TODO: do this correctly
    total_gap_drag = 0.000

    # ------------------------------------------------------------------
    #   Nacelle base drag
    # ------------------------------------------------------------------
    total_nacelle_base_drag = 0.0
    nacelle_base_drag_results = Data()
    
    for nacelle in geometry.nacelles:
        
        # calculate
        nacelle_base_drag = 0.5/12. * np.pi * nacelle.diameter * 0.2/vehicle_reference_area
        
        # dump
        nacelle_base_drag_results[nacelle.tag] = nacelle_base_drag * ones_1col
        
        # increment
        total_nacelle_base_drag += nacelle_base_drag
        

    # ------------------------------------------------------------------
    #   Fuselage upsweep drag
    # ------------------------------------------------------------------
    fuselage_upsweep_drag = 0.006 / vehicle_reference_area
    
    # ------------------------------------------------------------------
    #   Fuselage base drag
    # ------------------------------------------------------------------    
    fuselage_base_drag = 0.0
    
    # ------------------------------------------------------------------
    #   The final result
    # ------------------------------------------------------------------
    
    total_miscellaneous_drag = total_gap_drag          \
                             + total_nacelle_base_drag \
                             + fuselage_upsweep_drag   \
                             + fuselage_base_drag 
    
    
    # dump to results
    conditions.aerodynamics.drag_breakdown.miscellaneous = Data(
        fuselage_upsweep = fuselage_upsweep_drag     *ones_1col, 
        nacelle_base     = nacelle_base_drag_results ,
        fuselage_base    = fuselage_base_drag        *ones_1col,
        control_gaps     = total_gap_drag            *ones_1col,
        total            = total_miscellaneous_drag  *ones_1col,
    )
       
    return total_miscellaneous_drag *ones_1col
    
