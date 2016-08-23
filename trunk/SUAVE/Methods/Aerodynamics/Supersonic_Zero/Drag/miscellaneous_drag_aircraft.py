# miscellaneous_drag_aircraft.py
# 
# Created:  Aug 2014, T. Macdonald
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Analyses import Results

import numpy as np

# ----------------------------------------------------------------------
#  Miscellaneous Drag Aircraft
# ----------------------------------------------------------------------


def miscellaneous_drag_aircraft(state,settings,geometry):
    """ SUAVE.Methods.miscellaneous_drag_aircraft(Wing,segment)
        computes the miscellaneous drag associated with an aircraft
        
        Inputs:
            aircraft- An aircraft object is passed in
            segment - the segment object contains information regarding the mission segment
            Cl - wing Cl
        Outpus:
            cd_misc  - returns the miscellaneous drag assoicated with the wing
            
            >> try to minimize outputs
            >> pack up outputs into Data() if needed
        
        Assumptions:
            if needed
        
    """

    # unpack inputs
    configuration = settings
    
    trim_correction_factor = configuration.trim_drag_correction_factor    
    propulsors             = geometry.propulsors
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
    total_gap_drag = 0.0001

    # ------------------------------------------------------------------
    #   Nacelle base drag
    # ------------------------------------------------------------------
    total_nacelle_base_drag = 0.0
    nacelle_base_drag_results = Results()
    
    for propulsor in propulsors.values():
        
        # calculate
        nacelle_base_drag = 0.5/12. * np.pi * propulsor.nacelle_diameter * 0.2/vehicle_reference_area
        
        # dump
        nacelle_base_drag_results[propulsor.tag] = nacelle_base_drag * ones_1col
        
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
    conditions.aerodynamics.drag_breakdown.miscellaneous = Results(
        fuselage_upsweep = fuselage_upsweep_drag     *ones_1col, 
        nacelle_base     = nacelle_base_drag_results ,
        fuselage_base    = fuselage_base_drag        *ones_1col,
        control_gaps     = total_gap_drag            *ones_1col,
        total            = total_miscellaneous_drag  *ones_1col,
    )
       
    return total_miscellaneous_drag *ones_1col
    
