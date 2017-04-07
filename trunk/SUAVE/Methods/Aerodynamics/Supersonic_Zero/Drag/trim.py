## @ingroup methods-aerodynamics-Supersonic_Zero-Drag
# trim.py
# 
# Created:  Aug 2014, T. Macdonald
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# Suave imports
from SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#  Computes the trim drag
# ----------------------------------------------------------------------

## @ingroup methods-aerodynamics-Supersonic_Zero-Drag
def trim(state,settings,geometry):
    """Adjusts aircraft drag based on a trim correction

    Assumptions:
    None

    Source:
    Unknown

    Inputs:
    settings.trim_drag_correction_factor                   [Unitless]
    state.conditions.aerodynamics.drag_breakdown.untrimmed [Unitless]

    Outputs:
    aircraft_total_drag_trim_corrected                     [Unitless]

    Properties Used:
    N/A
    """     

    # unpack inputs
    conditions    = state.conditions
    configuration = settings
    
    trim_correction_factor     = configuration.trim_drag_correction_factor    
    untrimmed_drag             = conditions.aerodynamics.drag_breakdown.untrimmed
    
    # trim correction
    aircraft_total_drag_trim_corrected = trim_correction_factor * untrimmed_drag
    
    conditions.aerodynamics.drag_breakdown.trim_corrected_drag                  = aircraft_total_drag_trim_corrected    
    conditions.aerodynamics.drag_breakdown.miscellaneous.trim_correction_factor = trim_correction_factor

    return aircraft_total_drag_trim_corrected