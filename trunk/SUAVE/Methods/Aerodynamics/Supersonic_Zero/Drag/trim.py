# miscellaneous_drag_aircraft_ESDU.py
#
# Created:  Tarik, Jan 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# Suave imports
from SUAVE.Core import Results

# ----------------------------------------------------------------------
#  Computes the miscellaneous drag
# ----------------------------------------------------------------------
def trim(state,settings,geometry):
#def miscellaneous_drag_aircraft_ESDU(conditions,configuration,geometry):
    """ SUAVE.Methods.miscellaneous_drag_aircraft_ESDU(conditions,configuration,geometry):
        computes the miscellaneous drag based in ESDU 94044, figure 1

        Inputs:
            conditions      - data dictionary for output dump
            configuration   - not in use
            geometry        - SUave type vehicle

        Outpus:
            cd_misc  - returns the miscellaneous drag associated with the vehicle

        Assumptions:
            if needed

    """

    # unpack inputs
    
    conditions = state.conditions
    configuration = settings
    
    trim_correction_factor     = configuration.trim_drag_correction_factor    
    untrimmed_drag             = conditions.aerodynamics.drag_breakdown.untrimmed
    
    # trim correction
    aircraft_total_drag_trim_corrected = trim_correction_factor * untrimmed_drag
    
    conditions.aerodynamics.drag_breakdown.trim_corrected_drag = aircraft_total_drag_trim_corrected    
    
    
    conditions.aerodynamics.drag_breakdown.miscellaneous.trim_correction_factor = trim_correction_factor


    return aircraft_total_drag_trim_corrected
