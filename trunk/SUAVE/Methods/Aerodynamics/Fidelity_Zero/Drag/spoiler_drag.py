# spoiler_drag.py
#
# Created:  Anil, Jan 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# Suave imports
from SUAVE.Core import Results

# ----------------------------------------------------------------------
#  Computes the miscellaneous drag
# ----------------------------------------------------------------------
def spoiler_drag(state,settings,geometry):
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
    
    drag_breakdown             = conditions.aerodynamics.drag_breakdown

    # various drag components
    spoiler_drag = settings.spoiler_drag_increment

    # untrimmed drag
    conditions.aerodynamics.drag_breakdown.spoiler_drag = spoiler_drag
    
    
    return spoiler_drag
