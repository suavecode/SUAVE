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
def untrimmed(state,settings,geometry):
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
    parasite_total        = conditions.aerodynamics.drag_breakdown.parasite.total         
    induced_total         = conditions.aerodynamics.drag_breakdown.induced.total            
    compressibility_total = conditions.aerodynamics.drag_breakdown.compressible.total         
    miscellaneous_drag    = conditions.aerodynamics.drag_breakdown.miscellaneous.total 

    # untrimmed drag
    aircraft_untrimmed = parasite_total        \
                       + induced_total         \
                       + compressibility_total \
                       + miscellaneous_drag
    
    
    
    
    conditions.aerodynamics.drag_breakdown.untrimmed = aircraft_untrimmed
    
    return aircraft_untrimmed
