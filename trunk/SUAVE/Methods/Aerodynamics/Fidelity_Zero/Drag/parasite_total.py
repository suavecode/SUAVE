# parasite_drag_pylon.py
#
# Created:  Tarik, Jan 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# Suave imports
from SUAVE.Core import Results

# ----------------------------------------------------------------------
#  Computes the pyloan parasite drag
# ----------------------------------------------------------------------
#def parasite_drag_pylon(conditions,configuration,geometry):
def parasite_total(state,settings,geometry):
    """ SUAVE.Methods.parasite_drag_pylon(conditions,configuration,geometry):
        Simplified estimation, considering pylon drag a fraction of the nacelle drag

        Inputs:
            conditions      - data dictionary for output dump
            configuration   - not in use
            geometry        - SUave type vehicle

        Outpus:
            cd_misc  - returns the miscellaneous drag associated with the vehicle

        Assumptions:
            simplified estimation, considering pylon drag a fraction of the nacelle drag

    """

    # unpack
    total_wing_parasite_drag = state.conditions.aerodynamics.drag_breakdown.wing_parasite_total
    total_fuselage_parasite_drag = state.conditions.aerodynamics.drag_breakdown.pylon_parasite_total
    total_propulsor_parasite_drag = state.conditions.aerodynamics.drag_breakdown.propulsor_parasite_total
    total_pylon_parasite_drag = state.conditions.aerodynamics.drag_breakdown.fuselage_parasite_total
    
    # start conditions node
    drag_breakdown.parasite = Results()
    
    # from wings
    total_parasite_drag += total_wing_parasite_drag

    # from fuselage    
    total_parasite_drag += total_fuselage_parasite_drag   
    
    # from propulsors    
    total_parasite_drag += total_propulsor_parasite_drag
    
    # from pylons
    total_parasite_drag += total_pylon_parasite_drag
    
        
    # dump to condtitions
    state.conditions.aerodynamics.drag_breakdown.parasite.total = total_parasite_drag



    # done!
    return total_parasite_drag
