# miscellaneous_drag_aircraft_ESDU.py
# 
# Created:  Jan 2014, T. Orra
# Modified: Jan 2016, E. Botero    

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# SUAVE imports
from SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#  Computes the miscellaneous drag
# ----------------------------------------------------------------------

def miscellaneous_drag_aircraft_ESDU(state,settings,geometry):
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
    
    conditions    = state.conditions
    configuration = settings
    
    Sref      = geometry.reference_area
    ones_1col = conditions.freestream.mach_number *0.+1

    # Estimating total wetted area
    swet_tot        = 0.
    for wing in geometry.wings:
        wing = geometry.wings[wing]
        swet_tot += wing.areas.wetted

    for fuselage in geometry.fuselages:
        fuselage = geometry.fuselages[fuselage]
        swet_tot += fuselage.areas.wetted

    for propulsor in geometry.propulsors:
        propulsor = geometry.propulsors[propulsor]
        swet_tot += propulsor.areas.wetted * propulsor.number_of_engines

    swet_tot *= 1.10
    
    # Estimating excrescence drag, based in ESDU 94044, figure 1
    D_q = 0.40* (0.0184 + 0.000469 * swet_tot - 1.13*10**-7 * swet_tot ** 2)
    cd_excrescence = D_q / Sref

    # ------------------------------------------------------------------
    #   The final result
    # ------------------------------------------------------------------
    # dump to results
    conditions.aerodynamics.drag_breakdown.miscellaneous = Results(
        total_wetted_area         = swet_tot,
        reference_area            = Sref ,
        total                     = cd_excrescence *ones_1col, )

    return cd_excrescence *ones_1col
