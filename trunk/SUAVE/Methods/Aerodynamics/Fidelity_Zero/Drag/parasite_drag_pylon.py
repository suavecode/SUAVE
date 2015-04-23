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
def parasite_drag_pylon(conditions,configuration,geometry):
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
    print geometry
    # unpack
    pylon_factor        =  0.20 # 20% of propulsor drag
    n_propulsors        =  len(geometry.propulsors)  # number of propulsive system in vehicle (NOT # of ENGINES)
    
    pylon_parasite_drag = 0.00
    pylon_wetted_area   = 0.00
    pylon_cf            = 0.00
    pylon_compr_fact    = 0.00
    pylon_rey_fact      = 0.00
    pylon_FF            = 0.00

    # Estimating pylon drag
    for propulsor in geometry.propulsors:
        pylon_parasite_drag += pylon_factor *  conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].parasite_drag_coefficient
        pylon_wetted_area   += pylon_factor *  conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].wetted_area * propulsor.number_of_engines
        pylon_cf            += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].skin_friction_coefficient
        pylon_compr_fact    += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].compressibility_factor
        pylon_rey_fact      += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].reynolds_factor
        pylon_FF            += conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].form_factor
        
    pylon_cf            /= n_propulsors           
    pylon_compr_fact    /= n_propulsors   
    pylon_rey_fact      /= n_propulsors     
    pylon_FF            /= n_propulsors      
    
    # dump data to conditions
    pylon_result = Results(
        wetted_area               = pylon_wetted_area   ,
        reference_area            = geometry.reference_area   ,
        parasite_drag_coefficient = pylon_parasite_drag ,
        skin_friction_coefficient = pylon_cf  ,
        compressibility_factor    = pylon_compr_fact   ,
        reynolds_factor           = pylon_rey_fact   ,
        form_factor               = pylon_FF   ,
    )
    conditions.aerodynamics.drag_breakdown.parasite['pylon'] = pylon_result

    # done!
    return pylon_parasite_drag
