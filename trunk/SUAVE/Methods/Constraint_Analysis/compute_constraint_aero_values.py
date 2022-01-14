## @ingroup Methods-Constraint_Analysis
# compute_constraint_aero_values.py
# 
# Created:  Jan 2022, S. Karpuk
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import numpy as np
from SUAVE.Components.Wings                                                              import Main_Wing
from SUAVE.Methods.Constraint_Analysis.compressibility_drag_constraint                   import compressibility_drag_constraint     as  compressibility_drag
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency                   as  oswald_efficiency

# ------------------------------------------------------------------------------------
#  Compute maximum lift coefficient for the constraint analysis
# ------------------------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_constraint_aero_values(W_S,mach,q,vehicle,ca):
    """Computes useful aerodynamic quantities for the constraint diagram

        Assumptions:

        Source:

        Inputs:
            W_S                                     [N/m**2]
            mach                                    [Unitless]
            q                                       [Pa]
            vehicle.aspect_ratio                    [Unitless]
            ca.aerodynamics.cd_min_clean            [Unitless]
              

        Outputs:
            cd_0           [Unitless]
            k              [Unitless]

        Properties Used:

    """  

    # Unpack inputs
    cd_min    = ca.aerodynamics.cd_min_clean 

    main_wing = None
    wings     = vehicle.wings
   
    for wing in wings:
        if isinstance(wing,Main_Wing):
            main_wing = wing

    AR = main_wing.aspect_ratio

    CL = W_S/q

    # Calculate compressibility_drag
    cd_comp   = compressibility_drag(mach,CL,vehicle.wings['main_wing']) 
    cd_comp_e = compressibility_drag(mach,0,vehicle.wings['main_wing'])  
    cd0       = cd_min + cd_comp  

    # Calculate Oswald efficiency
    e = oswald_efficiency(ca,vehicle,cd_min+cd_comp_e)
    k = 1/(np.pi*e*AR)


    return k, cd0



    
