## @ingroup Methods-Constraint_Analysis
# Oswald_efficiency.py
# 
# Created:  Nov 2021, S. Karpuk
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
import numpy as np

# ------------------------------------------------------------------------------------
#  Compute maximum lift coefficient for the constraint analysis
# ------------------------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compressibility_drag_constraint(mach,cl,geometry):
    """Estimates drag due to compressibility for the constranint analysis using reduced number of variables and 

        Assumptions:
            Subsonic to low transonic
            Supercritical airfoil

        Source:
            adg.stanford.edu (Stanford AA241 A/B Course Notes)

        Inputs:
            mach                                    [Unitless]
            cl                                      [Unitless]
            geometry.sweep_quarter_chord            [radians]
                      thickness_to_chord            [Unitless]

        Outputs:
            cd_c           [Unitless]

        Properties Used:

    """  

    # Unpack inputs
    sweep = geometry.sweep_quarter_chord
    t_c   = geometry.thickness_to_chord

    cos_sweep = np.cos(sweep)

    # get effective Cl and sweep
    tc = t_c /(cos_sweep)
    cl = cl / (cos_sweep*cos_sweep)

    # compressibility drag based on regressed fits from AA241
    mcc_cos_ws = 0.922321524499352       \
                - 1.153885166170620*tc    \
                - 0.304541067183461*cl    \
                + 0.332881324404729*tc*tc \
                + 0.467317361111105*tc*cl \
                + 0.087490431201549*cl*cl
        
    # crest-critical mach number, corrected for wing sweep
    mcc = mcc_cos_ws / cos_sweep

    # divergence ratio
    mo_mc = mach/mcc
    
    # compressibility correlation, Shevell
    dcdc_cos3g = 0.0019*mo_mc**14.641
    
    # compressibility drag
    cd_c = dcdc_cos3g * cos_sweep*cos_sweep*cos_sweep


    return cd_c



    
