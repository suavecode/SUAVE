## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
# compressible_mixed_flat_plate.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Jan 2016, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from jax import jit
import jax.numpy as np


# ----------------------------------------------------------------------
#  Compressible Mixed Flat Plate
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
@jit
def compressible_mixed_flat_plate(Re,Ma,Tc,xt):
    """Computes the coefficient of friction for a flat plate given the 
    input parameters. Also returns the correction terms used in the
    computation.

    Assumptions:
    Reynolds number between 10e5 and 10e9
    xt between 0 and 1

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    Re (Reynolds number)                                             [Unitless]
    Ma (Mach number)                                                 [Unitless]
    Tc (temperature)                                                 [K]
    xt (turbulent transition point as a proportion of chord length)  [Unitless]

    Outputs:
    cf_comp (coefficient of friction)                                [Unitless]
    k_comp (compressibility correction)                              [Unitless]
    k_reyn (Reynolds number correction)                              [Unitless]

    Properties Used:
    N/A
    """     
    
    Rex = Re*xt
    Rex = np.maximum(Rex,0.0001)

    theta = 0.671*xt/(Rex**0.5)
    xeff  = (27.78*theta*Re**0.2)**1.25
    Rext  = Re*(1-xt+xeff)
    
    cf_turb  = 0.455/(np.log10(Rext)**2.58)
    cf_lam   = 1.328/(Rex**0.5)
    
    xt = np.ones_like(Re)*xt
    cf_start = 0.455/(np.log10(Re*xeff)**2.58)
    cf_start = np.where(xt==0.,0,cf_start)
    
    
    
    cf_inc = cf_lam*xt + cf_turb*(1-xt+xeff) - cf_start*xeff
    
    # compressibility correction
    Tw = Tc * (1. + 0.178*Ma*Ma)
    Td = Tc * (1. + 0.035*Ma*Ma + 0.45*(Tw/Tc - 1.))
    k_comp = (Tc/Td) 
    
    # reynolds correction
    Rd_w   = Re * (Td/Tc)**1.5 * ( (Td+216.) / (Tc+216.) )
    k_reyn = (Re/Rd_w)**0.2
    
    # apply corrections
    cf_comp = cf_inc * k_comp * k_reyn
    
    return cf_comp, k_comp, k_reyn