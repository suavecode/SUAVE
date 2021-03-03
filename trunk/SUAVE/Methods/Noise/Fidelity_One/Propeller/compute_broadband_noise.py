# compute_broadband_noise.py
#
# Created:  Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data
import numpy as np
from scipy.special import jv 

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.decibel_arithmetic import pressure_ratio_to_SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.dbA_noise          import A_weighting

from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import atmospheric_attenuation
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_arithmetic
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import senel_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import dbA_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_harmonic_to_third_octave

## @ingroupMethods-Noise-Fidelity_One-Propeller
def compute_broadband_noise(network,propeller,auc_opts,segment,settings, mic_loc, harmonic_test ):
     
     
     delta_star_p
     M
     L
     D_bar_h
     re
     
     # that the boundary layer displacement thicknesses on the two sides, as well as boundary layer thicknesses
     # to be used below, are computed as functions27 of Rec and a* which were derived from detailed 
     # near-wake flow measurements from the aforementioned isolated NACA 0012 blade section studies
     
     # Turbulent Boundary Layer - Trailing Edge noise
     ##  pressure side
     H_p
     
     G_TBL_TEp      = ((delta_star_p*(M**5)*L*D_bar_h)/re**2)* H_p 
     
     ## suction side
     H_s 
     
     G_TBL_TEs      = ((delta_star_p*(M**5)*L*D_bar_h)/re**2)* H_s  
     
     ## noise at angle of attack 
     H_alpha  
     
     T_TBL_TEalpha  = ((delta_star_s*(M**5)*L*D_bar_h)/re**2)* H_alpha  
     
     # summation of Turbulent Boundary Layer - Trailing Edge noise sources 
     G_TBL_TE = G_TBL_TEp + G_TBL_TEs + T_TBL_TEalpha  
     
     # Laminar-boundary layer vortex shedding noise 
     H_l      = 
     
     G_LBL_VS = ((delta_p*(M**5)*L*D_bar_h)/re**2)* H_l  
     
     # Blunt Trailing Edge Noise 
     H_b     = 
     
     G_BTE   = ((h*(M**5.5)*L*D_bar_h)/re**2)* H_b  
     
     # Tip noise 
     H_t 
     
     G_Tip  = (((M**2)*(M_max**3)*l*D_bar_h)/re**2)* H_t 
    
     # Addition of noise sources 
     G_self = G_TBL_TE + G_LBL_VS + G_BTE + G_Tip
     
     return G_self