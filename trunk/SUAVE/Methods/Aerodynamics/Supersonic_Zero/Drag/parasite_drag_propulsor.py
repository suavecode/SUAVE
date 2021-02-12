## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# parasite_drag_propulsor.py
# 
# Created:  Feb 2019, T. MacDonald
# Modified: Jan 2020, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import compressible_turbulent_flat_plate
from SUAVE.Core import Data
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender
from scipy.optimize import fsolve

import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Propulsors
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def parasite_drag_propulsor(state,settings,geometry):
    """Computes the parasite drag due to the propulsor

    Assumptions:
    Basic fit

    Source:
    Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach) (subsonic)
    http://aerodesign.stanford.edu/aircraftdesign/drag/BODYFORMFACTOR.HTML (supersonic)

    Inputs:
    state.conditions.freestream.
      mach_number                                [Unitless]
      temperature                                [K]
      reynolds_number                            [Unitless]
    geometry.      
      nacelle_diameter                           [m^2]
      areas.wetted                               [m^2]
      engine_length                              [m]

    Outputs:
    propulsor_parasite_drag                      [Unitless]

    Properties Used:
    N/A
    """
    
    # unpack inputs
    
    conditions    = state.conditions
    configuration = settings
    propulsor     = geometry
    
    low_mach_cutoff  = settings.begin_drag_rise_mach_number
    high_mach_cutoff = settings.end_drag_rise_mach_number    
        
    freestream = conditions.freestream
    
    Sref        = propulsor.nacelle_diameter**2 / 4 * np.pi
    Swet        = propulsor.areas.wetted
    
    l_prop  = propulsor.engine_length
    d_prop  = propulsor.nacelle_diameter
    
    # conditions
    freestream = conditions.freestream
    Mc = freestream.mach_number
    Tc = freestream.temperature    
    re = freestream.reynolds_number

    # reynolds number
    Re_prop = re*l_prop
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)

    
    # form factor according to Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach)
    k_prop_sub = 1. + 0.35 / (float(l_prop)/float(d_prop)) 
    
    # for supersonic flow (http://adg.stanford.edu/aa241/drag/BODYFORMFACTOR.HTML)
    k_prop_sup = 1.
    
    trans_spline = Cubic_Spline_Blender(low_mach_cutoff,high_mach_cutoff)
    h00 = lambda M:trans_spline.compute(M)
    
    k_prop = k_prop_sub*(h00(Mc)) + k_prop_sup*(1-h00(Mc))
    
    # Spillage drag
    spillage_Cd = spillage_drag(state, geometry, Sref)
    
    # --------------------------------------------------------
    # find the final result    
    propulsor_parasite_drag = k_prop * cf_prop * Swet / Sref  
    
    propulsor_parasite_drag += spillage_Cd
    # --------------------------------------------------------
    
    # dump data to conditions
    propulsor_result = Data(
        wetted_area               = Swet    , 
        reference_area            = Sref    , 
        parasite_drag_coefficient = propulsor_parasite_drag ,
        skin_friction_coefficient = cf_prop ,
        compressibility_factor    = k_comp  ,
        reynolds_factor           = k_reyn  , 
        form_factor               = k_prop  ,
        spillage                  = spillage_Cd ,
    )
    state.conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag] = propulsor_result    
    
    return propulsor_parasite_drag

def spillage_drag(state, geometry, prop_ref_area):
    
    M0 = state.conditions.freestream.mach_number
    A0 = geometry.supersonic_capture_area
    
    mdot = state.conditions.propulsion.air_mass_rate
    rho  = state.conditions.freestream.density
    u    = state.conditions.freestream.velocity
    Ai   = mdot/rho/u
    
    Ai_A0 = Ai/A0
    
    g0 = 32.174
    R = 53.35
    g = 1.4
    gp = 2.4
    gm = 0.4    
    
    Tt_T = 1 + gm/2*M0**2
    Pt_P0 = Tt_T**(g/gm)
    Pt1_Pt0 = np.ones_like(M0)
    dummy_Pt1_Pt0 = ((gp/2*M0**2)/Tt_T)**(g/gm) / (2*g/gp*M0**2 - gm/gp)**(1/gm)
    Pt1_Pt0[M0 > 1.] = dummy_Pt1_Pt0[M0 > 1.]
        
    WpTt_PtA = np.sqrt(g*g0/R)*M0/Tt_T**(gp/(2*gm))
    
    WpTt_PtA_Ai_A0_eta = WpTt_PtA*Ai_A0/Pt1_Pt0
    
    Ms = gmach(WpTt_PtA_Ai_A0_eta, g, 0.5)
    
    Pt_Ps = (1+gm/2*Ms**2)**(g/gm)
    
    Cds_raw = 2/(g*M0**2)*(Pt_P0*Pt1_Pt0/Pt_Ps*(1+g*Ms**2) - 1 - Ai_A0*g*M0**2)
    
    Cds = np.zeros_like(Cds_raw)
    Cds[Cds_raw > 0.] = Cds_raw[Cds_raw > 0.]
    
    Cds[M0 < 1.] = 0.
    
    # return Cd relative to the standard propulsor reference area
    Cds = Cds*A0/prop_ref_area
    
    return Cds

def gmach(Waf, Gam, Xi): 

    Gmach = np.ones_like(Waf)
    
    # nominal cases
    c = (Gam * 32.174 / 53.35) ** 0.5
    mult = (Gam - 1) / 2
    exp = (Gam + 1) / (2 * (1 - Gam))

    X1 = 0 # assumed Xi = 0.5
    X2 = 1
 
    f = (c * X1 * (1 + (mult * (X1 ** 2))) ** exp) - Waf
 
    Gx = np.ones_like(f)*X2
    Gx[f < 0] = X1
        
    def solve_func(x):
        x_temp = x[:,None]
        vec = (c * x_temp * (1 + (mult * (x_temp ** 2))) ** exp) - Waf
        return vec[:,0]
    
    res = fsolve(solve_func, Gx)
    Gx = res
 
    Gmach = Gx[:,None]
    
    # bounding cases
    Gmach[Waf <= 0.] = 0 # assumes Xi = 0.5
    Gmach[Waf > 0.53174] = 1.    

    return Gmach
    