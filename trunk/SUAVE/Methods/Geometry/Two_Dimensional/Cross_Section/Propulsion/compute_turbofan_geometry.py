# engine_geometry.py
#
# Created:  Jun 15, A. Variyar 
# Modified: Mar 16, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core  import Data

# package imports
import numpy as np
from math import pi, sqrt

# ----------------------------------------------------------------------
#  Correlation-based methods to compute engine geometry
# ----------------------------------------------------------------------

def compute_turbofan_geometry(turbofan, conditions):
    """ SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion.compute_engine_geometry
    computes turbofan geometry based on SLS thrust (from AA241 notes)
    """

    #unpack
    thrust            = turbofan.thrust
    core_nozzle       = turbofan.core_nozzle
    fan_nozzle        = turbofan.fan_nozzle
    bypass_ratio      = turbofan.bypass_ratio
    
    slsthrust         = turbofan.sealevel_static_thrust
    slsthrust         = slsthrust*0.224809

    #note; this script doesn't actually use conditions; however, it takes it as input to maintain common interface

    #based on 241 notes
    nacelle_diameter_in  = 1.0827*slsthrust**0.4134
    nacelle_diameter     = 0.0254*nacelle_diameter_in
    
    #compute exit area
    rho5_fan          = fan_nozzle.outputs.density
    U5_fan            = fan_nozzle.outputs.velocity
    rho5_core         = core_nozzle.outputs.density
    U5_core           = core_nozzle.outputs.velocity
    
    mass_flow         = thrust.mass_flow_rate_design
    mass_flow_fan     = mass_flow*bypass_ratio
    
    Ae_fan            = mass_flow_fan/(rho5_fan*U5_fan)
    Ae_core           = mass_flow/(rho5_core*U5_core)
    
    #compute other dimensions based on AA241 notes
    L_eng_in          = 2.4077*slsthrust**0.3876
    L_eng_m           = 0.0254*L_eng_in          #engine length in metres
    Amax              = (np.pi/4.)*(nacelle_diameter*nacelle_diameter)
    Ainlet            = .7*Amax
    Ainflow           = .8*Ainlet
    Aexit             = (Ae_fan+Ae_core)[0][0]
   

    # pack
    turbofan.engine_length    = L_eng_m
    turbofan.nacelle_diameter = nacelle_diameter
    turbofan.areas.maximum    = Amax 
    turbofan.areas.inflow     = Ainflow
    turbofan.areas.exit       = Aexit
    turbofan.areas.wetted     = .9*np.pi*turbofan.nacelle_diameter*turbofan.engine_length
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print