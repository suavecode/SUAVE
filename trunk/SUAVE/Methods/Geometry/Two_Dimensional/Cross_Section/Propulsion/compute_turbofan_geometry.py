## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
# engine_geometry.py
#
# Created:  Jun 15, A. Variyar 
# Modified: Mar 16, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core  import Data, Units

# package imports
import numpy as np
from math import pi, sqrt

# ----------------------------------------------------------------------
#  Correlation-based methods to compute engine geometry
# ----------------------------------------------------------------------

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
def compute_turbofan_geometry(turbofan, conditions):
    """Estimates geometry for a ducted fan.

    Assumptions:
    None

    Source:
    http://adg.stanford.edu/aa241/AircraftDesign.html

    Inputs:
    turbofan.sealevel_static_thrust [N]

    Outputs:
    turbofan.
      engine_length                 [m]
      nacelle_diameter              [m]
      areas.wetted                  [m^2]

    Properties Used:
    N/A
    """    

    #unpack
    thrust            = turbofan.thrust
    core_nozzle       = turbofan.core_nozzle
    fan_nozzle        = turbofan.fan_nozzle
    bypass_ratio      = turbofan.bypass_ratio
    
    slsthrust         = turbofan.sealevel_static_thrust*0.224809 #convert from N to lbs. in correlation
    #slsthrust         = slsthrust*0.224809

    #note; this script doesn't actually use conditions; however, it takes it as input to maintain common interface

    #based on 241 notes
    nacelle_diameter_in  = 1.0827*slsthrust**0.4134
    nacelle_diameter     = 0.0254*nacelle_diameter_in

    
    #compute other dimensions based on AA241 notes
    L_eng_in          = 2.4077*slsthrust**0.3876
    L_eng_m           = 0.0254*L_eng_in          #engine length in metres
   

    # pack
    turbofan.engine_length    = L_eng_m
    turbofan.nacelle_diameter = nacelle_diameter
  
    turbofan.areas.wetted     = 1.1*np.pi*turbofan.nacelle_diameter*turbofan.engine_length
    
    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print()