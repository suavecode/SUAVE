## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
# compute_turbofan_geometry.py
#
# Created:  Jun 2015, A. Variyar 
# Modified: Mar 2016, M. Vegh
#           Apr 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Correlation-based methods to compute engine geometry
# ----------------------------------------------------------------------

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
def compute_turbofan_geometry(turbofan, conditions=None):
    """Estimates geometry for a ducted fan.
    
    This script doesn't actually use conditions; however, it takes it as input to maintain common interface

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
    slsthrust         = turbofan.sealevel_static_thrust / Units.lbf #convert from N to lbs. in correlation

    #based on 241 notes
    nacelle_diameter_in  = 1.0827*slsthrust**0.4134
    nacelle_diameter     = 0.0254*nacelle_diameter_in

    
    #compute other dimensions based on AA241 notes
    L_eng_in          = 2.4077*slsthrust**0.3876
    L_eng_m           = L_eng_in * Units.inches        #engine length in metres
   

    # pack
    turbofan.engine_length    = L_eng_m
    turbofan.nacelle_diameter = nacelle_diameter 
    turbofan.inlet_diameter   = nacelle_diameter/np.sqrt(2.1)
  
    turbofan.areas.wetted     = 1.1*np.pi*turbofan.nacelle_diameter*turbofan.engine_length
    
    return turbofan