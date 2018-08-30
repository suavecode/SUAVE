## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
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

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Propulsion
def compute_ducted_fan_geometry(ducted_fan, conditions):
    """Estimates geometry for a ducted fan.
    
    Assumptions:
    None

    Source:
    None

    Inputs:
    ducted_fan.thrust.
      mass_flow_rate_design [kg/s]
    ducted_fan.fan_nozzle.
      outputs.velocity      [m/s]
      outputs.density       [kg/m^3]
      outputs.area_ratio    [-]
    conditions.freestream.
      velocity              [m/s]
      density               [kg/m^3]

    Outputs:
    ducted_fan.
      areas.maximum         [m^2]
      areas.wetted          [m^2]
      nacelle_diameter      [m]
      engine_length         [m]

    Properties Used:
    N/A
    """      

    # unpack
    thrust            = ducted_fan.thrust
    fan_nozzle        = ducted_fan.fan_nozzle
    mass_flow         = thrust.mass_flow_rate_design

    #evaluate engine at these conditions
    state=Data()
    state.conditions=conditions
    state.numerics= Data()
    ducted_fan.evaluate_thrust(state)
    
    #determine geometry
    U0       = conditions.freestream.velocity
    rho0     = conditions.freestream.density
    Ue       = fan_nozzle.outputs.velocity
    rhoe     = fan_nozzle.outputs.density
    Ae       = mass_flow[0][0]/(rhoe[0][0]*Ue[0][0]) #ducted fan nozzle exit area
    A0       = (mass_flow/(rho0*U0))[0][0]
    

   
    ducted_fan.areas.maximum = 1.2*Ae/fan_nozzle.outputs.area_ratio[0][0]
    ducted_fan.nacelle_diameter = 2.1*((ducted_fan.areas.maximum/np.pi)**.5)

    ducted_fan.engine_length    = 1.5*ducted_fan.nacelle_diameter
    ducted_fan.areas.wetted     = ducted_fan.nacelle_diameter*ducted_fan.engine_length*np.pi

    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print()