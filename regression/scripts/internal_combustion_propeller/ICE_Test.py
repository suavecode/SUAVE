# ICE_Test.py
# 
# Created: Feb 2020, M. Clarke 
 
""" setup file for a mission with a Cessna 172 with an internal combustion engine network
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np 
 

from SUAVE.Core import (
Data, Container,
)

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from Cessna_172      import vehicle_setup  
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
     
    # Define internal combustion engine from Cessna Regression Aircraft 
    vehicle    = vehicle_setup()
    ice_engine = vehicle.propulsors.internal_combustion.engine
    
    # Define conditions 
    conditions                                         = Data()
    conditions.freestream                              = Data() 
    conditions.propulsion                              = Data() 
    conditions.freestream.altitude                     = np.array([[8000]]) * Units.feet
    conditions.freestream.delta_ISA                    = 0.0
    conditions.propulsion.combustion_engine_throttle   = np.array([[0.8]])  
    
    ice_engine.power(conditions)   

    # Truth values for propeller with airfoil geometry defined 
    P_truth      = 81367.49237183
    P_sfc_truth  = 0.52
    FFR_truth    = 0.007149134158858348
    Q_truth      = 287.7786359548746
    
    P            = ice_engine.outputs.power[0][0]                            
    P_sfc        = ice_engine.outputs.power_specific_fuel_consumption 
    FFR          = ice_engine.outputs.fuel_flow_rate[0][0]                    
    Q            = ice_engine.outputs.torque[0][0]        
    
    # Store errors 
    error = Data()
    error.P      = np.max(np.abs(P     - P_truth    ))
    error.P_sfc  = np.max(np.abs(P_sfc - P_sfc_truth))
    error.FFR    = np.max(np.abs(FFR   - FFR_truth  ))
    error.Q      = np.max(np.abs(Q     - Q_truth    ))

    print('Errors:')
    print(error)

    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)

    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    