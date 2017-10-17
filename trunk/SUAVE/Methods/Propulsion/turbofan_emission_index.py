## @ingroup Methods-Propulsion
# turbofan_emission_index.py
# 
# Created:  Sep 2015, M. Vegh
# Modified: Feb 2016, E. Botero
#        

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#   turbofan_nox_emission_index
# ----------------------------------------------------------------------


## @ingroup Methods-Propulsion
def turbofan_emission_index(turbofan, state):
    """
    Outputs a turbofan's emission_index takens from a regression calculated
    from NASA's Engine Performance Program (NEPP)
    
    Inputs:
    turbofan.
      combustor.
        inputs.
          stagnation_pressure     [Pa]
          stagnation_temperature  [K]
        outputs.
          stagnation_temperature  [K]
          
    Outputs:      
    emission_index.
          NOx                     [kg/kg]
          CO2                     [kg/kg]
          H2O                     [kg/kg]
          SO2                     [kg/kg]
    
    Source: Antoine, Nicholas, Aircraft Optimization for Minimal Environmental Impact, pp. 31 (PhD Thesis)
    
    
    """
    
    results = turbofan(state)
    p3      = turbofan.combustor.inputs.stagnation_pressure/Units.psi
    T3      = turbofan.combustor.inputs.stagnation_temperature/Units.degR 
    T4      = turbofan.combustor.outputs.stagnation_temperature/Units.degR
    
    nox_emission_index = .004194*T4*((p3/439.)**.37)*np.exp((T3-1471.)/345.)
    CO2                = 3.155  # This is in kg/kg
    H2O                = 1.240  # This is in kg/kg 
    SO2                = 0.0008 # This is in kg/kg 
    
    #correlation in g Nox/kg fuel; convert to kg Nox/kg
    nox_emission_index = nox_emission_index * (Units.g/Units.kg) 
    
    emission_index = Data()
    emission_index.NOx = nox_emission_index
    emission_index.CO2 = CO2
    emission_index.H2O = H2O
    emission_index.SO2 = SO2
    
    
    return emission_index