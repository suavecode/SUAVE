## @ingroup Methods-Propulsion
# turbofan_nox_emission_index.py
# 
# Created:  Sep 2015, M. Vegh
# Modified: Feb 2016, E. Botero
#        

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#   turbofan_nox_emission_index
# ----------------------------------------------------------------------


## @ingroup Methods-Propulsion
def turbofan_nox_emission_index(turbofan, state):
    """
    Outputs a turbofan's nox_emission_index takens from a regression calculated
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
    nox_emission_index            [kg/kg]
    
    Source: Antione, Nicholas, Aircraft Optimization for Minimal Environmental Impact, pp. 31 (PhD Thesis)
    
    
    """
    
    results = turbofan(state)
    p3      = turbofan.combustor.inputs.stagnation_pressure/Units.psi
    T3      = turbofan.combustor.inputs.stagnation_temperature/Units.degR 
    T4      = turbofan.combustor.outputs.stagnation_temperature/Units.degR
    
    nox_emission_index = .004194*T4*((p3/439.)**.37)*np.exp((T3-1471.)/345.)
    
    #correlation in g Nox/kg fuel; convert to kg Nox/kg
    nox_emission_index = nox_emission_index * (Units.g/Units.kg) 
    
    return nox_emission_index