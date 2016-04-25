# engine_piston.py
# 
# Created:  Mar 2016, M. Vegh
# Modified: 

#From General Aviation Aircraft Design: Applied Methods and Procedures- Snorri Gudmundsson

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Piston Engine
# ----------------------------------------------------------------------

def engine_piston(rated_power):
    """ weight = SUAVE.Methods.Weights.Correlations.Propulsion.engine_jet(thrust_sls)
        Calculate the weight of the dry engine  
    
        Inputs:
                rated_power - rated power of a single engine [pounds]
        
        Outputs:
                weight - weight of the dry engine [kilograms]
            
        Assumptions:
                calculated engine weight from a correlation of engines 
    """    
    # setup
    p_engine_hp = rated_power/Units.horsepower
    # process
    weight = ((p_engine_hp-21.55)/.5515)* Units.lb # Convert lbs to kg
    
    return weight