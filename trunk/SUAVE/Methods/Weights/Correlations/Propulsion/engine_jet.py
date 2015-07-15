# engine_jet.py
# 
# Created:  Andrew Wendorff, Jan 2014
# Modified: Andrew Wendorff, Feb 2014      


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------


def engine_jet(thrust_sls):
    """ weight = SUAVE.Methods.Weights.Correlations.Propulsion.engine_jet(thrust_sls)
        Calculate the weight of the dry engine  
    
        Inputs:
                thrust_sls - sea level static thrust of a single engine [Newtons]
        
        Outputs:
                weight - weight of the dry engine [kilograms]
            
        Assumptions:
                calculated engine weight from a correlation of engines 
    """    
    # setup
    thrust_sls_en = thrust_sls / Units.force_pound # Convert N to lbs force  
    
    # process
    weight = (0.4054*thrust_sls_en ** 0.9255) * Units.lb # Convert lbs to kg
    
    return weight