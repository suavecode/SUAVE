## @ingroup Methods-Weights-Correlations-Propulsion
# engine_jet.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero   


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units

# ----------------------------------------------------------------------
#   Jet Engine
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Propulsion
def engine_jet(thrust_sls):
    """ Calculate the weight of the dry engine  
    
    Assumptions:
            calculated engine weight from a correlation of engines
    
    Source: 
            N/A
            
    Inputs:
            thrust_sls - sea level static thrust of a single engine [Newtons]
    
    Outputs:
            weight - weight of the dry engine                       [kilograms]
        
    Properties Used:
            N/A
    """     
    # setup
    thrust_sls_en = thrust_sls / Units.force_pound # Convert N to lbs force  
    
    # process
    weight = (0.4054*thrust_sls_en ** 0.9255) * Units.lb # Convert lbs to kg
    
    return weight