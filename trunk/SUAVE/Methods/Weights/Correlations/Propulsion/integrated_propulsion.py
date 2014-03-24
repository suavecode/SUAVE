# integrated_propulsion.py
# 
# Created:  Andrew Wendorff, Jan 2014
# Modified: Andrew Wendorff, Feb 2014       


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def integrated_propulsion(engine_jet,num_eng, engine_wt_factor = 1.6):
    """ weight = SUAVE.Methods.Correlations.Propulsion.integrated_propulsion(engine_jet,num_eng)
        Calculate the weight of the entire propulsion system        
                
        Inputs:
                engine_jet - dry weight of the engine [kilograms]
                num_eng - total number of engines on the aircraft [dimensionless]
                engine_wt_factor - weight increase factor for entire integrated propulsion system [dimensionless]
        
        Outputs:
                weight - weight of the full propulsion system [kilograms]
            
        Assumptions:
                The propulsion system is a fixed 60% greater than the dry engine alone. 
                The propulsion system includes the engines, engine exhaust, reverser, starting,
                controls, lubricating, and fuel systems. The nacelle and pylon weight are also
                part of this calculation.
    """     
    
    #process
    
    weight = engine_jet * num_eng * engine_wt_factor
    
    return weight
    