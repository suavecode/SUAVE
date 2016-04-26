# integrated_propulsion.py
# 
# Created:  Apr 2016, M. Vegh 

#from Aircraft Design: A Conceptual Approach by Raymer

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#   Integrated Propulsion
# ----------------------------------------------------------------------

def integrated_propulsion_general_aviation(engine_piston,num_eng, engine_wt_exponent=.922):
    """ weight = SUAVE.Methods.Correlations.Propulsion.integrated_propulsion_general_aviation(engine_piston,num_eng, engine_exponent=.922)
        Calculate the weight of the entire propulsion system        
                
        Inputs:
                engine_piston - dry weight of a single engine [kilograms]
                num_eng - total number of engines on the aircraft [dimensionless]
                engine_wt_exponent- weight exponent for entire integrated propulsion system [dimensionless]
        
        Outputs:
                weight - weight of the full propulsion system [kilograms]
            
        Assumptions:
                The propulsion system is a fixed 60% greater than the dry engine alone. 
                The propulsion system includes the engines, engine exhaust, reverser, starting,
                controls, lubricating, and fuel systems. The nacelle and pylon weight are also
                part of this calculation.
    """     
    
    wt_piston = engine_piston/Units.lbs
    weight = 2.575*(wt_piston**.922)*num_eng*Units.lbs
    
    return weight
    