
# integrated_propulsion_general_aviation.py
# 
# Created:  Sept 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units


# ----------------------------------------------------------------------
#   Integrated Propulsion
# ----------------------------------------------------------------------

def integrated_propulsion_general_aviation(engine_piston,num_eng, engine_wt_factor = 2.575, engine_wt_exp = .922):
    """ weight = SUAVE.Methods.Correlations.Propulsion.integrated_propulsion(engine_piston,num_eng)
        Calculate the weight of the entire propulsion system        
                
        Inputs:
                engine_piston - dry weight of a single engine[kilograms]
                num_eng - total number of engines on the aircraft [dimensionless]
                engine_wt_factor - weight increase factor for entire integrated propulsion system [dimensionless]
        
        Outputs:
                weight - weight of the full propulsion system [kilograms]
            
        Source:
                Source: Raymer, Aircraft Design, a Conceptual Approach
    """     
    engine_dry = engine_piston/Units.lbs
    weight = engine_wt_factor * (engine_dry**engine_wt_exp)*num_eng
    mass = weight*Units.lbs #convert to kg
    
    return mass


