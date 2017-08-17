## @ingroup Methods-Weights-Correlations-Propulsion
# integrated_propulsion.py
# 
# Created:  Jan 2014, M. A. Wendorff 
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#   Integrated Propulsion
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Propulsion
def integrated_propulsion(engine_jet,num_eng, engine_wt_factor = 1.6):
    """ Calculate the weight of the entire propulsion system 
    
    Assumptions:
            The propulsion system is a fixed 60% greater than the dry engine alone. 
            The propulsion system includes the engines, engine exhaust, reverser, starting,
            controls, lubricating, and fuel systems. The nacelle and pylon weight are also
            part of this calculation.           
            
    Source: 
            N/A
            
    Inputs:
            engine_jet - dry weight of the engine                                             [kilograms]
            num_eng - total number of engines on the aircraft                                 [dimensionless]
            engine_wt_factor - weight increase factor for entire integrated propulsion system [dimensionless]
    
    Outputs:
            weight - weight of the full propulsion system                                     [kilograms]
        
    Properties Used:
            N/A
    """   
    
    weight = engine_jet * num_eng * engine_wt_factor
    
    return weight
    