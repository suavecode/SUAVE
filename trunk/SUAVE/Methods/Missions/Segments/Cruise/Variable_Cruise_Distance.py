## @ingroup Methods-Missions-Segments-Cruise
# Variable_Cruise_Distance.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# --------------------------------------------------------------
#   Initialize - for cruise distance
# --------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Cruise
def initialize_cruise_distance(segment,state):
    """This is a method that allows your vehicle to land at prescribed landing weight

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.cruise_tag              [string]
    segment.distance                [meters]

    Outputs:
    state.unknowns.cruise_distance  [meters]

    Properties Used:
    N/A
    """         
    
    # unpack
    cruise_tag = segment.cruise_tag
    distance   = segment.segments[cruise_tag].distance
    
    # apply, make a good first guess
    state.unknowns.cruise_distance = distance
    
    return


# --------------------------------------------------------------
#   Unknowns - for cruise distance
# --------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def unknown_cruise_distance(segment,state):
    """This is a method that allows your vehicle to land at prescribed landing weight

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.cruise_tag              [string]
    state.unknowns.cruise_distance  [meters]

    Outputs:
    segment.distance                [meters]

    Properties Used:
    N/A
    """      
    
    # unpack
    distance = state.unknowns.cruise_distance
    cruise_tag = segment.cruise_tag
    
    # apply the unknown
    segment.segments[cruise_tag].distance = distance
    
    return


# --------------------------------------------------------------
#   Residuals - for Take Off Weight
# --------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def residual_landing_weight(segment,state):
    """This is a method that allows your vehicle to land at prescribed landing weight.
    This takes the final weight and compares it against the prescribed landing weight.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.segments[-1].conditions.weights.total_mass [kilogram]
    segment.target_landing_weight                    [kilogram]

    Outputs:
    state.residuals.landing_weight                   [kilogram]

    Properties Used:
    N/A
    """      
    
    # unpack
    landing_weight = state.segments[-1].conditions.weights.total_mass[-1]
    target_weight  = segment.target_landing_weight
    
    # this needs to go to zero for the solver to complete
    state.residuals.landing_weight = landing_weight - target_weight
    
    return
    
    