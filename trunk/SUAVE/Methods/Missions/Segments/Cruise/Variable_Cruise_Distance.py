## @ingroup Methods-Missions-Segments-Cruise
# Variable_Cruise_Distance.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# --------------------------------------------------------------
#   Initialize - for cruise distance
# --------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Cruise
def initialize_cruise_distance(segment):
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
    segment.state.unknowns.cruise_distance = distance
    
    return


# --------------------------------------------------------------
#   Unknowns - for cruise distance
# --------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def unknown_cruise_distance(segment):
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
    distance = segment.state.unknowns.cruise_distance
    cruise_tag = segment.cruise_tag
    
    # apply the unknown
    segment.segments[cruise_tag].distance = distance
    
    return


# --------------------------------------------------------------
#   Residuals - for Take Off Weight
# --------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def residual_landing_weight(segment):
    """This is a method that allows your vehicle to land at prescribed landing weight.
    This takes the final weight and compares it against the prescribed landing weight.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.state.segments[-1].conditions.weights.total_mass [kilogram]
    segment.target_landing_weight                            [kilogram]

    Outputs:
    segment.state.residuals.landing_weight                   [kilogram]

    Properties Used:
    N/A
    """      
    
    # unpack
    landing_weight = segment.segments[-1].state.conditions.weights.total_mass[-1]
    target_weight  = segment.target_landing_weight
    
    # this needs to go to zero for the solver to complete
    segment.state.residuals.landing_weight = (landing_weight - target_weight)/target_weight
    
    return



# --------------------------------------------------------------
#   Residuals - for Take Off Weight
# --------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def residual_state_of_charge(segment):
    """This is a method that allows your vehicle to land at a prescribed state of charge.
    This takes the final weight and compares it against the prescribed state of charge.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.state.segments[-1].conditions.propulsion.state_of_charge [None]
    segment.target_state_of_charge                                   [None]

    Outputs:
    segment.state.residuals.state_of_charge                          [None]

    Properties Used:
    N/A
    """      
    
    # unpack
    end_SOC    = segment.segments[-1].state.conditions.propulsion.state_of_charge[-1]
    target_SOC = segment.target_state_of_charge
    
    # this needs to go to zero for the solver to complete
    segment.state.residuals.state_of_charge = (end_SOC - target_SOC)/target_SOC
    
    return
    

    
    