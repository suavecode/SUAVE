# Variable_Cruise_Distance.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# --------------------------------------------------------------
#   Initialize - for cruise distance
# --------------------------------------------------------------

def initialize_cruise_distance(segment,state):
    
    # unpack
    cruise_tag = segment.cruise_tag
    distance   = segment.segments[cruise_tag].distance
    
    # apply, make a good first guess
    state.unknowns.cruise_distance = distance
    
    return


# --------------------------------------------------------------
#   Unknowns - for cruise distance
# --------------------------------------------------------------

def unknown_cruise_distance(segment,state):
    
    # unpack
    distance = state.unknowns.cruise_distance
    cruise_tag = segment.cruise_tag
    
    # apply the unknown
    segment.segments[cruise_tag].distance = distance
    
    return


# --------------------------------------------------------------
#   Residuals - for Take Off Weight
# --------------------------------------------------------------

def residual_landing_weight(segment,state):
    
    # unpack
    landing_weight = state.segments[-1].conditions.weights.total_mass[-1]
    target_weight  = segment.target_landing_weight
    
    # this needs to go to zero for the solver to complete
    state.residuals.landing_weight = landing_weight - target_weight
    
    return
    
    