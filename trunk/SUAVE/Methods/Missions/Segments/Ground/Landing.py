import Common

def initialize_conditions(segment,state):
    
    Common.initialize_conditions(segment,state)
    
    ## process initials
    #if initials:
        #m_initial = initials.weights.total_mass[0,0]
    #else:
    m_initial = segment.analyses.weights.vehicle.mass_properties.landing
          
    # apply initials
    conditions = state.conditions
    conditions.weights.total_mass[:,0] = m_initial

    throttle = segment.throttle	
    conditions.propulsion.throttle[:,0] = throttle        


    return conditions