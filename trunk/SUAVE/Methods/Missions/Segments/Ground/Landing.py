def initialize_conditions(segment,state):
    conditions = Ground_Segment.initialize_conditions(self,conditions,numerics,initials)
    
    # process initials
    if initials:
        m_initial = initials.weights.total_mass[0,0]
    else:
        m_initial = self.config.mass_properties.landing
          
    # apply initials
    conditions.weights.total_mass[:,0] = m_initial

    throttle = self.throttle	
    conditions.propulsion.throttle[:,0] = throttle        


    return conditions