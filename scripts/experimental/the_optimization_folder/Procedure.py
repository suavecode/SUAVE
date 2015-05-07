# Procedure.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units

# ----------------------------------------------------------------------        
#   Sizing
# ----------------------------------------------------------------------    

def simple_sizing(configs):
    
    base = configs.base
    base.pull_base()
    
    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 
    
    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted
    
    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers
    
    # diff the new data
    base.store_diff()
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = configs.landing
    
    # make sure base data is current
    landing.pull_base()
    
    # landing weight
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff
    
    # diff the new data
    landing.store_diff()
    
    # done!
    return

def post_process(interface):
    
    # Unpack data
    vehicle               = interface.configs.base    
    results               = interface.results
    mission_profile       = results.missions.base    
    short_field_results   = results.short_field
    takeoff_field_results = results.takeoff_field_length
    range_results         = results.max_range
    # Weights
    max_zero_fuel     = vehicle.mass_properties.max_zero_fuel
    operating_empty   = vehicle.mass_properties.operating_empty
    payload           = vehicle.mass_properties.payload   
    
    return results