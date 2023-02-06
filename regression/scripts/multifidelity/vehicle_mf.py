# Vehicles.py
# 
# Created:  May 2019, T. MacDonald
# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import MARC

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def setup():
    
    base_vehicle = base_setup()
    configs = configs_setup(base_vehicle)
    
    return configs
    
    
def base_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = MARC.Vehicle()
    vehicle.tag = 'Multifidelity'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.x1 = 0
    vehicle.x2 = 0

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle




# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = MARC.Components.Configs.Config.Container()

    base_config = MARC.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # done!
    return configs
