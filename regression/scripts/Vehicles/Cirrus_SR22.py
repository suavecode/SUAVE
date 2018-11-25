# SR22.py
# 
# Created:  Nov 2018, Stanislav Karpuk
# Modified: 
"""
    Vehicle set-up used for the V-n diagram test only
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# Python Imports
import numpy as np

# SUAVE Imports
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Core import (
    Data, Container,
)


def vehicle_setup():
          
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Cirrus SR-22' 
    vehicle.file_tag = 'SR22'   
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # vehicle category
    vehicle.category = 'normal'
    
    # mass properties
    vehicle.mass_properties.max_takeoff               = 598.7 * Units.kilogram 
    vehicle.mass_properties.takeoff                   = 598.7 * Units.kilogram   
    vehicle.mass_properties.cargo                     =  59.0  * Units.kilogram   
    
    # envelope properties
    vehicle.envelope.FARflag = 23
    vehicle.envelope.pos_limit_load    = 1.5
    vehicle.envelope.neg_limit_load    = -1
    vehicle.envelope.cruise_mach = np.array([0.16])

    # aerodynamic properties
    vehicle.maximum_lift_coefficient = 1.45
    vehicle.minimum_lift_coefficient = -1.0

    # basic parameters
    vehicle.reference_area         = 12.077 * Units['meters**2']  
    vehicle.passengers             = 3

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 11.1
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.14
    wing.taper                   = 0.5
    wing.spans.projected         = 11.68 * Units.meter
    wing.chords.mean_aerodynamic = 1.042 * Units.meter
    wing.areas.reference         = 12.077 * Units['meters**2']  
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.dynamic_pressure_ratio  = 1.0


    # add to vehicle
    vehicle.append_component(wing)
    
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
    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    configs.append(config)

    return configs  
