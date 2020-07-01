# DHC6_TwinOtter_wing_only.py
# 
# Created:  Nov 2018, Stanislav Karpuk
# Modified: 
"""
    Vehicle set-up used for the V-n diagram test only
    Data obtained from: Jane's all aircraft 
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
          
    vehicle 		= SUAVE.Vehicle()
    vehicle.tag 	= 'DHC6_TwinOtter'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    
    # mass properties
    vehicle.mass_properties.max_takeoff               = 5670.0 * Units.kilogram      
    vehicle.mass_properties.takeoff                   = 5670.0 * Units.kilogram   
  
    
    # envelope properties
    vehicle.envelope.category                   = 'commuter'
    vehicle.envelope.FAR_part_number 	        = 23
    vehicle.envelope.limit_loads.positive       = 3
    vehicle.envelope.limit_loads.negative       = -1
    vehicle.envelope.cruise_mach                = 0.3

    # aerodynamic properties
    vehicle.maximum_lift_coefficient = 1.4
    vehicle.minimum_lift_coefficient = -1.24

    # basic parameters
    vehicle.reference_area         = 39.00 * Units['meters**2']  
    vehicle.passengers             = 1

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 10.0
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.16
    wing.taper                   = 1.0
    wing.spans.projected         = 19.81 * Units.meter
    wing.chords.mean_aerodynamic = 1.98 * Units.meter
    wing.areas.reference         = 39.00 * Units['meters**2']  
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
