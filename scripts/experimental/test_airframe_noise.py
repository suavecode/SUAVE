#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     
#
# Author:      CARIDSIL
#
# Created:     21/07/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# test_noise.py
#
# Created:  Carlos
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports

import SUAVE
from SUAVE.Core import Units

from SUAVE.Methods.Noise.Fidelity_One.Airframe import noise_fidelity_one
from SUAVE.Methods.Noise.Fidelity_One.Engine import noise_SAE
from SUAVE.Methods.Noise.Fidelity_One import flight_trajectory


import numpy as np


from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    configs, analyses = full_setup()

    configs.finalize()
    analyses.finalize()

    #noise
    trajectory = flight_trajectory(configs)
    
    airframe_noise=noise_fidelity_one(configs,analyses,trajectory)
    

    
    print airframe_noise

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    analyses = base_analyses(vehicle)

    return configs, analyses

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing_737800'


    # ------------------------------------------------------------------
    #   Vehicle-level Properties for Airframe Noise Calculation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    #   Landing gear
    # ------------------------------------------------------------------

    vehicle.landing_gear = Data()
    vehicle.landing_gear.main_tire_diameter = 1.0
    vehicle.landing_gear.nose_tire_diameter = 1.0
    vehicle.landing_gear.main_strut_length = 1.0
    vehicle.landing_gear.nose_strut_length = 1.0
    vehicle.landing_gear.number_wheels = 2
    vehicle.landing_gear.gear_condition = 0   #0 for gear up and 1 for gear down


    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.spans.projected         = 35.66

    wing.areas.reference         = 124.862


    # ------------------------------------------------------------------
    #   Flaps
    # ------------------------------------------------------------------
    
    wing.flaps.chord = 1.0
    wing.flaps.area = 1.0
    wing.flaps.number_slots = 2.0


    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.spans.projected         = 14.146      #

    wing.areas.reference         = 32.488    #

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'

    wing.spans.projected         = 7.877      #

    wing.areas.reference         = 32.488    #

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle


def base_analyses(vehicle):

    analyses = SUAVE.Analyses.Vehicle()
 # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses

def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    base_config.wings['main_wing'].flaps.angle = 5. * Units.deg
    configs.append(base_config)

    # done!
   # return configs

    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'flight'

    config.initial_position = -2000*Units.m
    config.initial_time = 0.0
    config.velocity = 130*Units.knots
    config.altitute = 500*Units.ft

    configs.append(config)
    


    return configs


if __name__ == '__main__':
    main()

