# full_setup.py
# 
# Created:  SUave Team, Aug 2014
# Modified: 

""" setup file for a mission with a 737
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# the analysis functions
from the_aircraft_function import the_aircraft_function
from plot_mission import plot_mission


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # define the problem
    vehicle, mission = full_setup()
    
    # run the problem
    results = the_aircraft_function(vehicle,mission)
    
    # plot the new results
    plot_mission(vehicle,mission,results,'bo-')    
    
    # load older results
    old_results = load_results()
    
    # plt the old results
    plot_mission(vehicle,mission,old_results,'k-')
    
    # check the results
    check_results(results,old_results)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup()
    mission = mission_setup(vehicle)
    
    return vehicle, mission


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    """ goal: no scripting """
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing 737-800'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    
    # mass properties
    vehicle.mass_properties.max_takeoff               = 79015.8   # kg
    vehicle.mass_properties.operating_empty           = 62746.4   # kg
    vehicle.mass_properties.takeoff                   = 79015.8   # kg
    #vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff    # !!!!!!!!!!!!!!!!??? BAD BOY
    vehicle.mass_properties.cargo                     = 10000.0   # kg
    
    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5
    
    # basic parameters
    vehicle.reference_area         = 124.862
    vehicle.passengers             = 170
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.aspect_ratio            = 10.18
    wing.sweep                   = 25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.16
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 35.66    
    
    wing.chords.root             = 6.81
    wing.chords.tip              = 1.09
    wing.chords.mean_aerodynamic = 12.5
    
    wing.areas.reference         = 124.862 
    #wing.areas.wetted            = 2.0 * wing.areas.reference  # BAD FOR OPTIMIZATION, make this an analysis
    #wing.areas.exposed           = 0.8 * wing.areas.wetted
    #wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees
    
    wing.origin                  = [20,0,0]
    wing.aerodynamic_center      = [3,0,0] 
    
    wing.vertical                = False
    wing.symmetric               = True
    
    wing.eta                     = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.aspect_ratio            = 6.16      #
    wing.sweep                   = 30 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.4
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 14.146      #

    wing.chords.root             = 3.28
    wing.chords.tip              = 1.31    
    wing.chords.mean_aerodynamic = 8.0

    wing.areas.reference         = 32.488    #
    #wing.areas.wetted            = 2.0 * wing.areas.reference
    #wing.areas.exposed           = 0.8 * wing.areas.wetted
    #wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees  
    
    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]
    
    wing.vertical                = False 
    wing.symmetric               = True
    
    wing.eta                     = 0.9  
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.aspect_ratio            = 1.91      #
    wing.sweep                   = 25 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.25
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 7.877      #    

    wing.chords.root             = 6.60
    wing.chords.tip              = 1.65
    wing.chords.mean_aerodynamic = 8.0
    
    wing.areas.reference         = 32.488    #
    #wing.areas.wetted            = 2.0 * wing.areas.reference
    #wing.areas.exposed           = 0.8 * wing.areas.wetted
    #wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]    
    
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    
    wing.eta                     = 1.0
        
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 6
    fuselage.seat_pitch            = 1
    
    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.
    
    fuselage.lengths.nose          = 6.4
    fuselage.lengths.tail          = 8.0
    fuselage.lengths.cabin         = 44.0
    fuselage.lengths.total         = 58.4    
    fuselage.lengths.fore_space    = 6.
    fuselage.lengths.aft_space     = 5.    
    
    fuselage.width                 = 4.
    
    fuselage.heights.maximum       = 4.    #
    fuselage.heights.at_quarter_length          = 4. # Not correct
    fuselage.heights.at_three_quarters_length   = 4. # Not correct
    fuselage.heights.at_wing_root_quarter_chord = 4. # Not correct

    fuselage.areas.side_projected  = 4.* 59.8 #  Not correct
    fuselage.areas.wetted          = 688.64
    fuselage.areas.front_projected = 12.57
    
    fuselage.effective_diameter    = 4.0
    
    fuselage.differential_pressure = 10**5 * Units.pascal    # Maximum differential pressure
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'
    
    turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()
    
    #turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.98     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.99     #
    turbofan.lpc_pressure_ratio            = 1.14     #
    turbofan.hpc_pressure_ratio            = 13.415   #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1450.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.thrust.design                 = 25000.0  #
    turbofan.number_of_engines             = 2.0      #
    
    # size the turbofan
    turbofan.A2          =   1.753
    turbofan.df          =   1.580
    turbofan.nacelle_dia =   1.580
    turbofan.A2_5        =   0.553
    turbofan.dhc         =   0.857
    turbofan.A7          =   0.801
    turbofan.A5          =   0.191
    turbofan.Ao          =   1.506
    turbofan.mdt         =   9.51
    turbofan.mlt         =  22.29
    turbofan.mdf         = 355.4
    turbofan.mdlc        =  55.53
    turbofan.D           =   1.494
    turbofan.mdhc        =  49.73  
    
    # add to vehicle
    vehicle.append_component(turbofan)    
    
    
    # done!
    return vehicle

#: def vehicle_setup()
    
    
def vehicle_finalize(vehicle):
    
    vehicle.mass_properties.max_zero_fuel = 0.9 * vehicle.mass_properties.max_takeoff
    
    for wing in vehicle.wings:
        wing.areas.wetted            = 2.0 * wing.areas.reference  
        wing.areas.exposed           = 0.8 * wing.areas.wetted
        wing.areas.affected          = 0.6 * wing.areas.wetted    
    
    config = configs.landing    
    config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff    # !!!!!!!!!!??????????????
    
    return vehicle
    
# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------
    
def configs_setup():
    """ goal: increment on top of a base vehicle config, expand in finalize() """
    ######## PROBLEM: Deleting elements?,, don't allow for now.  can overwrite lists though

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Configs()
    
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config()
    config.tag = 'cruise'
    
    configs.append(config)
    
    
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config()
    config.tag = 'takeoff'
    
    config.wings['Main Wing'].flaps_angle = 20. * Units.deg
    config.wings['Main Wing'].slats_angle = 25. * Units.deg
    
    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config()
    config.tag = 'landing'
    
    config.wings['Main Wing'].flaps_angle = 30. * Units.deg
    config.wings['Main Wing'].slats_angle = 25. * Units.deg

    config.Vref_VS_ratio = 1.23
    config.maximum_lift_coefficient = 2.
    
    #config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff    # !!!!!!!!!!??????????????
    
    configs.append(config)
    
    # done!
    return configs

#: def configs_setup()


# ----------------------------------------------------------------------
#   Analyses Setup
# ----------------------------------------------------------------------

def analyses_setup():
    """ goal: be independent of vehicle """

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    
    analyses = SUAVE.Attributes.Analyses()
    
    """ should have at least the following defaulted, and has a smart append to assign
    analyses.aerodynamics
    analyses.propulsion
    analyses.energy
    analyses.weights
    analyses.structures
    analyses.loads
    analyses.stability
    """
    
    # ------------------------------------------------------------------
    #   Aerodynamics Analysis
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.configuration.drag_coefficient_increment = 0.0000
    
    analyses.append(aerodynamics)
    # updates analyses.aerodynamics

    # ------------------------------------------------------------------
    #   Stability Analysis
    # ------------------------------------------------------------------ 
    
    stability = SUAVE.Analyses.Flight_Dynamics.Fidelity_Zero()
    
    analyses.append(stability)
    
    # ------------------------------------------------------------------
    #   Energy Analysis
    # ------------------------------------------------------------------     
    
    energy = SUAVE.Analyses.Energy.Networks.Solar_Network()
    
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #   Propulsion Analysis
    # ------------------------------------------------------------------     
    
    propulsion = SUAVE.Analyses.Propulsion.Fidelity_Zero()
    
    # TODO: figure out relationship with energy network
    
    # ------------------------------------------------------------------
    #   Sizing
    # ------------------------------------------------------------------     

    sizing = SUAVE.Analyses.Geometry.Sizing.Fidelity_Zero()
    
    analyses.append(sizing)
    
    
    # done!
    return vehicle    

#: def analyses_setup()


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(configs,analyses):
    """ again, independent of vehicle """
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'The Test Mission'

    # atmospheric model
    planet     = SUAVE.Attributes.Planets.Earth()
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = atmosphere                                         #!!!!!! Just Barely OK
    
    mission.airport = airport
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 1"
    
    # connect vehicle configuration
    segment.config         = configs.takeoff
    
    # connect analyses setup
    segment.analyses       = analyses
    
    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    #segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Mach_Constant_Rate()
    segment.tag = "Climb - 2"
    
    # connect vehicle configuration
    segment.config       = configs.takeoff
    
    # connect analyses setup
    segment.analyses     = analyses
    
    # define segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet 
    
    #segment.altitude_start = 3.0   * Units.km ## Optional
    segment.altitude_end = 8.0   * Units.km
    segment.air_speed    = 190.0 * Units['m/s']
    segment.climb_rate   = 6.0   * Units['m/s']
    #segment.mach_number  = 0.5
    #segment.climb_rate   = 6.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 3"
    
    # connect vehicle configuration
    segment.config       = configs.takeoff
    
    # connect analyses setup
    segment.analyses     = analyses
    
    # define segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet       
    
    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config       = configs.takeoff
    
    # connect analyses setup
    segment.analyses     = analyses
    
    # define segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    #segment.altitude     = 10.668  * Units.km     # Optional
    segment.air_speed    = 230.412 * Units['m/s']
    segment.distance     = 3933.65 * Units.km
        
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config       = configs.takeoff
    
    # connect analyses setup
    segment.analyses     = analyses
    
    # define segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet 
    
    segment.altitude_end = 5.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 2"

    # connect vehicle configuration
    segment.config       = configs.takeoff
    
    # connect analyses setup
    segment.analyses     = analyses
    
    # define segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet    
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)
    

    # done!
    return mission

#: def mission_setup()

    
if __name__ == '__main__': 
    main()
    plt.show()