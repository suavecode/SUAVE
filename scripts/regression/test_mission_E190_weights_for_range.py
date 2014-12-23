# full_setup.py
#
# Created:  SUave Team, Aug 2014
# Modified:

""" setup file for a mission with a E190
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Structure import (Data)
from SUAVE.Attributes import Units

#  package imports
import numpy as np

# the analysis functions
from SUAVE.Methods.Performance  import size_weights_given_mission_range

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # define the problem
    vehicle, mission = full_setup()

    # run the problem
    # defining inputs of the module
    cruise_segment_tag = 'Cruise'
    mission_payload =             [11792. , 9868. ,    10. ]
    target_range =  np.multiply ( [ 3470. , 3158. ,  3804. ] , 1. * Units.nautical_mile )

    # call module
    distance,fuel,tow = size_weights_given_mission_range(vehicle,mission,cruise_segment_tag,mission_payload,target_range)

    # check the results
    results = Data()
    results.dist = distance
    results.fuel = fuel
    results.tow  = tow

    check_results(results)

    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup() # imported from E190 test script
    mission = mission_setup(vehicle)

    return vehicle, mission


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer_E190'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # mass properties
    vehicle.mass_properties.max_takeoff               = 50480.0   # kg
    vehicle.mass_properties.operating_empty           = 27817.0   # kg
    vehicle.mass_properties.takeoff                   = 50480.0   # kg
    vehicle.mass_properties.max_zero_fuel             = 39610.0   # kg
    vehicle.mass_properties.cargo                     = 0.0 * Units.kilogram
    vehicle.mass_properties.max_payload               = 11792. * Units.kilogram
    vehicle.mass_properties.max_fuel                  = 12795.

    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct

    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 92.00
    vehicle.passengers             = 114
    vehicle.systems.control        = "fully powered"
    vehicle.systems.accessories    = "medium range"

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'

    wing.aspect_ratio            = 8.4
    wing.sweep                   = 22.0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.16
    wing.span_efficiency         = 1.0

    wing.spans.projected         = 27.8

    wing.chords.root             = 5.7057
    wing.chords.tip              = 0.9129
    wing.chords.mean_aerodynamic = 3.8878

    wing.areas.reference         = 92.0
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted

    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

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

    wing.aspect_ratio            = 5.5
    wing.sweep                   = 34.5 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.11
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 11.958

    wing.chords.root             = 3.9175
    wing.chords.tip              = 0.4309
    wing.chords.mean_aerodynamic = 2.6401

    wing.areas.reference         = 26.0
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted

    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 2.0 * Units.degrees

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

    wing.aspect_ratio            = 1.7      #
    wing.sweep                   = 25 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 0.10
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 5.2153     #

    wing.chords.root             = 5.5779
    wing.chords.tip              = 0.5577
    wing.chords.mean_aerodynamic = 3.7524

    wing.areas.reference         = 16.0    #
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]

    wing.vertical                = True
    wing.symmetric               = False

    wing.eta                     = 1.0

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'

    fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 4
    fuselage.seat_pitch            = 0.7455

    fuselage.fineness.nose         = 2.0
    fuselage.fineness.tail         = 3.0

    fuselage.lengths.nose          = 6.0
    fuselage.lengths.tail          = 9.0
    fuselage.lengths.cabin         = 21.24
    fuselage.lengths.total         = 36.24
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.

    fuselage.width                 = 3.0

    fuselage.heights.maximum       = 3.4    #
    fuselage.heights.at_quarter_length          = 3.4 # Not correct
    fuselage.heights.at_three_quarters_length   = 3.4 # Not correct
    fuselage.heights.at_vertical_root_quarter_chord = 3.4 # Not correct

    fuselage.areas.side_projected  = 239.20
    fuselage.areas.wetted          = 327.01
    fuselage.areas.front_projected = 8.0110

    fuselage.effective_diameter    = 3.2

    fuselage.differential_pressure = 10**5 * Units.pascal    # Maximum differential pressure

    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------

    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'

    turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()

    turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.99     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.98     #
    turbofan.lpc_pressure_ratio            = 1.9      #
    turbofan.hpc_pressure_ratio            = 10.0     #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1500.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.thrust.design                 = 20300.0  #
    turbofan.number_of_engines                 = 2.0      #

    # turbofan sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()

    sizing_segment.M   = 0.78          #
    sizing_segment.alt = 10.668         #
    sizing_segment.T   = 223.0        #
    sizing_segment.p   = 0.265*10**5  #

    # size the turbofan
    turbofan.engine_sizing_1d(sizing_segment)

    # add to vehicle
    vehicle.append_component(turbofan)

    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------

    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    aerodynamics.configuration.drag_coefficient_increment = 0.0000
    aerodynamics.initialize(vehicle)

    # build stability model
    stability = SUAVE.Attributes.Flight_Dynamics.Fidelity_Zero()
    stability.initialize(vehicle)
    aerodynamics.stability = stability
    vehicle.aerodynamics_model = aerodynamics

    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------

    vehicle.propulsion_model = vehicle.propulsors

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.configs.takeoff

    # --- Takeoff Configuration ---
    takeoff_config = vehicle.configs.takeoff

    takeoff_config.wings['Main Wing'].flaps_angle = 20. * Units.deg
    takeoff_config.wings['Main Wing'].slats_angle = 25. * Units.deg

    takeoff_config.V2_VS_ratio = 1.21
    takeoff_config.maximum_lift_coefficient = 2.
    #takeoff_config.max_lift_coefficient_factor = 1.0

    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")

    landing_config.wings['Main Wing'].flaps_angle = 30. * Units.deg
    landing_config.wings['Main Wing'].slats_angle = 25. * Units.deg

    landing_config.Vref_VS_ratio = 1.23
    landing_config.maximum_lift_coefficient = 2.
    #landing_config.max_lift_coefficient_factor = 1.0

    landing_config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'EMBRAER_E190AR test mission'

    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    airport_short_field = SUAVE.Attributes.Airports.Airport()
    airport_short_field.altitude     =  0.0  * Units.ft
    airport_short_field.delta_isa    =  0.0
    airport_short_field.field_lenght = 1400. * Units.m
    airport_short_field.atmosphere   = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport
    mission.airport_short_field  = airport_short_field

    mission.range_for_fuel = 600. * Units.nautical_mile


    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = "CLIMB_250KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff

    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 250.0 * Units.knots
    segment.throttle       = 1.0

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = "CLIMB_280KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 32000. * Units.ft
    segment.air_speed    = 350.0  * Units.knots
    segment.throttle     = 1.0

    # dummy for post process script
    segment.climb_rate   = 0.1

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = "CLIMB_Final"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 35000. * Units.ft
    segment.air_speed    = 391.0  * Units.knots
    segment.throttle     = 1.0

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 450. * Units.knots #230.  * Units['m/s']
    ## 35kft:
    # 415. => M = 0.72
    # 450. => M = 0.78
    # 461. => M = 0.80
    ## 37kft:
    # 447. => M = 0.78
    segment.distance   = 2500. * Units.nmi

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_M0.77"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 9.31  * Units.km
    segment.air_speed    = 440.0 * Units.knots
    segment.descent_rate = 2600. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_290KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 365.0 * Units.knots
    segment.descent_rate = 2300. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_250KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1500. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Mission definition complete
    # ------------------------------------------------------------------

    return mission


def check_results(new_results):

    # check segment values
    truth = Data()

    truth.fuel = [ 14246. , 12234. , 12162.]
    truth.tow  = [ 53855. , 49919. , 39989.]

    error = Data()
    error.fuel = np.max(np.abs(new_results.fuel-truth.fuel))
    error.tow  = np.max(np.abs(new_results.tow-truth.tow))

    for k,v in error.items():
        assert(np.abs(v)<0.5)


if __name__ == '__main__':
    main()
