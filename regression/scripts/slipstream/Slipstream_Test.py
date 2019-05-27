# Cessna_172.py
# 
# Created:  Mar 2019, M. Clarke

""" setup file for a mission with a twin prop modified C172 SP NAV III with 
propeller interaction 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

from SUAVE.Core import Data , Container
from SUAVE.Methods.Propulsion import propeller_design

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():

    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
     
    # lift coefficient  
    lift_coefficient              = results.segments['cruise'].conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true         = 0.41648757412447335 
    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments['cruise'].conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional_lift['main_wing'][2][:]
    sectional_lift_coeff_true       = np.array([0.39688074, 0.39789599, 0.39759175, 0.39598577, 0.39302907,
                                                0.38861399, 0.38256543, 0.37461901, 0.36437972, 0.3512425 ,
                                                0.33422483, 0.31154751, 0.2791921 , 0.21920585, 0.16904757,
                                                0.13540296, 0.12179052, 0.12765071, 0.15139584, 0.19181159,
                                                0.25356333, 0.31326464, 0.37308351, 0.43511796, 0.49967181,
                                                0.56782375, 0.64134112, 0.70135903, 0.74678748, 0.77579888,
                                                0.78678922, 0.77687711, 0.73886236, 0.64294577, 0.59171574,
                                                0.55466008, 0.52468674, 0.49886926, 0.47560758, 0.4538663 ,
                                                0.43288281, 0.41202613, 0.39071187, 0.36833567, 0.34420292,
                                                0.31743009, 0.2867711 , 0.25024397, 0.20411721, 0.13892142])
    print(sectional_lift_coeff)
    diff_Cl                       = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  max(np.abs((sectional_lift_coeff - sectional_lift_coeff_true)/sectional_lift_coeff_true)) < 1e-6 
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup() 
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses) 
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------
def analyses_setup(configs):

    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()      

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # Single point analysis
    
    
    # save results 
    #save_results(results, vec_configs)
    
    # plt the old results
    plot_mission(results)


    return analyses
# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Tube_Wing()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.process.compute.lift.inviscid_wings.settings.use_surrogate = False
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()    
    stability.geometry = vehicle
    analyses.append(stability)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors 
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    # done!
    return analyses    


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Cessna_172_SP'


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # mass properties
    vehicle.mass_properties.max_takeoff   = 2550. * Units.pounds
    vehicle.mass_properties.takeoff       = 2550. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = 2550. * Units.pounds
    vehicle.mass_properties.cargo         = 0.

    # envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.8

    # basic parameters
    vehicle.reference_area         = 174. * Units.feet**2
    vehicle.passengers             = 4

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 174. * Units.feet**2
    wing.spans.projected         = 36.  * Units.feet + 1. * Units.inches

    wing.chords.root             = 66. * Units.inches
    wing.chords.tip              = 45. * Units.inches
    wing.chords.mean_aerodynamic = 58. * Units.inches # Guess
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 1.5 * Units.degrees

    wing.origin                  = [80.* Units.inches,0,0]
    wing.aerodynamic_center      = [22.* Units.inches,0,0]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.95
    wing.areas.reference         = 5800. * Units.inches**2
    wing.spans.projected         = 136.  * Units.inches

    wing.chords.root             = 55. * Units.inches
    wing.chords.tip              = 30. * Units.inches
    wing.chords.mean_aerodynamic = 43. * Units.inches # Guess
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [246.* Units.inches,0,0]
    wing.aerodynamic_center      = [20.* Units.inches,0,0]
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 0.9

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'

    wing.sweeps.quarter_chord    = 25. * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 3500. * Units.inches**2
    wing.spans.projected         = 73.   * Units.inches

    wing.chords.root             = 66. * Units.inches
    wing.chords.tip              = 27. * Units.inches
    wing.chords.mean_aerodynamic = 48. * Units.inches # Guess
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [237.* Units.inches,0,  0.623]
    wing.aerodynamic_center      = [20.* Units.inches,0,0]

    wing.vertical                = True
    wing.symmetric               = False
    wing.t_tail                  = False

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'

    fuselage.seats_abreast         = 2.

    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.

    fuselage.lengths.nose          = 60.  * Units.inches
    fuselage.lengths.tail          = 161. * Units.inches
    fuselage.lengths.cabin         = 105. * Units.inches
    fuselage.lengths.total         = 326. * Units.inches
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.

    fuselage.width                 = 42. * Units.inches

    fuselage.heights.maximum       = 62. * Units.inches
    fuselage.heights.at_quarter_length          = 62. * Units.inches
    fuselage.heights.at_three_quarters_length   = 62. * Units.inches
    fuselage.heights.at_wing_root_quarter_chord = 23. * Units.inches

    fuselage.areas.side_projected  = 8000.  * Units.inches**2.
    fuselage.areas.wetted          = 30000. * Units.inches**2.
    fuselage.areas.front_projected = 42.* 62. * Units.inches**2.

    fuselage.effective_diameter    = 50. * Units.inches


    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #   Piston Propeller Network
    # ------------------------------------------------------------------

    # build network
    net = SUAVE.Components.Energy.Networks.Internal_Combustion_Propeller()
    net.number_of_engines = 2.
    net.nacelle_diameter  = 42 * Units.inches
    net.engine_length     = 0.01 * Units.inches
    net.areas             = Data()
    net.rated_speed       = 2700. * Units.rpm
    net.areas.wetted      = 0.01

    # Component 1 the engine
    net.engine = SUAVE.Components.Energy.Converters.Internal_Combustion_Engine()
    net.engine.sea_level_power    = 180. * Units.horsepower
    net.engine.flat_rate_altitude = 0.0
    net.engine.speed              = 2700. * Units.rpm
    net.engine.BSFC               = 0.52


    # Design the Propeller
    prop  = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = 2.0
    prop.freestream_velocity = 135.*Units['mph']
    prop.angular_velocity    = 1250.  * Units.rpm
    prop.tip_radius          = 76./2. * Units.inches
    prop.hub_radius          = 8.     * Units.inches
    prop.design_Cl           = 0.8
    prop.design_altitude     = 12000. * Units.feet
    prop.design_thrust       = 0.0
    prop.design_power        = .32 * 180. * Units.horsepower
    prop                     = propeller_design(prop)
    prop.origin = [[2.,2.5,0.]]
    net.propeller        = prop

    # add the network to the vehicle
    vehicle.append_component(net)



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
 
     # done!
    return configs

# ----------------------------------------------------------------------
#   Sizing for the Vehicle Configs
# ----------------------------------------------------------------------
def simple_sizing(configs):

    base = configs.base
    base.pull_base()

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 1.75 * wing.areas.reference
        wing.areas.exposed  = 0.8  * wing.areas.wetted
        wing.areas.affected = 0.6  * wing.areas.wetted


    # diff the new data
    base.store_diff()


    # done!
    return

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment() 
    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.base )
    segment.altitude_start = 0.0 * Units.feet
    segment.altitude_end   = 8500. * Units.feet
    segment.air_speed      = 105.  * Units['mph']  
    segment.climb_rate     = 500.  * Units['ft/min']
    
    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend(analyses.base)

    segment.altitude  = 8500. * Units.feet
    segment.air_speed = 132.   *Units['mph']  
    segment.distance  = 50.   * Units.nautical_mile
    
    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Descent Segment: constant Speed, constant rate segment 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "decent"

    segment.analyses.extend( analyses.base )

    segment.altitude_start = 8500. * Units.feet
    segment.altitude_end   = 0.      * Units.feet
    segment.air_speed      = 80.    * Units['mph'] 
    
    segment.climb_rate     = -300.  * Units['ft/min']
    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------
    return mission



def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------

    missions.base = base_mission

    # done!
    return missions  


if __name__ == '__main__': 
    main()    
    plt.show()