

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


import SUAVE
from SUAVE.Core import Units

from time import time

import pylab as plt

#SUAVE.Analyses.Process.verbose = True
import sys
sys.path.append('../Vehicles')
sys.path.append('../B737')
from Boeing_737 import vehicle_setup, configs_setup
from Stopped_Rotor import vehicle_setup as vehicle_setup_SR 

import mission_B737
# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    # Setup for converging on weight
    
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    analyses = mission_B737.analyses_setup(configs)
    mission  = mission_setup(configs,analyses)
    
    configs.finalize()
    analyses.finalize()
    
    results = mission.evaluate()
    results = results.merged()
    
    plot_results(results)
    
    distance_regression = 3966186.5678927945
    distance_calc       = results.conditions.frames.inertial.position_vector[-1,0]
    error_distance      = abs((distance_regression - distance_calc )/distance_regression)
    assert error_distance < 1e-6
    
    error_weight = abs(mission.target_landing_weight - results.conditions.weights.total_mass[-1,0])
    print('landing weight error' , error_weight)
    assert error_weight < 1e-6
    
    
    
    
    # Setup for converging on SOC, using the stopped rotor vehicle
    
    vehicle_SR, analyses_SR = full_setup_SR()
    analyses_SR.finalize()
    mission_SR              = analyses_SR.mission   
    results_SR              = mission_SR.evaluate()
    results_SR              = results_SR.merged()
    
    distance_regression_SR = 126309.83688626593
    distance_calc_SR       = results_SR.conditions.frames.inertial.position_vector[-1,0]
    error_distance_SR      = abs((distance_regression_SR - distance_calc_SR )/distance_regression_SR)
    assert error_distance_SR < 1e-6   
    
    error_soc = abs(mission_SR.target_state_of_charge- results_SR.conditions.propulsion.state_of_charge[-1,0])
    print('landing state of charge error' , error_soc)
    assert error_soc < 1e-6    
    
    
    return
    
    
def mission_setup(configs,analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    
    mission = SUAVE.Analyses.Mission.Vary_Cruise.Given_Weight()
    mission.tag = 'the_mission'
    
    # the cruise tag to vary cruise distance
    mission.cruise_tag = 'cruise'
    mission.target_landing_weight = analyses.base.weights.vehicle.mass_properties.operating_empty
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments    
    
    # base segment
    base_segment = Segments.Segment()
    base_segment.state.numerics.number_control_points = 4
    base_segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability   = SUAVE.Methods.skip    
        
    
    # ------------------------------------------------------------------
    #   Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb"
    
    segment.analyses.extend( analyses.takeoff )
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 5.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = 4000.00 * Units.km
        
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent"

    segment.analyses.extend( analyses.landing )
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']    
    
    mission.append_segment(segment)
    
    return mission


def mission_setup_SR(vehicle,analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    
    mission = SUAVE.Analyses.Mission.Vary_Cruise.Given_State_of_Charge()
    mission.tag = 'the_mission'
    
    # the cruise tag to vary cruise distance
    mission.cruise_tag = 'cruise'
    mission.target_state_of_charge = 0.5
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments    
    
    # base segment
    base_segment = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row    
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability     = SUAVE.Methods.skip    
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.lift_cruise.unpack_unknowns_transition
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.lift_cruise.residuals_transition
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.lift_cruise.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)    
    
        
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses )
    
    segment.altitude  = 1000.0 * Units.ft
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 60.    * Units.miles     
    segment.battery_energy = vehicle.propulsors.lift_cruise.battery.max_energy
    

    segment.state.unknowns.propeller_power_coefficient = 0.16 * ones_row(1)  
    segment.state.unknowns.throttle                    = 0.80 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.lift_cruise.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.lift_cruise.residuals_no_lift    
    
    mission.append_segment(segment)


    return mission


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup_SR():
    
    # vehicle data
    vehicle  = vehicle_setup_SR() 

    # vehicle analyses
    analyses = base_analysis_SR(vehicle)

    # mission analyses
    mission  = mission_setup_SR(vehicle,analyses)

    analyses.mission = mission
    
    return  vehicle, analyses


def base_analysis_SR(vehicle):

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
    weights = SUAVE.Analyses.Weights.Weights_Electric_Lift_Cruise()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

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

    return analyses    

def plot_results(results):
    
    plt.figure('Altitude')
    plt.plot( results.conditions.frames.inertial.position_vector[:,0,None] / Units.km ,
              results.conditions.freestream.altitude / Units.km ,
              'bo-' )
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    
    plt.figure('Angle of Attack')
    plt.plot( results.conditions.frames.inertial.position_vector[:,0,None] / Units.km ,
              results.conditions.aerodynamics.angle_of_attack / Units.deg ,
              'bo-' )
    plt.xlabel('Distance (km)')
    plt.ylabel('Angle of Attack (deg)') 
    
    plt.figure('Weight')
    plt.plot( results.conditions.frames.inertial.position_vector[:,0,None] / Units.km ,
              results.conditions.weights.total_mass / Units.kg ,
              'bo-' )
    plt.xlabel('Distance (km)')
    plt.ylabel('Vehicle Total Mass (kg)')     
    

if __name__ == '__main__':
    main()
    plt.show(block=True)
