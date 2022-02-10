

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


import SUAVE
from SUAVE.Core import Units, Data

from time import time

import pylab as plt

import scipy as sp
import numpy as np


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
    
    distance_regression = 3964052.0508565805
    distance_calc       = results.conditions.frames.inertial.position_vector[-1,0]
    print('distance_calc = ', distance_calc)
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
    
    distance_regression_SR = 101644.15747038861

    distance_calc_SR       = results_SR.conditions.frames.inertial.position_vector[-1,0]
    print('distance_calc_SR = ', distance_calc_SR)
    error_distance_SR      = abs((distance_regression_SR - distance_calc_SR )/distance_regression_SR)
    assert error_distance_SR < 1e-6   
    
    error_soc = abs(mission_SR.target_state_of_charge- results_SR.conditions.propulsion.battery_state_of_charge[-1,0])
    print('landing state of charge error' , error_soc)
    assert error_soc < 1e-6    
    
    
    return


def find_propeller_max_range_endurance_speeds(analyses,altitude,CL_max,up_bnd,delta_isa):
    
    
    # setup a mission that runs a single point segment without propulsion
    def mini_mission():
        
        # ------------------------------------------------------------------
        #   Initialize the Mission
        # ------------------------------------------------------------------
        mission = SUAVE.Analyses.Mission.Sequential_Segments()
        mission.tag = 'the_mission'
        
        # ------------------------------------------------------------------
        #  Single Point Segment 1: constant Speed, constant altitude
        # ------------------------------------------------------------------ 
        segment = SUAVE.Analyses.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_No_Propulsion()
        segment.tag = "single_point" 
        segment.analyses.extend(analyses) 
        segment.altitude    = altitude
        segment.air_speed   = 100.
        segment.temperature_deviation = delta_isa
        segment.state.numerics.tolerance_solution = 1e-6
        segment.state.numerics.max_evaluations    = 500
    
        # add to misison
        mission.append_segment(segment)    

        return mission
        
    
    # This is what's called by the optimizer for CL**3/2 /CD Max
    def single_point_3_halves(X):
        
        # Update the mission
        mission.segments.single_point.air_speed = X
        mission.segments.single_point.state.unknowns.body_angle = np.array([[15.0]]) * Units.degrees
        
        # Run the Mission      
        point_results = mission.evaluate()    
                
        CL = point_results.segments.single_point.conditions.aerodynamics.lift_coefficient
        CD = point_results.segments.single_point.conditions.aerodynamics.drag_coefficient
        
        three_halves = -(CL**(3/2))/CD # Negative because optimizers want to make things small
        
        if not point_results.segments.single_point.converged:
            three_halves = 1.      
        
        return three_halves
    
    
    
    # This is what's called by the optimizer for L/D Max
    def single_point_LDmax(X):
        
        # Modify the mission for the next iteration
        mission.segments.single_point.air_speed = X
        mission.segments.single_point.state.unknowns.body_angle = np.array([[15.0]]) * Units.degrees
        
        # Run the Mission      
        point_results = mission.evaluate()    
                
        CL = point_results.segments.single_point.conditions.aerodynamics.lift_coefficient
        CD = point_results.segments.single_point.conditions.aerodynamics.drag_coefficient
        
        L_D = -CL/CD # Negative because optimizers want to make things small
        
        if not point_results.segments.single_point.converged:
            L_D = 1.

        return L_D


    # ------------------------------------------------------------------
    #   Run the optimizer to solve
    # ------------------------------------------------------------------    
    
    # Setup the a mini mission
    mission = mini_mission()
    
    # Takeoff mass:
    mass = analyses.aerodynamics.geometry.mass_properties.takeoff
    
    # Calculate the stall speed
    Vs = stall_speed(analyses,mass,CL_max,altitude,delta_isa)[0][0]
    
    # The final results to save
    results = Data()
    
    # Wrap an optimizer around both functions to solve for CL**3/2 /CD max
    outputs_32 = sp.optimize.minimize_scalar(single_point_3_halves,bounds=(Vs,up_bnd),method='bounded')    
    
    # Pack the results
    results.cl32_cd = Data()
    results.cl32_cd.air_speed = outputs_32.x
    results.cl32_cd.cl32_cd   = -outputs_32.fun[0][0]
    
    # Wrap an optimizer around both functions to solve for L/D Max
    outputs_ld = sp.optimize.minimize_scalar(single_point_LDmax,bounds=(Vs,up_bnd),method='bounded')    
        
    # Pack the results
    results.ld_max = Data()
    results.ld_max.air_speed = outputs_ld.x
    results.ld_max.L_D_max   = -outputs_ld.fun[0][0]    
  
    return results


def stall_speed(analyses,mass,CL_max,altitude,delta_isa):
    
    # Unpack
    atmo  = analyses.atmosphere 
    S     = analyses.aerodynamics.geometry.reference_area
    
    # Calculations
    atmo_values       = atmo.compute_values(altitude,delta_isa)
    rho               = atmo_values.density
    sea_level_gravity = atmo.planet.sea_level_gravity
    
    W = mass*sea_level_gravity 

    V = np.sqrt(2*W/(rho*S*CL_max))
    
    return V
    
    
    
def mission_setup(configs,analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    
    mission = SUAVE.Analyses.Mission.Variable_Range_Cruise.Given_Weight()
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
    
    mission = SUAVE.Analyses.Mission.Variable_Range_Cruise.Given_State_of_Charge()
    mission.tag = 'the_mission'
    
    # the cruise tag to vary cruise distance
    mission.cruise_tag = 'cruise'
    mission.target_state_of_charge = 0.51
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments    
    
    # base segment
    base_segment = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row    
    base_segment.state.numerics.number_control_points        = 2
    base_segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability     = SUAVE.Methods.skip    
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip    
    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses )
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 40.  * Units.ft
    segment.climb_rate                                       = 500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.networks.lift_cruise.battery.max_energy
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip  
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment,\
                                                                                    initial_lift_rotor_power_coefficient = 0.01,
                                                                                    initial_throttle_lift = 0.9)
    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses )
    
    segment.altitude  = 1000.0 * Units.ft    
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 60.    * Units.miles     
    segment.state.unknowns.throttle = 0.80 * ones_row(1)
    
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment,initial_prop_power_coefficient=0.16)

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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL() 
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
    energy.network = vehicle.networks 
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
