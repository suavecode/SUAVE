

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

import mission_B737
# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
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
