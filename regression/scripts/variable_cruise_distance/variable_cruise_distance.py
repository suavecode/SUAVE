

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import mission_B737

import SUAVE
from SUAVE.Core import Units

from time import time

import pylab as plt

#SUAVE.Analyses.Process.verbose = True

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    vehicle  = mission_B737.vehicle_setup()
    configs  = mission_B737.configs_setup(vehicle)
    analyses = mission_B737.analyses_setup(configs)
    mission  = mission_setup(configs,analyses)
    
    vehicle.mass_properties.takeoff = 70000 * Units.kg
    
    configs.finalize()
    analyses.finalize()
    
    results = mission.evaluate()
    results = results.merged()
    
    plot_results(results)
    
    error = abs(mission.target_landing_weight - results.conditions.weights.total_mass[-1,0])
    print 'landing weight error' , error
    assert error < 1.
    
    return
    
    
def mission_setup(configs,analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    
    #mission = SUAVE.Analyses.Mission.Sequential_Segments()
    #mission = SUAVE.Analyses.Mission.All_At_Once()
    mission = SUAVE.Analyses.Mission.Vary_Cruise.Given_Weight()
    mission.tag = 'the_mission'
    
    # the cruise tag to vary cruise distance
    mission.cruise_tag = 'cruise'
    mission.target_landing_weight = 40000.0 * Units.kg
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
        
    
    # ------------------------------------------------------------------
    #   Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate()
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
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = 2000.00 * Units.km
        
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
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
