# full_setup.py
# 
# Created:  SUave Team, Aug 2014
# Modified: 

""" setup file for a mission with a E190
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

from test_mission_Embraer_E190 import vehicle_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # define the problem
    vehicle, mission = full_setup()
    
    # run the problem
    results = the_aircraft_function(vehicle,mission)
    
    # check the results
    check_results(results)
    
    # post process the results
    plot_mission(vehicle,mission,results)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup() # imported from E190 test script
    mission = mission_setup(vehicle)
    
    return vehicle, mission


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
    
    mission.airport = airport    
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_250KCAS"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 138.0 * Units['m/s']
    segment.climb_rate     = 3000. * Units['ft/min']
    
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
    segment.air_speed    = 200.0  * Units['m/s']
    segment.throttle     = 1.0
    
    # dummy for post process script
    segment.climb_rate   = 0.1
    
    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "CLIMB_Final"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 37000. * Units.ft
    segment.air_speed    = 230.0  * Units['m/s']
    segment.climb_rate   = 200.   * Units['ft/min']

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
    
    segment.air_speed  = 230.  * Units['m/s']
    segment.distance   = 2050. * Units.nmi
    
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
    segment.air_speed    = 230.0 * Units['m/s']
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
    segment.air_speed    = 200.0 * Units['m/s']
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
    segment.air_speed    = 140.0 * Units['m/s']
    segment.descent_rate = 1500. * Units['ft/min']    
    
    # append to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission

#: def define_mission()


def check_results(new_results):
    
    # load old results
    old_results = load_results()
    
    # check segment values
    check_list = [
        'mission_profile.segments.Cruise.conditions.aerodynamics.angle_of_attack',
        'mission_profile.segments.Cruise.conditions.aerodynamics.drag_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.lift_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cm_alpha',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cn_beta',
        'mission_profile.segments.Cruise.conditions.propulsion.throttle',
        'mission_profile.segments.Cruise.conditions.propulsion.fuel_mass_rate',
    ]
    
    # gets a key recursively from a '.' string
    def get_key(data,keys):
        if isinstance(keys,str):
            keys = keys.split('.')
        k = keys.pop(0)
        if keys:
            return get_key(data[k],keys)
        else:
            return data[k]    
    
    # do the check
    for k in check_list:
        print k
        
        old_val = np.max( get_key(old_results,k) )
        new_val = np.max( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Max:' , err
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k
        
        old_val = np.min( get_key(old_results,k) )
        new_val = np.min( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Min:' , err
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        
    
    # check high level outputs
    def check_vals(a,b):
        if isinstance(a,Data):
            for k in a.keys():
                err = check_vals(a[k],b[k])
                if err is None: continue
                print 'outputs' , k
                print 'Error:' , err
                assert np.abs(err) < 1e-6 , 'Outputs Check Failed : %s' % k  
        else:
            return (a-b)/a

    # do the check
    check_vals(old_results.output,new_results.output)
    
    return

def load_results():
    return SUAVE.Plugins.VyPy.data.load('results_mission_E190_constThr.pkl')
    
def save_results(results):
    SUAVE.Plugins.VyPy.data.save(results,'results_mission_E190_constThr.pkl')
    
if __name__ == '__main__': 
    main()
    plt.show()