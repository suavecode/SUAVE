# concorde.py
# 
# Created:  Aug 2014, SUAVE Team
# Modified: Nov 2016, T. MacDonald
#           Jul 2017, T. MacDonald
#           Aug 2018, T. MacDonald
#           Nov 2018, T. MacDonald

""" setup file for a mission with Concorde
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
# Units allow any units to be specificied with SUAVE then automatically converting them the standard
from SUAVE.Core import Units

# Numpy is use extensively throughout SUAVE
import numpy as np
import copy, time
from SUAVE.Core import Data, Container

# Imports library to plot common figures
from SUAVE.Plots.Mission_Plots import *

# This is a sizing function to fill turbojet parameters
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Methods.Center_of_Gravity.compute_fuel_center_of_gravity_longitudinal_range \
     import compute_fuel_center_of_gravity_longitudinal_range
from SUAVE.Methods.Center_of_Gravity.compute_fuel_center_of_gravity_longitudinal_range \
     import plot_cg_map 

# import vehicle and analyses
import sys
sys.path.append('../Vehicles')
from Concorde import vehicle_setup, configs_setup



# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # First we construct the baseline aircraft
    configs, analyses = full_setup()
    
    # Any sizing functions are included here to size components
    simple_sizing(configs)
    
    # Here we finalize the configuration and analysis settings
    configs.finalize()
    analyses.finalize()
    
    ## Use these scripts to test OpenVSP functionality if desired
    #from SUAVE.Input_Output.OpenVSP.vsp_write import write
    #write(configs.base,'Concorde')

    # These functions analyze the mission
    mission = analyses.missions.base
    results = mission.evaluate()
    
    masses, cg_mins, cg_maxes = compute_fuel_center_of_gravity_longitudinal_range(configs.base)
    plot_cg_map(masses, cg_mins, cg_maxes)  
    
    results.fuel_tank_test = Data()
    results.fuel_tank_test.masses   = masses
    results.fuel_tank_test.cg_mins  = cg_mins
    results.fuel_tank_test.cg_maxes = cg_maxes
    
    # save results 
    #save_results(results)
    
    # load old results
    old_results = load_results()   

    # plt the old results
    plot_mission(results, line_color = 'bo-')
    plot_mission(old_results, line_color = 'k-')
    plt.show()

    # check the results
    check_results(results,old_results) 
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():
    
    # Vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # Vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # Mission analyses
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
    
    analyses = SUAVE.Analyses.Analysis.Container()
    
    # build a base analysis for each config
    for tag,config in list(configs.items()):
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
    aerodynamics.geometry = vehicle
    
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    
    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors #what is called throughout the mission (at every time step))
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
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_color):
    # Plot Flight Conditions 
    plot_flight_conditions(results,line_color)
    
    # Plot Aerodynamic Forces 
    plot_aerodynamic_forces(results,line_color)
    
    # Plot Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results,line_color)
    
    # Drag Components
    plot_drag_components(results,line_color)
    
    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results,line_color)
    
    # Plot Velocities 
    plot_aircraft_velocities(results,line_color)  
 
        
    return

def simple_sizing(configs):
    
    base = configs.base
    base.pull_base()
    
    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 
    
    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers
    
    # diff the new data
    base.store_diff()
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = configs.landing
    
    # make sure base data is current
    landing.pull_base()
    
    # landing weight
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff
    
    # diff the new data
    landing.store_diff()
    
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
    #   First Climb Segment
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    
    segment.analyses.extend( analyses.base )
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle = ones_row(1) * 7. * Units.deg   
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.05   * Units.km
    segment.air_speed      = 128.6 * Units['m/s']
    segment.climb_rate     = 20.32 * Units['m/s']
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude_end   = 4.57   * Units.km
    segment.air_speed      = 205.8  * Units['m/s']
    segment.climb_rate     = 10.16  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Third Climb Segment: linear Mach
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_3"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude_end = 7.60   * Units.km
    segment.mach_start   = 0.64
    segment.mach_end     = 1.0
    segment.climb_rate   = 5.05  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Fourth Climb Segment: linear Mach
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_4"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude_end = 15.24   * Units.km
    segment.mach_start   = 1.0
    segment.mach_end     = 2.02
    segment.climb_rate   = 5.08  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------
    #   Fourth Climb Segment
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_5"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude_end = 18.288   * Units.km
    segment.mach_number  = 2.02
    segment.climb_rate   = 0.65  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Cruise Segment
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses.base )
    
    segment.mach       = 2.02
    segment.distance   = 2000.0 * Units.km
        
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   First Descent Segment
    # ------------------------------------------------------------------    
    
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "descent_1"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude_end = 6.8   * Units.km
    segment.mach_start   = 2.02
    segment.mach_end     = 1.0
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment
    # ------------------------------------------------------------------    
    
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "descent_2"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude_end = 3.0   * Units.km
    segment.mach_start   = 1.0
    segment.mach_end     = 0.65
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------    
    #   Third Descent Segment
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    segment.analyses.extend( analyses.base )
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 130.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
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
    
def check_results(new_results,old_results):

    # check segment values
    check_list = [
        'segments.cruise.conditions.aerodynamics.angle_of_attack',
        'segments.cruise.conditions.aerodynamics.drag_coefficient',
        'segments.cruise.conditions.aerodynamics.lift_coefficient',
        'segments.cruise.conditions.propulsion.throttle',
        'segments.cruise.conditions.weights.vehicle_mass_rate',
        'fuel_tank_test.masses',
        'fuel_tank_test.cg_mins',
        'fuel_tank_test.cg_maxes',
    ]

    # do the check
    for k in check_list:
        print(k)

        old_val = np.max( old_results.deep_get(k) )
        new_val = np.max( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print('Error at Max:' , err)
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k

        old_val = np.min( old_results.deep_get(k) )
        new_val = np.min( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print('Error at Min:' , err)
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        

        print('')


    return


def load_results():
    return SUAVE.Input_Output.SUAVE.load('results_mission_concorde.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_mission_concorde.res')
    return    
        
if __name__ == '__main__': 
    main()    
    plt.show()
        
