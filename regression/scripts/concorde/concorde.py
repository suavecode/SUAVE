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
# Scipy is required here for integration functions used in post processing
import scipy as sp
from scipy import integrate

# Post processing plotting tools are imported here
import pylab as plt

# copy is used to copy variable that should not be linked
# time is used to measure run time if needed
import copy, time

# More basic SUAVE function
from SUAVE.Core import (
Data, Container,
)

import sys
sys.path.append('../Vehicles')
from Concorde import vehicle_setup, configs_setup


# This is a sizing function to fill turbojet parameters
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Methods.Center_of_Gravity.compute_fuel_center_of_gravity_longitudinal_range \
     import compute_fuel_center_of_gravity_longitudinal_range
from SUAVE.Methods.Center_of_Gravity.compute_fuel_center_of_gravity_longitudinal_range \
     import plot_cg_map 

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
    
    # load older results
    #save_results(results)
    old_results = load_results()   

    # plt the old results
    plot_mission(results)
    plot_mission(old_results,'k-')
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()
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

def plot_mission(results,line_style='bo-'):

    # ------------------------------------------------------------------
    #   Throttle
    # ------------------------------------------------------------------
    plt.figure("Throttle History")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.segments[i].conditions.propulsion.throttle[:,0]
        axes.plot(time, eta, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)


    # ------------------------------------------------------------------
    #   Angle of Attack
    # ------------------------------------------------------------------

    plt.figure("Angle of Attack History")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)


    # ------------------------------------------------------------------
    #   Fuel Burn Rate
    # ------------------------------------------------------------------
    plt.figure("Fuel Burn Rate")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mdot = results.segments[i].conditions.weights.vehicle_mass_rate[:,0]
        axes.plot(time, mdot, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Fuel Burn Rate (kg/s)')
    axes.grid(True)


    # ------------------------------------------------------------------
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.segments[i].conditions.freestream.altitude[:,0] / Units.km
        axes.plot(time, altitude, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)


    # ------------------------------------------------------------------
    #   Vehicle Mass
    # ------------------------------------------------------------------
    plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)


    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces")
    for segment in list(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(4,1,1)
        axes.plot( time , Lift , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Lift (N)')
        axes.grid(True)

        axes = fig.add_subplot(4,1,2)
        axes.plot( time , Drag , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag (N)')
        axes.grid(True)

        axes = fig.add_subplot(4,1,3)
        axes.plot( time , Thrust , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Thrust (N)')
        axes.grid(True)

        LD = Lift/Drag
        axes = fig.add_subplot(4,1,4)
        axes.plot( time , LD , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('L/D')
        axes.grid(True)            

    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients")
    for segment in list(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL')
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , CDrag , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CD')
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Drag   , line_style )
        if line_style == 'bo-':
            axes.plot( time , Thrust , 'ro-' )      
        else:
            axes.plot( time , Thrust , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag and Thrust (N)')
        axes.grid(True)


    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Drag Components")
    axes = plt.gca()
    for i, segment in enumerate(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

        if line_style == 'bo-':
            axes.plot( time , cdp , 'ko-', label='CD_P' )
            axes.plot( time , cdi , 'bo-', label='CD_I' )
            axes.plot( time , cdc , 'go-', label='CD_C' )
            axes.plot( time , cdm , 'yo-', label='CD_M' )
            axes.plot( time , cd  , 'ro-', label='CD'   )
            if i == 0:
                axes.legend(loc='upper center')            
        else:
            axes.plot( time , cdp , line_style )
            axes.plot( time , cdi , line_style )
            axes.plot( time , cdc , line_style )
            axes.plot( time , cdm , line_style )
            axes.plot( time , cd  , line_style )            

    axes.set_xlabel('Time (min)')
    axes.set_ylabel('CD')
    axes.grid(True)
    
    # ------------------------------------------------------------------    
    #   Concorde Debug
    # ------------------------------------------------------------------
     
    fig = plt.figure("Velocity and Density")
    dist_base = 0.0
    for segment in list(results.segments.values()):
            
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        velocity   = segment.conditions.freestream.velocity[:,0]
        density   = segment.conditions.freestream.density[:,0]
        mach_number   = segment.conditions.freestream.mach_number[:,0]
        
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , velocity , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Velocity (m/s)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mach_number , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Mach')
        axes.grid(True)
        
        distance = np.array([dist_base] * len(time))
        distance[1:] = integrate.cumtrapz(velocity*1.94,time/60.0)+dist_base
        dist_base = distance[-1]
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , distance , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Distance (nmi)')    

    # ------------------------------------------------------------------    
    #   SUAVE Paper Chart
    # ------------------------------------------------------------------
    
    # This is chart provided in paper presented at Aviation 2015

    fig = plt.figure("SUAVE Paper Chart")
    fig.set_figheight(10)
    fig.set_figwidth(6.5)
    axes = fig.add_subplot(3,1,1)
    for segment in list(results.segments.values()):             
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        altitude  = segment.conditions.freestream.altitude[:,0] / Units.km
        axes.plot( time , altitude , line_style )
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)   
    
    axes = fig.add_subplot(3,1,2)
    for segment in list(results.segments.values()): 
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        mach_number   = segment.conditions.freestream.mach_number[:,0]
        axes.plot( time , mach_number , line_style )
    axes.set_xlabel('Time (min)')
    axes.set_ylabel('Mach')
    axes.grid(True)  
    
    axes = fig.add_subplot(3,1,3)
    for segment in list(results.segments.values()): 
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        LD = Lift/Drag
        axes.plot( time , LD , line_style )
    axes.set_xlabel('Time (min)')
    axes.set_ylabel('L/D')
    axes.grid(True)   
    #plt.savefig('Concorde_data.pdf')
        
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
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    
    segment.analyses.extend( analyses.climb )
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 4000. * Units.ft
    segment.airpseed       = 250.  * Units.kts
    segment.climb_rate     = 4000. * Units['ft/min']
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 8000. * Units.ft
    segment.airpseed     = 250.  * Units.kts
    segment.climb_rate   = 2000. * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_2"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 33000. * Units.ft
    segment.mach_start   = .45
    segment.mach_end     = 0.95
    segment.climb_rate   = 3000. * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Third Climb Segment: linear Mach, constant segment angle 
    # ------------------------------------------------------------------    
      
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_3"
    
    segment.analyses.extend( analyses.climb )
    
    segment.altitude_end = 34000. * Units.ft
    segment.mach_start   = 0.95
    segment.mach_end     = 1.0
    segment.climb_rate   = 2000.  * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment) 

    # ------------------------------------------------------------------
    #   Third Climb Segment: linear Mach, constant segment angle 
    # ------------------------------------------------------------------    
      
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_4"
    
    segment.analyses.extend( analyses.climb )
    
    segment.altitude_end = 40000. * Units.ft
    segment.mach_start   = 1.0
    segment.mach_end     = 1.7
    segment.climb_rate   = 1750.  * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Fourth Climb Segment: linear Mach, constant segment angle 
    # ------------------------------------------------------------------    
      
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "climb_5"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 50000. * Units.ft
    segment.mach_start   = 1.7
    segment.mach_end     = 2.02
    segment.climb_rate   = 750.  * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)     
    

    # ------------------------------------------------------------------
    #   Fourth Climb Segment: linear Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    ## Cruise-climb
    
    segment = Segments.Climb.Constant_Mach_Constant_Rate(base_segment)
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 56500. * Units.ft
    segment.mach_number  = 2.02
    segment.climb_rate   = 50.  * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    #   This segment is here primarily to test functionality of Constant_Mach_Constant_Altitude
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "level_cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.mach       = 2.02
    segment.distance   = 1. * Units.nmi
    segment.state.numerics.number_control_points = 4
        
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   First Descent Segment: decceleration
    # ------------------------------------------------------------------    
      
    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "decel_1"
    
    segment.analyses.extend( analyses.cruise )
    segment.acceleration      = -1.  * Units['m/s/s']
    segment.air_speed_start   = 2.02*573. * Units.kts
    segment.air_speed_end     = 1.5*573.  * Units.kts
    
    # add to mission
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------
    #   First Descent Segment
    # ------------------------------------------------------------------    
      
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "descent_1"
    
    segment.analyses.extend( analyses.cruise )
    segment.altitude_end = 41000. * Units.ft
    segment.mach_start = 1.5
    segment.mach_end   = 1.3
    segment.descent_rate = 2000. * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------
    #   First Descent Segment: decceleration
    # ------------------------------------------------------------------    
      
    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "decel_2"
    
    segment.analyses.extend( analyses.cruise )
    segment.acceleration      = -.5  * Units['m/s/s']
    segment.air_speed_start   = 1.35*573. * Units.kts
    segment.air_speed_end     = 0.95*573.  * Units.kts
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------
    #   First Descent Segment
    # ------------------------------------------------------------------    
      
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "descent_2"
    
    segment.analyses.extend( analyses.cruise )
    segment.altitude_end = 10000. * Units.ft
    segment.mach_start = 0.95
    segment.mach_end   = 250./638. # 638 is speed of sound in knots at 10,000 ft
    segment.descent_rate = 2000. * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------
    #   First Descent Segment
    # ------------------------------------------------------------------    
      
    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"
    
    segment.analyses.extend( analyses.cruise )
    segment.altitude_end = 0. * Units.ft
    segment.air_speed    = 250. * Units.kts
    segment.descent_rate = 1000. * Units['ft/min']
    
    # add to mission
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
        
