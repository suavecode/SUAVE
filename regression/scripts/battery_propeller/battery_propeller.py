# test_VTOL.py
# 
# Created: Feb 2020, M. Clarke
#
""" setup file for electric aircraft regression """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from X57_Maxwell    import vehicle_setup, configs_setup 

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
    
    # save results 
    #save_results(results)
    
    # plt the old results
    plot_mission(results,configs.base) 
      
    # load and plot old results  
    old_results = load_results()
    plot_mission(old_results,configs.base) 
 
    
    # RPM of rotor check during hover
    RPM        = results.segments.climb_1.conditions.propulsion.rpm[3][0]
    RPM_true   = 1328.4527794415255
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0]
    lift_coefficient_true   = 0.3837615929789279
    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3    
    
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
    mission  = mission_setup(configs_analyses,vehicle)
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.AERODAS()
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
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results ,vec_configs):
    prop_radius_ft = vec_configs.propulsors.propulsor.propeller.tip_radius*3.28084 # convert to ft 
    # ------------------------------------------------------------------
    #   Aero Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Aero Conditions")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)): 

        time = results.segments[i].conditions.frames.inertial.time[:,0] 
        cl   = results.segments[i].conditions.aerodynamics.lift_coefficient[:,0,None] 
        cd   = results.segments[i].conditions.aerodynamics.drag_coefficient[:,0,None] 
        aoa  = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d  = cl/cd

        axes = fig.add_subplot(2,2,1)
        axes.plot( time , aoa , 'bo-' )
        axes.set_ylabel('Angle of Attack (deg)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 

        axes = fig.add_subplot(2,2,2)
        axes.plot( time , cl, 'bo-' )
        axes.set_ylabel('CL' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 

        axes = fig.add_subplot(2,2,3)
        axes.plot( time , cd, 'bo-' )
        axes.set_xlabel('Time (s)' )
        axes.set_ylabel('CD' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 

        axes = fig.add_subplot(2,2,4)
        axes.plot( time , l_d, 'bo-' )
        axes.set_xlabel('Time (s)' )
        axes.set_ylabel('L/D' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
          


    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Propulsor")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):  

        time   = results.segments[i].conditions.frames.inertial.time[:,0]
        rpm    = results.segments[i].conditions.propulsion.rpm [:,0] 
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        torque = results.segments[i].conditions.propulsion.motor_torque
        effp   = results.segments[i].conditions.propulsion.etap[:,0]
        effm   = results.segments[i].conditions.propulsion.etam[:,0]
        prop_omega = results.segments[i].conditions.propulsion.rpm*0.104719755  
        ts = prop_omega*prop_radius_ft

        axes = fig.add_subplot(2,3,1)
        axes.plot(time, rpm, 'bo-')
        axes.set_ylabel('RPM' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        
        axes = fig.add_subplot(2,3,2)
        axes.plot(time, thrust, 'bo-')
        axes.set_ylabel('Thrust (N)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 

        axes = fig.add_subplot(2,3,3)
        axes.plot(time, torque, 'bo-' ) 
        axes.set_ylabel('Torque (N-m)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')  

        axes = fig.add_subplot(2,3,4)
        axes.plot(time, effp, 'bo-' )
        axes.set_xlabel('Time (s)' )
        axes.set_ylabel('Propeller Efficiency ($\eta_{prop}$)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        plt.ylim((0,1))

        axes = fig.add_subplot(2,3,5)
        axes.plot(time, effm, 'bo-' )
        axes.set_xlabel('Time (s)' )
        axes.set_ylabel('Motor Efficiency ($\eta_{motor}$)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        plt.ylim((0,1))

        axes = fig.add_subplot(2,3,6)
        axes.plot(time, ts, 'bo-' )
        axes.set_xlabel('Time (s)' )
        axes.set_ylabel('Tip Speed (ft/s)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')     
 

    return

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

def mission_setup(analyses,vehicle): 
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0. * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.005 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)        
    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy                   = vehicle.propulsors.propulsor.battery.max_energy * 0.89
    segment.altitude_start                   = 2500.0  * Units.feet
    segment.altitude_end                     = 8012    * Units.feet 
    segment.air_speed                        = 96.4260 * Units['mph'] 
    segment.climb_rate                       = 700.034 * Units['ft/min']  
    segment.state.unknowns.throttle          = 0.85 * ones_row(1)  

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 8012   * Units.feet
    segment.air_speed                 = 140.91 * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.9 *  ones_row(1)   

    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Descent Segment: constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "decent" 
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start            = 8012  * Units.feet
    segment.altitude_end              = 2500  * Units.feet
    segment.air_speed                 = 140.91 * Units['mph']  
    segment.climb_rate                = - 500.401  * Units['ft/min']  
    segment.state.unknowns.throttle  = 0.9 * ones_row(1)  
    
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


def load_results():
    return SUAVE.Input_Output.SUAVE.load('electric_aircraft.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'electric_aircraft.res')
    
    return  
if __name__ == '__main__': 
    main()    
