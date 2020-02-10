# ICE_Test.py
# 
# Created: Feb 2020, M. Clarke
#
""" setup file for a mission with a Cessna 172 with an internal combustion engine network
"""

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
 
from Cessna_172      import vehicle_setup, configs_setup 
import copy

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
    save_results(results)
    
    # plt the old results
    plot_mission(results,configs.base) 
    
    # save, load and plot old results 
    save_results(results)
    old_results = load_results()
    plot_mission(old_results,configs.base) 
 
    
    # RPM of rotor check during hover
    RPM        = results.segments.climb.conditions.propulsion.rpm[0][0]
    RPM_true   = 1689.0341273520216
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  


    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true   = 0.4379214920009708

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
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.settings.drag_coefficient_increment = 0.0000

    # landing analysis
    aerodynamics = analyses.landing.aerodynamics


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
    weights = SUAVE.Analyses.Weights.Weights()
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

def plot_mission(results, configs ):
     
    prop_radius_ft = configs.propulsors.internal_combustion.propeller.tip_radius/Units.feet # convert to ft      
    
    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------


    fig = plt.figure("Aerodynamic Forces",figsize=(8,6))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta  = segment.conditions.propulsion.throttle[:,0]
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust	

        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , 'bo-' )
        axes.set_ylabel('Thrust (lbf)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)             
        axes.grid(True)

        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Throttle')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)             
        axes.grid(True)	
    #plt.savefig("Cessna_Piston_Aero_Forces.png")  


    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        aoa = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d = CLift/CDrag

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift ,'bo-' )
        axes.set_ylabel('Lift Coefficient')
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , l_d ,'bo-' )
        axes.set_ylabel('L/D')
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , aoa , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('AOA (deg)')
        axes.grid(True)
    #plt.savefig("Cessna_Piston_Aero.png")  
    
    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Drag Components",figsize=(8,10))
    axes = plt.gca()
    for i, segment in enumerate(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]
 
        axes.plot( time , cdp , 'ko-', label='CD parasite' )
        axes.plot( time , cdi , 'bo-', label='CD induced' )
        axes.plot( time , cdc , 'go-', label='CD compressibility' )
        axes.plot( time , cdm , 'yo-', label='CD miscellaneous' )
        axes.plot( time , cd  , 'ro-', label='CD total'   )
        axes.legend(loc='upper center')            
     

    axes.set_xlabel('Time (min)')
    axes.set_ylabel('CD')
    axes.grid(True)
    #plt.savefig("Cessna_Piston_Drag.png")     
    # ------------------------------------------------------------------
    #   Altitude, sfc, vehicle weight
    # ------------------------------------------------------------------

    fig = plt.figure("Altitude_sfc_weight",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        aoa    = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d    = CLift/CDrag
        mass   = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.ft
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust	

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude ,'bo-' )
        axes.set_ylabel('Altitude (ft)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , sfc , 'bo-')
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('sfc (lb/lbf-hr)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , 'bo-' )
        axes.set_ylabel('Weight (lb)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)
        plt.ylim((1800,2550))
   
        
    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Velocities",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta  = segment.conditions.propulsion.throttle[:,0]
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust
        velocity  = segment.conditions.freestream.velocity[:,0]
        pressure  = segment.conditions.freestream.pressure[:,0]
        density  = segment.conditions.freestream.density[:,0]
        EAS = velocity * np.sqrt(density/1.225)
        mach = segment.conditions.freestream.mach_number[:,0]


        axes = fig.add_subplot(3,1,1)
        axes.plot( time , velocity / Units.kts, 'bo-' )
        axes.set_ylabel('velocity (kts)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , EAS / Units.kts,'bo-')
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Equivalent Airspeed')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)    
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , mach , 'bo-')
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Mach')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)          
    #plt.savefig("Cessna_Piston_Velocity.png")        
   
    
    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Propulsor")
    fig.set_size_inches(10, 8)
    for i in range(len(results.segments)):  

        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.rpm[:,0] 
        power  = results.segments[i].conditions.propulsion.power[:,0] / Units.horsepower
        torque = results.segments[i].conditions.propulsion.propeller_torque 
        prop_omega = rpm*0.104719755  
        ts = prop_omega*prop_radius_ft        

        axes = fig.add_subplot(2,2,1)
        axes.plot(time, rpm, 'bo-')
        axes.set_ylabel('RPM')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)
        axes.grid(True)       
        #plt.ylim((0,2700))

        axes = fig.add_subplot(2,2,2)
        axes.plot(time, power, 'bo-')
        axes.set_ylabel('Power (hp)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)      
        axes.grid(True)   
        plt.ylim((0,180))

        axes = fig.add_subplot(2,2,3)
        axes.plot(time, torque, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Torque (N-m)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)   
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, ts, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Tip Speed (ft/s)')
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.get_yaxis().get_major_formatter().set_useOffset(False)     
        axes.grid(True)           
    #plt.savefig("Cessna_Piston_Propulsion.png")          
        
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
    #   First Climb Segment: constant Speed, constant rate segment 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb"

    segment.analyses.extend( analyses.takeoff )

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 13500.* Units.ft
    segment.air_speed      = 100  * Units['knots']
    segment.climb_rate     = 695.  * Units['ft/min']

    # add to misison
    mission.append_segment(segment)
       
    # ------------------------------------------------------------------
    #   First Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses.cruise )

    segment.altitude  = 13500. * Units.ft
    segment.air_speed = 140. *Units['mph']
    segment.distance  = 100.   * Units.nautical_mile

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
    return SUAVE.Input_Output.SUAVE.load('cessna_mission.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'cessna_mission.res')
    return 

if __name__ == '__main__': 
    main()    
    plt.show()
    