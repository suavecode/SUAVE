# Bell_X1_tut.py
# 
# Created:  April 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# Python Imports
import numpy as np
import pylab as plt

# Vehicle Import
import sys
sys.path.append('../Vehicles')
from Bell_X1 import vehicle_setup, configs_setup

# SUAVE Imports
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Methods.Propulsion.liquid_rocket_sizing import liquid_rocket_sizing
from SUAVE.Input_Output.Results import  print_parasite_drag,  \
    print_compress_drag, \
    print_engine_data,   \
    print_mission_breakdown, \
    print_weight_breakdown

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

    # plt the results
    plot_mission(results)
    plt.show()

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

def simple_sizing(configs):
    base = configs.base
    base.pull_base()

    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted

    # diff the new data
    base.store_diff()

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

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()

    # ------------------------------------------------------------------
    # Climb Segment: Constant Throttle, Constant Speed
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    
    segment.tag = "climb_self"
    segment.analyses.extend(analyses.general)
    segment.altitude_start                       = 23000.0 * Units.feet
    segment.altitude_end                         = 43000.0 * Units.feet
    segment.throttle                             = 1.00
    segment.air_speed                            = 150.0   * Units.m/Units.s
    segment.state.unknowns.wind_angle            = segment.state.ones_row(1) * 13.0 * Units.deg
    segment.state.numerics.number_control_points = 16
   
    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    # Cruise Segment: Constant Throttle, Constant Altitude, Subsonic to M = 1
    # ------------------------------------------------------------------
    segment = Segments.Cruise.Constant_Throttle_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    segment.analyses.extend(analyses.general)    
    segment.altitude        = 43000.0 * Units.feet
    segment.throttle        = 1.0
    segment.air_speed_start = 150.0   * Units.m/Units.s
    segment.air_speed_end   = 316.0   * Units['m/s']
    segment.state.numerics.number_control_points = 32

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

    return missions  

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------
def plot_mission(results,line_style='bo-'):

    axis_font = {'fontname':'Arial', 'size':'14'}    

    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces",figsize=(8,6))
    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta    = segment.conditions.propulsion.throttle[:,0]
        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , line_style )
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Throttle',axis_font)
        axes.grid(True)	
        #plt.savefig("B737_engine.pdf")
        #plt.savefig("B737_engine.png")

    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients",figsize=(8,10))
    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        aoa = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d = CLift/CDrag
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , line_style )
        axes.set_ylabel('Lift Coefficient',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)        
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , l_d , line_style )
        axes.set_ylabel('L/D',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , aoa , 'ro-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('AOA (deg)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)
        #plt.savefig("B737_aero.pdf")
        #plt.savefig("B737_aero.png")

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

        if line_style == 'bo-':
            axes.plot( time , cdp , 'ko-', label='CD parasite' )
            axes.plot( time , cdi , 'bo-', label='CD induced' )
            axes.plot( time , cdc , 'go-', label='CD compressibility' )
            axes.plot( time , cdm , 'yo-', label='CD miscellaneous' )
            axes.plot( time , cd  , 'ro-', label='CD total'   )

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
    #plt.savefig("B737_drag.pdf")
    #plt.savefig("B737_drag.png")

    # ------------------------------------------------------------------
    #   Altitude, sfc, vehicle weight
    # ------------------------------------------------------------------
    fig = plt.figure("Altitude_sfc_weight",figsize=(8,10))
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        mass     = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.ft
        mdot     = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust   =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc      = (mdot / Units.lb) / (thrust /Units.lbf) * Units.hr
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude , line_style )
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , sfc , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , 'ro-' )
        axes.set_ylabel('Weight (lb)',axis_font)
        axes.grid(True)

        #plt.savefig("B737_mission.pdf")
        #plt.savefig("B737_mission.png")

    # ------------------------------------------------------------------
    #   Velocities
    # ------------------------------------------------------------------
    fig = plt.figure("Velocities",figsize=(8,10))
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift     = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag     = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust   = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lb
        velocity = segment.conditions.freestream.velocity[:,0]
        pressure = segment.conditions.freestream.pressure[:,0]
        density  = segment.conditions.freestream.density[:,0]
        EAS      = velocity * np.sqrt(density/1.225)
        mach     = segment.conditions.freestream.mach_number[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , velocity / Units.kts, line_style )
        axes.set_ylabel('velocity (kts)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , EAS / Units.kts, line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Equivalent Airspeed',axis_font)
        axes.grid(True)    

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , mach , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Mach',axis_font)
        axes.grid(True)           

    return

if __name__ == '__main__': 

    main()    
    plt.show()