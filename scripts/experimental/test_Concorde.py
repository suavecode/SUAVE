# test_Concorde.py
# 
# Created:  Tim MacDonald, 6/25/14
# Modified: Tim MacDonald, 7/21/14

""" evaluate a mission with a Concorde
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Attributes.Aerodynamics import Conditions

import numpy as np
import pylab as plt

import copy, time


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle
    vehicle = define_vehicle()
    
    # define the mission
    mission = define_mission(vehicle)
    
    # evaluate the mission
    results = evaluate_mission(vehicle,mission)
    
    # plot results
    post_process(vehicle,mission,results)
    
    return


# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Concorde'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # Data mostly from www.concordesst.com

    # mass properties
    vehicle.Mass_Props.m_full       = 186880   # kg
    vehicle.Mass_Props.m_empty      = 78700    # kg
    vehicle.Mass_Props.m_takeoff    = 185000   # kg
    vehicle.Mass_Props.m_flight_min = 100000   # kg - Note: Actual value is unknown

    # basic parameters
    vehicle.delta    = 55.0                     # deg
    vehicle.S        = 358.25                   # m^2
    vehicle.A_engine = np.pi*(1.212/2)**2       # m^2   
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.sref      = 358.25         #
    wing.ar        = 1.83           #
    wing.span      = 25.6           #
    wing.sweep     = 55 * Units.deg #
    wing.symmetric = True           #
    wing.t_c       = 0.03           #
    wing.taper     = 0              # Estimated based on drawing

    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac   = 14.02                 #
    wing.S_exposed   = 0.8*wing.area_wetted  #
    wing.S_affected  = 0.6*wing.area_wetted  #
    wing.e           = 0.74                   # Actual value is unknown
    wing.twist_rc    = 0.0*Units.degrees     #
    wing.twist_tc    = 3.0*Units.degrees     #
    wing.highlift    = False                 #
    wing.highmach    = True
    wing.vortexlift  = True
    #wing.hl          = 1                    #
    #wing.flaps_chord = 20                   #
    #wing.flaps_angle = 20                   #
    #wing.slats_angle = 10                   #
    
    #print wing
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertcal Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertcal Stabilizer'    
    
    wing.sref      = 33.91          #
    wing.ar        = 1.07           # 
    wing.span      = 11.32          #
    wing.sweep     = 55 * Units.deg # Estimate
    wing.symmetric = False          #
    wing.t_c       = 0.04           # Estimate
    wing.taper     = 0.25           # Estimate
    wing.highmach  = True
    
    # size the wing planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac  = 8.0                   # Estimate
    wing.S_exposed  = 1.0*wing.area_wetted  #
    wing.S_affected = 0.0*wing.area_wetted  #  
    wing.e          = 0.9                   #
    wing.twist_rc   = 0.0*Units.degrees     #
    wing.twist_tc   = 0.0*Units.degrees     #
    wing.vertical   = True    
        
    
        
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    fuselage.num_coach_seats = 0    # Actually ~120, using 0 for length simplicity
    fuselage.seats_abreast   = 4    #
    fuselage.seat_pitch      = 0    #
    fuselage.fineness_nose   = 3.48 #
    fuselage.fineness_tail   = 3.48 #
    fuselage.fwdspace        = 20.83#
    fuselage.aftspace        = 20.83#
    fuselage.width           = 2.87 #
    fuselage.height          = 3.30 #
    
    # size fuselage planform
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Turbojet
    # ------------------------------------------------------------------    
    
    #turbojet = SUAVE.Components.Propulsors.Turbojet2PASS()
    turbojet = SUAVE.Components.Propulsors.Turbojet_SupersonicPASS()
    turbojet.tag = 'Turbojet Variable Nozzle'
    
    turbojet.propellant = SUAVE.Attributes.Propellants.Jet_A1()
    
    turbojet.analysis_type                 = '1D'     #
    turbojet.diffuser_pressure_ratio       = 1.0      # 1.0 either not known or not relevant
    turbojet.fan_pressure_ratio            = 1.0      #
    turbojet.fan_nozzle_pressure_ratio     = 1.0      #
    turbojet.lpc_pressure_ratio            = 3.1      #
    turbojet.hpc_pressure_ratio            = 5.0      #
    turbojet.burner_pressure_ratio         = 1.0      #
    turbojet.turbine_nozzle_pressure_ratio = 1.0      #
    turbojet.Tt4                           = 1450.0   #
    turbojet.design_thrust                 = 139451.  # 31350 lbs
    turbojet.no_of_engines                 = 4.0      #
    turbojet.engine_length                 = 11.5     # meters - includes 3.4m inlet
    
    # turbojet sizing conditions
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
    # Note: Sizing designed to give roughly nominal values - M = 2.02 is not achieved at 35,000 ft
    
    sizing_segment.M   = 2.02                    #
    sizing_segment.alt = 35000 * Units.ft        #
    sizing_segment.T   = 218.0                   #
    sizing_segment.p   = 0.239*10**5             #
    
    # size the turbojet
    turbojet.engine_sizing_1d(sizing_segment)     
    
    # add to vehicle
    vehicle.append_component(turbojet)


    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero_Supersonic()
    aerodynamics.initialize(vehicle)
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------     
    
    vehicle.propulsion_model = vehicle.Propulsors

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.Configs.takeoff
    

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle

#: def define_vehicle()


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def define_mission(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'The Test Mission'

    # initial mass
    mission.m0 = vehicle.Mass_Props.m_full # linked copy updates if parent changes
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    

    
    # ------------------------------------------------------------------
    #   Sixth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 6"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        

    segment.altitude_start = 0.0    * Units.km
    segment.altitude_end = 3.05     * Units.km
    segment.air_speed    = 128.6    * Units['m/s']
    segment.climb_rate   = 20.32    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------
    #   Seventh Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 7"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        

    segment.altitude_end = 4.57     * Units.km
    segment.air_speed    = 205.8    * Units['m/s']
    segment.climb_rate   = 10.16    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment) 
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 8"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 7.60 * Units.km # 
    segment.mach_number_start = 0.64
    segment.mach_number_end  = 1.0 
    segment.climb_rate   = 5.08    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)  
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 9"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 15.24 * Units.km # 50000 ft
    segment.mach_number_start = 1.0
    segment.mach_number_end  = 2.02 
    segment.climb_rate   = 5.08    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------    
    #   Cruise-Climb Segment 1: constant speed, constant throttle
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Climb.Constant_Mach_Constant_Rate()
    segment.tag = "Cruise-Climb"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 18.288 * Units.km # 55,000ft - 18.288 km for 60,000 ft
    segment.mach_number  = 2.02  
    segment.climb_rate   = 0.5    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Mach_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.altitude   = 18.288  * Units.km     # Optional
    segment.mach       = 2.02
    segment.distance   = 2000.0 * Units.km
        
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Analyses.Mission.Segments.Descent.Linear_Mach_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 6.8  * Units.km
    segment.mach_number_start = 2.02
    segment.mach_number_end = 1.0
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Analyses.Mission.Segments.Descent.Linear_Mach_Constant_Rate()
    segment.tag = "Descent - 2"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 3.0  * Units.km
    segment.mach_number_start = 1.0
    segment.mach_number_end = 0.65
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
      
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Analyses.Mission.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 5"

    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet    
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 130.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)       

    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission

#: def define_mission()


# ----------------------------------------------------------------------
#   Evaluate the Mission
# ----------------------------------------------------------------------
def evaluate_mission(vehicle,mission):
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    results = SUAVE.Methods.Performance.evaluate_mission(mission)
   
    
    # ------------------------------------------------------------------    
    #   Compute Useful Results
    # ------------------------------------------------------------------
    #SUAVE.Methods.Results.compute_energies(results,summary=True)
    #SUAVE.Methods.Results.compute_efficiencies(results)
    #SUAVE.Methods.Results.compute_velocity_increments(results)
    #SUAVE.Methods.Results.compute_alpha(results)    
    
    return results

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,mission,results):
    
    
    # ------------------------------------------------------------------    
    #   Throttle
    # ------------------------------------------------------------------
    fig = plt.figure("Throttle and Fuel Burn")
    tot_energy = 0.0
    for segment in results.Segments.values():
        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        eta  = segment.conditions.propulsion.throttle[:,0]
        mdot = segment.conditions.propulsion.fuel_mass_rate[:,0]
        velocity   = segment.conditions.freestream.velocity[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , eta , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Throttle')
        axes.grid(True)
        
        #axes = fig.add_subplot(3,1,2)
        #axes.plot( time , mdot , 'bo-' )
        #axes.set_xlabel('Time (mins)')
        #axes.set_ylabel('Fuel Burn Rate (kg/s)')
        #axes.grid(True)  
        
        power = velocity*Thrust/1000.0
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , power , 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Required (kW)')
        axes.grid(True)   
        
        power = velocity*Thrust
        mdot_power = mdot*segment.config.propulsion_model['Turbojet Variable Nozzle'].propellant.specific_energy
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , power/mdot_power , 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Total Efficiency')
        axes.grid(True)          
        
        tot_energy = tot_energy + np.trapz(power/1000.0,time*60)
    print 'Integrated Power Required: %.0f kJ' % tot_energy
                  

    # ------------------------------------------------------------------    
    #   Angle of Attack
    # ------------------------------------------------------------------
    
    plt.figure("Angle of Attack History")
    axes = plt.gca()    
    for i in range(len(results.Segments)):     
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.Segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)        

    
    
    # ------------------------------------------------------------------    
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()    
    for i in range(len(results.Segments)):     
        time     = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.Segments[i].conditions.freestream.altitude[:,0] / Units.km
        axes.plot(time, altitude, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Vehicle Mass
    # ------------------------------------------------------------------    
    plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.Segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)
    
    mo = vehicle.Mass_Props.m_full
    mf = mass[-1]
    D_m = mo-mf
    spec_energy = vehicle.Propulsors[0].propellant.specific_energy
    tot_energy = D_m*spec_energy
    print "Total Energy Used          %.0f kJ (does not account for efficiency loses)" % (tot_energy/1000.0)

    # ------------------------------------------------------------------    
    #   Concorde Debug
    # ------------------------------------------------------------------
     
    fig = plt.figure("Velocity and Density")
    for segment in results.Segments.values():
            
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        velocity   = segment.conditions.freestream.velocity[:,0]
        density   = segment.conditions.freestream.density[:,0]
        mach_number   = segment.conditions.freestream.mach_number[:,0]
        
        axes = fig.add_subplot(2,1,1)
        axes.plot( time , velocity , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Velocity (m/s)')
        axes.grid(True)
        
        axes = fig.add_subplot(2,1,2)
        axes.plot( time , mach_number , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Mach')
        axes.grid(True)       
    
    
    
    # ------------------------------------------------------------------    
    #   Aerodynamics
    # ------------------------------------------------------------------
    
    fig = plt.figure("Aerodynamic Forces")
    for segment in results.Segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , Lift , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Lift (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , Drag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag (N)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Lift/Drag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('L/D')
        axes.grid(True)        
        
        #axes = fig.add_subplot(3,1,3)
        #axes.plot( time , Thrust , 'bo-' )
        #axes.set_xlabel('Time (min)')
        #axes.set_ylabel('Thrust (N)')
        #axes.grid(True)
        
    # ------------------------------------------------------------------    
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    
    fig = plt.figure("Aerodynamic Coefficients")
    for segment in results.Segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        CLV    = segment.conditions.aerodynamics.lift_breakdown.vortex[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , 'bo-' )
        axes.plot( time , CLV , 'yo-')  
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , CDrag , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CD')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Drag   , 'bo-' )
        axes.plot( time , Thrust , 'ro-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag and Thrust (N)')
        axes.grid(True)
    
    
    # ------------------------------------------------------------------    
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    
    fig = plt.figure("Drag Components")
    axes = plt.gca()    
    for i, segment in enumerate(results.Segments.values()):
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]
        
        
        axes.plot( time , cdp , 'ko-', label='CD_P' )
        axes.plot( time , cdi , 'bo-', label='CD_I' )
        axes.plot( time , cdc , 'go-', label='CD_C' )
        axes.plot( time , cdm , 'yo-', label='CD_M' )
        axes.plot( time , cd  , 'ro-', label='CD'   )
        
        if i == 0:
            axes.legend(loc='upper center')
        
    axes.set_xlabel('Time (min)')
    axes.set_ylabel('CD')
    axes.grid(True)
    
    
    return     



# ---------------------------------------------------------------------- 
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    profile_module = False
        
    if not profile_module:
        
        main()
        plt.show()        
        
    else:
        profile_file = 'log_Profile.out'
        
        import cProfile
        cProfile.run('import tut_mission_Boeing_737800 as tut; tut.profile()', profile_file)
        
        import pstats
        p = pstats.Stats(profile_file)
        p.sort_stats('time').print_stats(20)        
        
        import os
        os.remove(profile_file)
    
#: def main()


def profile():
    t0 = time.time()
    vehicle = define_vehicle()
    mission = define_mission(vehicle)
    results = evaluate_mission(vehicle,mission)
    print 'Run Time:' , (time.time()-t0)