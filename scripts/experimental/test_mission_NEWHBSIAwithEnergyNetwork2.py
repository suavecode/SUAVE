# test_mission_NEWHBSIA.py
# 
# Created:  Emilio Botero, June 2014

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')


import SUAVE
from SUAVE.Attributes import Units

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

import numpy as np
import pylab as plt

import matplotlib

import copy, time

import solar_energy_network_5 as solarenergynetwork
from solar_energy_network_5 import Network
import Prop_Design


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
    vehicle.tag = 'HB-SIA'
    vehicle.Propulsors.propulsor = solarenergynetwork.Network()
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.Mass_Props.m_full       = 1600.    # kg
    vehicle.Mass_Props.m_empty      = 1600.    # kg
    vehicle.Mass_Props.m_takeoff      = 1600.    # kg
    
    # basic parameters
    vehicle.delta    = 0.                     # deg
    vehicle.S        = 200.                  # m^2
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   
    
    #print vehicle
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.sref      = vehicle.S     #
    wing.span      = 63.4          #m
    wing.ar        = (wing.span**2)/vehicle.S           #
    wing.sweep     = vehicle.delta * Units.deg #
    wing.symmetric = True          #
    wing.t_c       = 0.12          #
    wing.taper     = 1             #    
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chord_mac   = wing.sref/wing.span   #
    wing.S_exposed   = 0.8*wing.area_wetted  # might not be needed as input
    wing.S_affected  = 0.6*wing.area_wetted  # part of high lift system
    wing.e           = 0.97                  #
    wing.alpha_rc    = 0.0                   #
    wing.alpha_tc    = 0.0                   #
    wing.twist_rc    = 0.0*Units.degrees     #
    wing.twist_tc    = 0.0*Units.degrees     #  
    wing.highlift    = False                 
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.sref      = vehicle.S*880./10748.*1.1  #m^2
    wing.ar        = (63.4**2)/200                    #
    wing.sweep     = 0 * Units.deg              #
    wing.symmetric = True          
    wing.t_c       = 0.12                       #
    wing.taper     = 1                          #
    wing.twist_rc    = 0.0*Units.degrees     #
    wing.twist_tc    = 0.0*Units.degrees     #      
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #      
    wing.e          = 0.95                   #
    wing.alpha_rc   = 0.                   #
    wing.alpha_tc   = 0.                   #
  
    # add to vehicle
    vehicle.append_component(wing)    
    

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.sref      = vehicle.S*880./10748.*1.1  #m^2
    wing.ar        = (63.4**2)/200                    #
    wing.sweep     = 0 * Units.deg              #
    wing.symmetric = True          
    wing.t_c       = 0.12                       #
    wing.taper     = 1                          #
    wing.twist_rc    = 0.0*Units.degrees     #
    wing.twist_tc    = 0.0*Units.degrees     #          
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.S_exposed  = 0.8*wing.area_wetted  #
    wing.S_affected = 0.6*wing.area_wetted  #      
    wing.e          = 0.95                  #
    wing.alpha_rc   = 0.                    #
    wing.alpha_tc   = 0.                    #
  
    # add to vehicle
    vehicle.append_component(wing)  
    
    
    # ------------------------------------------------------------------
    #   Propulsor
    # ------------------------------------------------------------------
    
    #Design the Propeller
    Prop_attributes = Data()
    Prop_attributes.nu     = 0.00001789/0.8
    Prop_attributes.B      = 2. #Two blades
    Prop_attributes.V      = 19. # freestream m/s
    Prop_attributes.omega  = 1000.*(2.*np.pi/60.0)#rad/s
    Prop_attributes.R      = 0.5
    Prop_attributes.Rh     = 0.0508
    Prop_attributes.Des_CL = 0.8
    Prop_attributes.rho    = 0.8
    rho                    = Prop_attributes.rho
    
    print(Prop_attributes.omega)

    thrust = 0.0 # It is designed for power not thrust (just a guess)
    power  = 13000.0 #The motors are rated for 7.5 kw
    
    V  = Prop_attributes.V
    R  = Prop_attributes.R
    
    Prop_attributes.Tc = 2.*thrust/(rho*(V**2.)*np.pi*(R**2.))
    Prop_attributes.Pc = 2.*power/(rho*(V**3.)*np.pi*(R**2.))
    
    print(Prop_attributes.Tc)
    
    Prop_attributes = Prop_Design.main(Prop_attributes)
    
    print(Prop_attributes.Cp)
    
    # build network
    net = Network()
    
    # Component 1 the Sun?
    sun = solarenergynetwork.solar()
    net.solar_flux = sun
    
    # Component 2 the solar panels
    panel = solarenergynetwork.solar_panel()
    panel.A = 11628.*0.01
    panel.eff = 0.22
    net.solar_panel = panel
    
    # Component 3 the ESC
    esc = solarenergynetwork.ESC()
    esc.eff = 0.95 #Gundlach for brushless motors
    net.esc = esc
    
    # Component 5 the Propeller
    prop = solarenergynetwork.Propeller()
    prop.Prop_attributes = Prop_attributes
    net.propeller = prop
    
    # Component 4 the Motor
    motor = solarenergynetwork.Motor()
    #motor.Res = 0.1 #This is the largest Neumotors motor, 15000 W
    #motor.io = 2.2
    #motor.kv = 120.*(2.*np.pi/60.) #RPM/volt converted to rad/s 
    motor.Res = 0.111#This is the largest Neumotors motor, 15000 W
    motor.io = 0.7
    motor.kv = 32.*(2.*np.pi/60.) #RPM/volt converted to rad/s      
    motor.propradius = prop.Prop_attributes.R
    motor.propCp = prop.Prop_attributes.Cp
    motor.G   = 2. #Gear ratio
    motor.etaG = .99 # Gear box efficiency
    motor.exp_i = 500. #Expected current
    net.motor = motor
    
    # Component 6 the Payload
    payload = solarenergynetwork.Payload()
    payload.draw = 250. #Watts 
    net.payload = payload
    
    # Component 7 the Avionics
    avionics = solarenergynetwork.Avionics()
    avionics.draw = 50. #Watts  
    net.avionics = avionics
    
    # Component 8 the Battery
    bat = solarenergynetwork.Battery()
    bat.mass = 450.      #kg
    bat.type = 'Li-Ion'
    bat.R0 = 0.07446
    net.battery = bat
   
    #Component 9 the system logic controller and MPPT
    logic = solarenergynetwork.Solar_Logic() 
    logic.systemvoltage = 300.0
    logic.MPPTeff = 0.95
    net.solar_logic = logic
    
    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    aerodynamics.initialize(vehicle)
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Not so Simple Propulsion Model
    # ------------------------------------------------------------------ 
    vehicle.propulsion_model = net
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

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def define_mission(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'The Test Mission'

    # initial mass
    mission.m0 = vehicle.Mass_Props.m_full # linked copy updates if parent changes
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    
    # ------------------------------------------------------------------
    #   Climb Segment: Constant Speed, constant Rate of Climb
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.takeoff
    
    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 15.4  * Units['m/s']
    segment.climb_rate     = 100.*(Units.ft/Units.minute)   
    
    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.altitude   = 8.0  * Units.km     # Optional
    segment.air_speed  = 19.0 * Units['m/s']
    segment.distance   = 100.0 * Units.km
        
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 15.4 * Units['m/s']
    segment.descent_rate = 0.5   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission

# ----------------------------------------------------------------------
#   Evaluate the Mission
# ----------------------------------------------------------------------
def evaluate_mission(vehicle,mission):
    
    # ------------------------------------------------------------------    
    #   Run Mission
    # ------------------------------------------------------------------
    results = SUAVE.Methods.Performance.evaluate_mission(mission)
    
    return results

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def post_process(vehicle,mission,results):

    # ------------------------------------------------------------------    
    #   Throttle
    # ------------------------------------------------------------------
    plt.figure("Throttle History")
    axes = plt.gca()
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.Segments[i].conditions.propulsion.throttle[:,0]
        
        axes.plot(time, eta, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)
    
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
    #   Atmosphere
    # ------------------------------------------------------------------
    #title = "Atmosphere"
    #plt.figure(7)    
    #plt.title(title)
    #for segment in results.Segments.values():

        #plt.subplot(3,1,1)
        #plt.plot( segment.t / Units.minute , 
                  #segment.rho * np.ones_like(segment.t),
                  #'bo-' )
        #plt.xlabel('Time (min)')
        #plt.ylabel('Density (kg/m^3)')
        #plt.grid(True)
        
        #plt.subplot(3,1,2)
        #plt.plot( segment.t / Units.minute , 
                  #segment.p * np.ones_like(segment.t) ,
                  #'bo-' )
        #plt.xlabel('Time (min)')
        #plt.ylabel('Pressure (Pa)')
        #plt.grid(True)
        
        #plt.subplot(3,1,3)
        #plt.plot( segment.t / Units.minute , 
                  #segment.T * np.ones_like(segment.t) ,
                  #'bo-' )
        #plt.xlabel('Time (min)')
        #plt.ylabel('Temperature (K)')
        #plt.grid(True)
    
    
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
        axes.plot( time , Thrust , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Thrust (N)')
        axes.grid(True)
        
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

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , 'bo-' )
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
    
    plt.show()     
    
    return     

# ---------------------------------------------------------------------- 
#   Module Tests
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    profile_module = False
        
    if not profile_module:
        main()
        
    else:
        profile_file = 'log_Profile.out'
        
        import cProfile
        cProfile.run('import tut_mission_Boeing_737800 as tut; tut.profile()', profile_file)
        
        import pstats
        p = pstats.Stats(profile_file)
        p.sort_stats('time').print_stats(20)        
        
        import os
        os.remove(profile_file)

def profile():
    t0 = time.time()
    vehicle = define_vehicle()
    mission = define_mission(vehicle)
    results = evaluate_mission(vehicle,mission)
    print 'Run Time:' , (time.time()-t0)    