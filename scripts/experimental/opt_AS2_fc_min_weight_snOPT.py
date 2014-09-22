# opt_AS2_fc_min_weight.py
# 
# Created:  Tim MacDonald, 6/25/14
# Modified: Tim MacDonald, 9/10/14

""" evaluate a mission with an AS2
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
import pyOpt
#import mpi4py
import time
from SUAVE.Attributes import Units
from SUAVE.Attributes.Aerodynamics import Conditions
from fuel_cell_network import Network
import fuel_cell_network

import numpy as np
import scipy as sp
from scipy import integrate
import pylab as plt

import copy, time


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    t0 = time.time()
    
    # build the vehicle
    vehicle = define_vehicle()
    
    # define the mission
    mission = define_mission(vehicle)
    
    # evaluate the mission
    
    
    base_weight = 22500 # kg
    #cell_power_weight = 1500 # W/kg
    #range_goal = 5000.0 # nmi
    
    results = evaluate_mission(vehicle,mission)
    
    #inputs = np.array([base_weight,base_weight+10000.0])/10000.0
    inputs = np.array([48396.0,61895.0])/10000.0
    
    outputs = sp.optimize.fmin_slsqp(opt_func,inputs,args=(vehicle,mission),f_eqcons=constraints,iter=25,acc=1e-5)
    
    
    vehicle.Mass_Props.m_empty = outputs[0]*10000.0
    vehicle.Mass_Props.m_takeoff = outputs[1]*10000.0
    
    results = evaluate_mission(vehicle,mission)
    
    # plot results
    post_process(vehicle,mission,results)
    
    tf = time.time()
    
    print tf-t0
    
    return

# ----------------------------------------------------------------------
# Optimize Functions
# ----------------------------------------------------------------------

def opt_func(inputs,vehicle,mission):
    
    vehicle.Mass_Props.m_empty   = inputs[0]*10000.0
    vehicle.Mass_Props.m_takeoff = inputs[1]*10000.0
    vehicle.Configs.cruise.Mass_Props.m_empty   = inputs[0]*10000.0
    vehicle.Configs.cruise.Mass_Props.m_takeoff = inputs[1]*10000.0
    
    results = evaluate_mission(vehicle,mission)
    
    m_empty = vehicle.Configs.cruise.Mass_Props.m_empty
    mass_base = vehicle.Configs.cruise.Mass_Props.m_takeoff
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.Segments[i].conditions.weights.total_mass[:,0]
        mdot = results.Segments[i].conditions.propulsion.fuel_mass_rate[:,0]
        mass_from_mdot = np.array([mass_base] * len(time))
        mass_from_mdot[1:] = -integrate.cumtrapz(mdot,time*60.0)+mass_base
        mass_base = mass_from_mdot[-1]
             
        
    m_fuel = vehicle.Configs.cruise.Mass_Props.m_takeoff - mass_base
    output = m_fuel
    
    print output
    
    return output/10000.0
    
def constraints(inputs,vehicle,mission):

    vehicle.Mass_Props.m_empty   = inputs[0]*10000.0
    vehicle.Mass_Props.m_takeoff = inputs[1]*10000.0
    vehicle.Configs.cruise.Mass_Props.m_empty   = inputs[0]*10000.0
    vehicle.Configs.cruise.Mass_Props.m_takeoff = inputs[1]*10000.0
    
    m_empty = vehicle.Configs.cruise.Mass_Props.m_empty
    mass_base = vehicle.Configs.cruise.Mass_Props.m_takeoff
    
    results = evaluate_mission(vehicle,mission)
    cell_power_weight = 1500 # W/kg
    base_weight_frame = 22500 # kg - includes fuel reserves
    fuel_reserve_weight = 1000 
    base_weight = base_weight_frame + fuel_reserve_weight
    max_power = 0.0
    #powers = np.ones(len(results.Segments))
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.Segments[i].conditions.weights.total_mass[:,0]
        mdot = results.Segments[i].conditions.propulsion.fuel_mass_rate[:,0]
        mass_from_mdot = np.array([mass_base] * len(time))
        mass_from_mdot[1:] = -integrate.cumtrapz(mdot,time*60.0)+mass_base
        mass_base = mass_from_mdot[-1]
        
        velocity   = results.Segments[i].conditions.freestream.velocity[:,0]
        Thrust     = results.Segments[i].conditions.frames.body.thrust_force_vector[:,0]
        power      = velocity*Thrust

        max_current = np.max(power)
        max_power = np.max(np.array([max_power,max_current]))    
        
    m_takeoff = vehicle.Configs.cruise.Mass_Props.m_takeoff
    m_fuel = m_takeoff - mass_base    
    m_empty = vehicle.Configs.cruise.Mass_Props.m_empty
    m_fuel_cell_req = max_power/cell_power_weight
    m_fuel_cell     = m_empty - base_weight
    
    outputs = np.array([m_fuel_cell-m_fuel_cell_req,m_takeoff-m_fuel-m_empty])
    
    print outputs
    
    return outputs/10000.0

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Aerion AS2'

    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    n_select = 3
    if n_select == 0:
        vehicle_propellant = SUAVE.Attributes.Propellants.Jet_A()
        vehicle.Mass_Props.m_full       = 53000    # kg
        vehicle.Mass_Props.m_empty      = 22500    # kg
        vehicle.Mass_Props.m_takeoff    = 52000    # kg       
    elif n_select == 1:
        vehicle_propellant = SUAVE.Attributes.Propellants.Jet_A()
        vehicle.Mass_Props.m_full       = 73000    # kg
        vehicle.Mass_Props.m_empty      = 42000    # kg
        vehicle.Mass_Props.m_takeoff    = 72000    # kg   
    elif n_select == 2:
        vehicle_propellant = SUAVE.Attributes.Propellants.Liquid_Natural_Gas()
        vehicle.Mass_Props.m_full       = 67000    # kg
        vehicle.Mass_Props.m_empty      = 42000    # kg
        vehicle.Mass_Props.m_takeoff    = 66000    # kg          
    elif n_select == 3:
        vehicle_propellant = SUAVE.Attributes.Propellants.Liquid_H2()
        vehicle.Mass_Props.m_full       = 53000    # kg
        vehicle.Mass_Props.m_empty      = 42000    # kg
        vehicle.Mass_Props.m_takeoff    = 52000    # kg      
    # mass properties
    # 53000 / 22500 base weight
    # 71000 / 40500 with fuel cell using Jet A
    # 65000 / 40500 with fuel cell using LNG
    # 51000 / 40500 with fuel cell using LH2
    #vehicle_propellant = SUAVE.Attributes.Propellants.Jet_A1()
    #vehicle_propellant = SUAVE.Attributes.Propellants.Liquid_Natural_Gas()
    #vehicle_propellant = SUAVE.Attributes.Propellants.Liquid_H2()
    #vehicle.Mass_Props.m_full       = 71000    # kg
    #vehicle.Mass_Props.m_empty      = 40500    # kg
    #vehicle.Mass_Props.m_takeoff    = 70000    # kg
    vehicle.Mass_Props.m_flight_min = 25000    # kg - Note: Actual value is unknown

    # basic parameters
    vehicle.delta    = 0.0                      # deg
    vehicle.S        = 125.4                    # m^2
    vehicle.A_engine = np.pi*(1.0/2)**2       # m^2   
    
    
     # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'

    wing.Areas.reference = 125.4    #
    wing.aspect_ratio    = 3.63     #
    wing.Spans.projected = 21.0     #
    wing.sweep           = 0 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.03
    wing.taper           = 0.7

    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.Chords.mean_aerodynamic = 7.0
    wing.Areas.exposed = 0.8*wing.Areas.wetted
    wing.Areas.affected = 0.6*wing.Areas.wetted
    wing.span_efficiency = 0.74
    wing.Twists.root = 0.0*Units.degrees
    wing.Twists.tip  = 2.0*Units.degrees
    wing.vertical = False
    
    wing.high_lift    = False                 #
    wing.high_mach    = True
    wing.vortex_lift  = False
    wing.transition_x = 0.9
    
    #print wing
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #   Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    
    wing.Areas.reference = 24.5     #
    wing.aspect_ratio    = 2.0      #
    wing.Spans.projected = 7.0      #
    wing.sweep           = 0 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.03
    wing.taper           = 0.5

    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.Chords.mean_aerodynamic = 3.0
    wing.Areas.exposed = 0.8*wing.Areas.wetted
    wing.Areas.affected = 0.6*wing.Areas.wetted
    wing.span_efficiency = 0.74
    wing.Twists.root = 0.0*Units.degrees
    wing.Twists.tip  = 2.0*Units.degrees
    wing.vertical = False
    
    wing.high_lift    = False                 #
    wing.high_mach    = True
    wing.vortex_lift  = False
    wing.transition_x = 0.9
    
    #print wing
    # add to vehicle
    vehicle.append_component(wing)    
    
    
    # ------------------------------------------------------------------
    #   Vertcal Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertcal Stabilizer'    
    
    wing.Areas.reference = 33.91    #
    wing.aspect_ratio    = 1.3      #
    wing.Spans.projected = 3.5      #
    wing.sweep           = 45 * Units.deg
    wing.symmetric       = False
    wing.thickness_to_chord = 0.04
    wing.taper           = 0.5

    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.Chords.mean_aerodynamic = 4.2
    wing.Areas.exposed = 1.0*wing.Areas.wetted
    wing.Areas.affected = 0.0*wing.Areas.wetted
    wing.span_efficiency = 0.9
    wing.Twists.root = 0.0*Units.degrees
    wing.Twists.tip  = 0.0*Units.degrees
    wing.vertical = True
    
        
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.number_coach_seats = 0
    fuselage.seats_abreast = 2
    fuselage.seat_pitch = 0
    fuselage.Fineness.nose = 4.0
    fuselage.Fineness.tail = 4.0
    fuselage.Lengths.fore_space = 16.3
    fuselage.Lengths.aft_space  = 16.3
    fuselage.width = 2.2
    fuselage.Heights.maximum = 1.9
    
    # size fuselage planform
    SUAVE.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    
    # ------------------------------------------------------------------
    #  Ducted Fan / Fuel Cell Model
    # ------------------------------------------------------------------    

    ductedfan = SUAVE.Components.Propulsors.Turbojet_SupersonicPASS()
    ductedfan.tag = 'Ducted Fan'
    ductedfan.nacelle_dia = (.5)*2
    ductedfan.engine_length = 10.0
    ductedfan.no_of_engines = 3.0
    ductedfan.propellant = vehicle_propellant
    
    # add to vehicle
    vehicle.append_component(ductedfan)    
    
    net = Network()
    
    net.propellant = vehicle_propellant
    net.fuel_cell = fuel_cell_network.Fuel_Cell()
    net.fuel_cell.inputs.propellant = vehicle_propellant
    net.fuel_cell.efficiency = 0.8
    net.fuel_cell.max_mdot = 1.0
    net.motor = fuel_cell_network.Motor()
    net.motor.efficiency = 0.95
    net.propulsor = fuel_cell_network.Propulsor()
    net.propulsor.A0 = (.5)**2*np.pi

#    vehicle.append_component(net)

    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero_Supersonic()
    aerodynamics.initialize(vehicle)
    vehicle.aerodynamics_model = aerodynamics
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
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

#: def define_vehicle()


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
    #   Sixth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 6"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        

    segment.altitude_start = 0.0    * Units.km
    segment.altitude_end = 3.05     * Units.km
    segment.air_speed    = 128.6    * Units['m/s']
    segment.climb_rate   = 4000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------
    #   Seventh Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 7"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        

    segment.altitude_end = 4.57     * Units.km
    segment.air_speed    = 205.8    * Units['m/s']
    segment.climb_rate   = 1000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment) 
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 8"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 9.77 * Units.km # 
    segment.mach_number_start = 0.64
    segment.mach_number_end  = 1.0 
    segment.climb_rate   = 1000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)  
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 9"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 12.95 * Units.km # 51000 ft
    segment.mach_number_start = 1.0
    segment.mach_number_end  = 1.22
    segment.climb_rate   = 1000    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment)  
    
    # ------------------------------------------------------------------
    #   Eighth Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Linear_Mach_Constant_Rate()
    segment.tag = "Climb - 10"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 15.54 * Units.km # 51000 ft
    segment.mach_number_start = 1.22
    segment.mach_number_end  = 1.4
    segment.climb_rate   = 200    * Units['ft/min']
    
    # add to mission
    mission.append_segment(segment) 
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Mach_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.altitude   = 15.54  * Units.km     # Optional
    segment.mach       = 1.4
    # 1687 for 3000 nmi
    
    desired_range = 4000.0
    cruise_dist = desired_range - 1313.0
    segment.distance   = cruise_dist * Units.nmi
        
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Linear_Mach_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.Configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 6.8  * Units.km
    segment.mach_number_start = 1.4
    segment.mach_number_end = 1.0
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Linear_Mach_Constant_Rate()
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

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
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
    #base_time = 0.0
    max_power = (vehicle.Mass_Props.m_empty - 1000 - 22500.0)*1500
    for segment in results.Segments.values():
        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        eta  = segment.conditions.propulsion.throttle[:,0]
        max_mdot = segment.config.propulsion_model.fuel_cell.max_mdot
        e = segment.config.propulsion_model.fuel_cell.efficiency
        spec_energy = segment.config.propulsion_model.fuel_cell.inputs.propellant.specific_energy
        power = spec_energy*eta*max_mdot*e
        mdot = segment.conditions.propulsion.fuel_mass_rate[:,0]
        velocity   = segment.conditions.freestream.velocity[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , power/1000.0 , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Power Output (kW)')
        axes.grid(True)
        
        #axes = fig.add_subplot(3,1,2)
        #axes.plot( time , mdot , 'bo-' )
        #axes.set_xlabel('Time (mins)')
        #axes.set_ylabel('Fuel Burn Rate (kg/s)')
        #axes.grid(True)  
        
        power = velocity*Thrust/1000.0
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , power , 'bo-' )
        axes.plot( time , np.array([max_power/1000.0] * len(time)) , 'r--')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Required (kW)')
        axes.grid(True)   
        
        power = velocity*Thrust
        mdot_power = mdot*segment.config.propulsion_model.propellant.specific_energy
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
    #   Efficiency
    # ------------------------------------------------------------------

    #plt.figure("Efficiency")
    #axes = plt.gca()    
    #for i in range(len(results.Segments)):     
        #time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        #e = results.Segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        #axes.plot(time, aoa, 'bo-')
    #axes.set_xlabel('Time (mins)')
    #axes.set_ylabel('Angle of Attack (deg)')
    #axes.grid(True)        
    
    
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
    m_empty = vehicle.Mass_Props.m_empty
    mass_base = vehicle.Mass_Props.m_takeoff
    for i in range(len(results.Segments)):
        time = results.Segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.Segments[i].conditions.weights.total_mass[:,0]
        mdot = results.Segments[i].conditions.propulsion.fuel_mass_rate[:,0]
        eta  = results.Segments[i].conditions.propulsion.throttle[:,0]
        mass_from_mdot = np.array([mass_base] * len(time))
        mass_from_mdot[1:] = -integrate.cumtrapz(mdot,time*60.0)+mass_base
        axes.plot(time, mass_from_mdot, 'b--')
        axes.plot(time, mass, 'bo-')
        mass_base = mass_from_mdot[-1]
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
    dist_base = 0.0
    for segment in results.Segments.values():
            
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        velocity   = segment.conditions.freestream.velocity[:,0]
        density   = segment.conditions.freestream.density[:,0]
        mach_number   = segment.conditions.freestream.mach_number[:,0]
        
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , velocity , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Velocity (m/s)')
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mach_number , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Mach')
        axes.grid(True)
        
        distance = np.array([dist_base] * len(time))
        distance[1:] = integrate.cumtrapz(velocity*1.94,time/60.0)+dist_base
        dist_base = distance[-1]
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , distance , 'bo-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Distance (nmi)')
    
    
    
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
    
    base_weight = 22500.0
    base_weight_str = 'Base Weight = ' + '%.0f' % base_weight + ' kg\n'
    
    m_takeoff = vehicle.Mass_Props.m_takeoff
    m_takeoff_str = 'Takeoff Weight = ' + '%.0f' % m_takeoff + ' kg\n'
    
    m_empty   = vehicle.Mass_Props.m_empty
    m_empty_str = 'Empty Weight = ' + '%.0f' % m_empty + ' kg\n'
    
    m_fuel_cell = m_empty - base_weight
    m_fuel_cell_str = 'Fuel Cell Weight = ' + '%.0f' % m_fuel_cell + ' kg\n'
    
    m_fuel      = m_takeoff - m_empty
    m_fuel_str = 'Fuel Weight = ' + '%.0f' % m_fuel + ' kg\n'
    
    total_range = distance[-1]
    total_range_str = 'Range = ' + '%.0f' % total_range + ' nmi\n'
    
    prop_name = vehicle.propulsion_model.propellant.tag
    prop_str = 'Propellant Type = ' + prop_name + '\n'
    
    f = open(prop_name+'_%.0f'%total_range+'.txt','w')
    f.write(prop_str)
    f.write(total_range_str)
    f.write(m_fuel_str)
    f.write(m_takeoff_str)
    f.write(m_empty_str)
    f.write(m_fuel_cell_str)
    f.write(base_weight_str)
    f.write('Reserve Fuel = 1000 kg')
    f.close
    
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