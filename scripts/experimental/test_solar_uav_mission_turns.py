# test_solar_UAV_mission.py
# 
# Created:  Emilio Botero, July 2014

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

from SUAVE.Components.Energy.Networks.Solar_Network import Solar_Network
from SUAVE.Components.Energy.Converters.Propeller_Design import Propeller_Design

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    weight      = 200.0 #Kg
    weight_last = 0.0 
    delta       = weight - weight_last
    
    while delta > 0.5:
    
        # build the vehicle
        vehicle = define_vehicle(weight)
        
        weight_last = weight
        weight      = vehicle.mass_properties.mass
        delta       = np.abs(weight - weight_last)
        
    print('Converged Vehicle Weight:')
    print(weight)
    
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

def define_vehicle(weight):
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Solar'
    vehicle.propulsors.propulsor = SUAVE.Components.Energy.Networks.Solar_Network()
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = weight
    vehicle.mass_properties.operating_empty = weight
    vehicle.mass_properties.max_takeoff     = weight 
    
    # basic parameters
    vehicle.reference_area          = 80.               # m^2         
    vehicle.envelope.ultimate_load  = 2.0
    vehicle.qm                      = 0.5*1.225*(25.**2.) #Max q
    vehicle.Ltb                     = 10. # Tail boom length
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.areas.reference    = vehicle.reference_area     #
    wing.spans.projected    = 40.          #m
    wing.aspect_ratio       = (wing.spans.projected*2)/wing.areas.reference 
    wing.sweep              = 0. * Units.deg #
    wing.symmetric          = True          #
    wing.thickness_to_chord = 0.12          #
    wing.taper              = 1.             #    
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected  #
    wing.areas.exposed           = 0.8*wing.areas.wetted  # might not be needed as input
    wing.areas.affected          = 0.6*wing.areas.wetted # part of high lift system
    wing.span_efficiency         = 0.97                  #
    wing.twists.root             = 0.0*Units.degrees     #
    wing.twists.tip              = 0.0*Units.degrees     #  
    wing.highlift                = False  
    wing.vertical                = False 
    wing.eta                     = 1.0
    wing.Nwr                     = 26.
    wing.Nwer                    = 2.
    wing.transition_x_u          = 0.6
    wing.transition_x_l          = 0.9
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.areas.reference    = vehicle.reference_area*.15  #m^2
    wing.aspect_ratio       = 20.            #
    wing.spans.projected    = np.sqrt(wing.aspect_ratio*wing.areas.reference)
    wing.sweep              = 0 * Units.deg   #
    wing.symmetric          = True          
    wing.thickness_to_chord = 0.12                       #
    wing.taper              = 1.             #
    wing.twists.root        = 0.0*Units.degrees     #
    wing.twists.tip         = 0.0*Units.degrees     #
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected  #
    wing.areas.exposed           = 0.8*wing.areas.wetted  # might not be needed as input
    wing.areas.affected          = 0.6*wing.areas.wetted # part of high lift system    
    wing.span_efficiency         = 0.95                   #
    wing.twists.root             = 0.                     #
    wing.twists.tip              = 0.                     #
    wing.Nwr                     = 5.
  
    # add to vehicle
    vehicle.append_component(wing)    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.areas.reference    = vehicle.reference_area*.1 #m^2
    wing.aspect_ratio       = 20.             #
    wing.spans.projected    = np.sqrt(wing.aspect_ratio*wing.areas.reference)
    wing.sweep              = 0 * Units.deg         #
    wing.symmetric          = True          
    wing.thickness_to_chord = 0.12                  #
    wing.taper              = 1.                    #
    wing.twists.root        = 0.0*Units.degrees     #
    wing.twists.tip         = 0.0*Units.degrees     #       
    
    # size the wing planform
    SUAVE.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected  #
    wing.areas.exposed           = 0.8*wing.areas.wetted  # might not be needed as input
    wing.areas.affected          = 0.6*wing.areas.wetted # part of high lift system
    wing.span_efficiency         = 0.97                  #
    wing.twists.root             = 0.0*Units.degrees     #
    wing.twists.tip              = 0.0*Units.degrees     #  
    wing.Nwr                     = 5.
  
    # add to vehicle
    vehicle.append_component(wing)  
    
    # ------------------------------------------------------------------
    #   Propulsor
    # ------------------------------------------------------------------
    
    # build network
    net             = Solar_Network()
    net.num_motors  = 1.
    net.nacelle_dia = 0.2
    
    # Component 1 the Sun?
    sun            = SUAVE.Components.Energy.Properties.solar()
    net.solar_flux = sun
    
    # Component 2 the solar panels
    panel                 = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.A               = vehicle.reference_area + vehicle.wings['Horizontal Stabilizer'].areas.reference
    panel.eff             = 0.22
    panel.mass_properties.mass = panel.A*.55
    net.solar_panel       = panel
    
    # Component 3 the ESC
    esc     = SUAVE.Components.Energy.Distributors.ESC()
    esc.eff = 0.95 # Gundlach for brushless motors
    net.esc = esc
    
    # Component 5 the Propeller
    
    # Propeller design point
    design_altitude = 23.0 * Units.km
    Velocity        = 50.0 # freestream m/s
    RPM             = 300.
    Blades          = 2.0
    Radius          = 4.25
    Hub_Radius      = 0.0508
    Thrust          = 0.0     #Specify either thrust or power to design for
    Power           = 13000.0 #Specify either thrust or power to design for
    
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    p, T, rho, a, mu = atmosphere.compute_values(design_altitude)
    
    # Design the Propeller
    prop_attributes        = Data()
    prop_attributes.nu     = mu/rho
    prop_attributes.a      = a
    prop_attributes.T      = T
    prop_attributes.B      = Blades 
    prop_attributes.V      = Velocity
    prop_attributes.omega  = RPM*(2.*np.pi/60.0)
    prop_attributes.R      = Radius
    prop_attributes.Rh     = Hub_Radius
    prop_attributes.Des_CL = 0.7
    prop_attributes.rho    = rho
    prop_attributes.Tc     = 2.*Thrust/(rho*(Velocity**2.)*np.pi*(Radius**2.))
    prop_attributes.Pc     = 2.*Power/(rho*(Velocity**3.)*np.pi*(Radius**2.))
    prop_attributes        = Propeller_Design(prop_attributes)
    
    prop                 = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes = prop_attributes
    net.propeller        = prop

    # Component 4 the Motor
    motor     = SUAVE.Components.Energy.Converters.Motor()
    motor.res             = 0.008
    motor.io              = 4.5
    motor.kv              = 120.*(2.*np.pi/60.) # RPM/volt converted to rad/s     
    motor.propradius      = prop.prop_attributes.R
    motor.propCp          = prop.prop_attributes.Cp
    motor.g               = 20. # Gear ratio
    motor.etaG            = .98 # Gear box efficiency
    motor.exp_i           = 160. # Expected current
    motor.mass_properties.mass = 2.0
    net.motor             = motor    
    
    # Component 6 the Payload
    payload                 = SUAVE.Components.Energy.Peripherals.Payload()
    payload.draw            = 250. #Watts 
    payload.mass_properties.mass = 25.0 * Units.kg
    net.payload             = payload
    
    # Component 7 the Avionics
    avionics      = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.draw = 25. #Watts  
    net.avionics  = avionics
    
    # Component 8 the Battery
    bat                 = SUAVE.Components.Energy.Storages.Battery()
    bat.mass_properties.mass = 150. #kg
    bat.type            = 'Li-Ion'
    bat.R0              = 0.07
    net.battery         = bat
   
    #Component 9 the system logic controller and MPPT
    logic               = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.systemvoltage = 65.0
    logic.MPPTeff       = 0.95
    net.solar_logic     = logic
    
    vehicle.mass_properties.breakdown = SUAVE.Methods.Weights.Correlations.Solar_HPA.empty(vehicle)
    
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
    #   Add up all of the masses
    # ------------------------------------------------------------------
    wingmass  = vehicle.wings['Main Wing'].mass_properties.mass
    HTmass    = vehicle.wings['Horizontal Stabilizer'].mass_properties.mass
    VTmass    = vehicle.wings['Vertical Stabilizer'].mass_properties.mass
    batmass   = bat.mass_properties.mass
    motmass   = motor.mass_properties.mass
    paylmass  = payload.mass_properties.mass
    panelmass = panel.mass_properties.mass
    
    total_mass = (wingmass + HTmass + VTmass + batmass +
                  motmass + paylmass + panelmass)
    
    vehicle.mass_properties.mass = total_mass    
    
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
    mission.m0 = vehicle.mass_properties.max_takeoff # linked copy updates if parent changes
    
    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet     = SUAVE.Attributes.Planets.Earth()
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Altitude_Radius_Loiter()
    segment.tag = "Loiter"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.altitude       = 20.0  * Units.km     # Optional
    segment.air_speed      = 50.0  * Units['m/s']
    segment.radius         = 2.0   * Units.km
    segment.time           = 2.0   * Units.minutes
    segment.battery_energy = vehicle.propulsion_model.battery.max_energy() #Charge the battery to start
    segment.latitude       = 37.4300
    segment.longitude      = -122.1700    
    segment.numerics.n_control_points = 32
        
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
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.segments[i].conditions.propulsion.throttle[:,0]
        
        axes.plot(time, eta, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)
    
    plt.figure("Angle of Attack History")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)            

    # ------------------------------------------------------------------    
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.segments[i].conditions.freestream.altitude[:,0] / Units.km
        axes.plot(time, altitude, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)    
    
    # ------------------------------------------------------------------    
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces")
    for segment in results.segments.values():
        
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
    for segment in results.segments.values():
        
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
    for i, segment in enumerate(results.segments.values()):
        
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
    
    
    # ------------------------------------------------------------------    
    #   Battery Energy
    # ------------------------------------------------------------------
    plt.figure("Battery Energy")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        energy = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        axes.plot(time, energy, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Battery Energy (J)')
    axes.grid(True)   
    
    # ------------------------------------------------------------------    
    #   Solar Flux
    # ------------------------------------------------------------------
    plt.figure("Solar Flux")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        energy = results.segments[i].conditions.propulsion.solar_flux[:,0] 
        axes.plot(time, energy, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Solar Flux ($W/m^{2}$)')
    axes.grid(True)      
    
    # ------------------------------------------------------------------    
    #   Current Draw
    # ------------------------------------------------------------------
    plt.figure("Current Draw")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        energy = results.segments[i].conditions.propulsion.current[:,0] 
        axes.plot(time, energy, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Current Draw (Amps)')
    axes.grid(True)  
    
    # ------------------------------------------------------------------    
    #   Motor RPM
    # ------------------------------------------------------------------
    plt.figure("Motor RPM")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        energy = results.segments[i].conditions.propulsion.rpm[:,0] 
        axes.plot(time, energy, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Motor RPM')
    axes.grid(True)
    
    # ------------------------------------------------------------------    
    #   Battery Draw
    # ------------------------------------------------------------------
    plt.figure("Battery Charging")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        energy = results.segments[i].conditions.propulsion.battery_draw[:,0] 
        axes.plot(time, energy, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Battery Charging (Watts)')
    axes.grid(True)    
    
    # ------------------------------------------------------------------    
    #   Propulsive efficiency
    # ------------------------------------------------------------------
    plt.figure("Prop")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        etap = results.segments[i].conditions.propulsion.etap[:,0] 
        axes.plot(time, etap, 'bo-')
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Etap')
    axes.grid(True)      
    
    # ------------------------------------------------------------------    
    #   Flight Path
    # ------------------------------------------------------------------
    plt.figure("Flight Path")
    axes = plt.gca()    
    for i in range(len(results.segments)):     
        lat = results.segments[i].conditions.frames.planet.latitude[:,0] 
        lon = results.segments[i].conditions.frames.planet.longitude[:,0] 
        axes.plot(lon, lat, 'bo-')
    axes.set_ylabel('Latitude')
    axes.set_xlabel('Longitude')
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