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

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

from SUAVE.Components.Energy.Networks.Solar_Network import Solar_Network
from SUAVE.Methods.Propulsion     import propeller_design

# =============================================================================
# External Python modules
# =============================================================================

import numpy as np
import scipy as sp
import pylab as plt
import matplotlib
import copy, time

import pyOpt 
import pyOpt.pySNOPT

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except:
    raise ImportError('mpi4py is required for parallelization')
#end



# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    tinitial = time.time()
    
    # Inputs
    inputs = np.array([.2,0.5, 0.5, 0.5, 0.5, 0.5, 1.4,0.5, 0.5, 0.5, 0.5, 0.5, .7])
    

    # build the vehicle
    vehicle = define_vehicle(inputs)
    
    # define the mission
    mission = define_mission(vehicle,inputs)  
    
    #inputs = np.array([4.38426387e-01, 2.78402866e+00, 2.90969814e+00, 1.61344924e+00,
                       #2.55831700e+00, 1.00000000e-05, 1.61398089e+00, 1.38578361e+00,
                       #1.25524133e+00, 9.69395115e-01, 2.29906316e+00, 1.00000000e+00])
                       
    #inputs = [  2.02840808e-01   1.29508252e+00   8.48112178e-01   1.14300381e+00
   #1.02964276e+00   1.00000000e-05   1.83272257e+00   5.87133406e-01
   #5.73175990e-01   1.13395204e+00   2.07268811e+00   9.71275042e-01]
    
    # Have the optimizer call the wrapper
    mywrap = lambda inputs:wrap(inputs,vehicle,mission)
    
    opt_prob = pyOpt.Optimization('Fb',mywrap)
    opt_prob.addObj('Battery')
    opt_prob.addVar('Vehicle Weight','c',lower=0.06,upper=0.5,value=inputs[0])
    opt_prob.addVar('x2','c',lower=1e-5,upper=3.0,value=inputs[1])
    opt_prob.addVar('x3','c',lower=1e-5,upper=3.0,value=inputs[2])
    opt_prob.addVar('x4','c',lower=1e-5,upper=3.0,value=inputs[3])
    opt_prob.addVar('x5','c',lower=1e-5,upper=3.0,value=inputs[4])
    opt_prob.addVar('x6','c',lower=1e-5,upper=3.0,value=inputs[5])
    opt_prob.addVar('x7','c',lower=1e-5,upper=3.0,value=inputs[6])
    opt_prob.addVar('x8','c',lower=1e-5,upper=3.0,value=inputs[7])
    opt_prob.addVar('x9','c',lower=1e-5,upper=3.0,value=inputs[8])
    opt_prob.addVar('x10','c',lower=1e-5,upper=3.0,value=inputs[9])
    opt_prob.addVar('x11','c',lower=1e-5,upper=3.0,value=inputs[10])
    opt_prob.addVar('x12','c',lower=1e-5,upper=1.0,value=inputs[11])
    opt_prob.addConGroup('g',1,'i')
    
    opt = pyOpt.pySNOPT.SNOPT()

    print opt_prob
    outputs = opt(opt_prob, sens_type='FD',sens_mode='pgc')
                 
    if myrank==0:

        vehicle,mission,results = run_plane(outputs[1])
        
        deltat = time.time() - tinitial
        print('Time Elapsed')
        print(deltat)
        # Plot results    
        post_process(vehicle,mission,results)
        
    
    
    return

# ----------------------------------------------------------------------
#   wrap
# ----------------------------------------------------------------------
def wrap(inputs,vehicle,mission):
    
    # Change vehicle masses
    weight = inputs[0]*1000.
    vehicle.mass_properties.takeoff         = weight
    vehicle.mass_properties.operating_empty = weight
    vehicle.mass_properties.max_takeoff     = weight   
    
    print('Inputs')
    print inputs
    
    # Resize
    vehicle.mass_properties.breakdown = SUAVE.Methods.Weights.Correlations.Human_Powered.empty(vehicle)
    wingmass  = vehicle.wings['main_wing'].mass_properties.mass
    motmass   = vehicle.propulsion_model.motor.mass_properties.mass
    paylmass  = vehicle.propulsion_model.payload.mass_properties.mass
    panelmass = vehicle.propulsion_model.solar_panel.mass_properties.mass
    
    batmass = weight - (wingmass + motmass + paylmass + panelmass)
    
    vehicle.propulsion_model.battery.mass_properties.mass = batmass
    
    # Update the configs
    vehicle.configs.takeoff.mass_properties.max_takeoff     = weight
    vehicle.configs.takeoff.mass_properties.operating_empty = weight
    vehicle.configs.takeoff.mass_properties.takeoff         = weight
    vehicle.configs.cruise.mass_properties.max_takeoff      = weight
    vehicle.configs.cruise.mass_properties.operating_empty  = weight
    vehicle.configs.cruise.mass_properties.takeoff          = weight    
    
    # Redefine the mission
    mission = define_mission(vehicle,inputs)      
    
    #Recharge the battery to start
    mission.segments.Climb1.battery_energy =  batmass * (250. * Units.watts * Units.hr ) * inputs[11]
    
    # evaluate the mission
    results = evaluate_mission(vehicle,mission)
    
    #Process the results, we want to maximize battery energy
    end_energy = results.segments[-1].conditions.propulsion.battery_energy[-1,0]

    output = -end_energy/(100.*250.*3600.) #This normalizes the output a bit
    
    print('Output')
    print output
    
    if np.isnan(end_energy):
        fail = 1
    else:
        fail = 0
        
    const = [0]
    
    # 24 hour flight     
    const[0] = (24. * Units.hr - results.segments[-1].conditions.frames.inertial.time[-1,0])/(3600.)
    
    print const
    
    return output,const,fail

# ----------------------------------------------------------------------
#   wrap
# ----------------------------------------------------------------------
def run_plane(outputs):
    
    #The first element of inputs is the weight, the rest are throttles
    
    # build the vehicle
    vehicle = define_vehicle(outputs)
    
    # define the mission
    mission = define_mission(vehicle,outputs)
    
    # evaluate the mission
    results = evaluate_mission(vehicle,mission)
    
    return vehicle,mission,results


# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def define_vehicle(inputs):
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Solar'
    vehicle.propulsors.propulsor = SUAVE.Components.Energy.Networks.Solar_Network()
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------  
    
    weight = inputs[0]*1000
    # mass properties
    vehicle.mass_properties.takeoff         = weight
    vehicle.mass_properties.operating_empty = weight
    vehicle.mass_properties.max_takeoff     = weight 
    
    # basic parameters
    vehicle.envelope.ultimate_load = 2.25
    vehicle.qm            = 0.5*1.225*(20.**2.) #Max q
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio       = 25.
    wing.spans.projected    = 40. * Units.m
    wing.areas.reference    = (wing.spans.projected**2)/wing.aspect_ratio
    wing.sweep              = 0. * Units.deg
    wing.symmetric          = True
    wing.thickness_to_chord = 0.12
    wing.taper              = 1.0      
    
    vehicle.reference_area  = wing.areas.reference
    
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
    wing.number_ribs             = 26.
    wing.number_end_ribs         = 2.
    wing.transition_x_u          = 0.6
    wing.transition_x_l          = 0.9
    
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Propulsor
    # ------------------------------------------------------------------
    
    # build network
    net = Solar_Network()
    net.number_motors = 2.
    net.nacelle_dia   = 0.2
    
    # Component 1 the Sun?
    sun = SUAVE.Components.Energy.Processes.Solar_Radiation()
    net.solar_flux = sun
    
    # Component 2 the solar panels
    panel = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.area                 = vehicle.reference_area
    panel.efficiency           = 0.20
    panel.mass_properties.mass = panel.area*.55
    net.solar_panel            = panel
    
    # Component 3 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc       = esc
    
    # Component 5 the Propeller

    # Design the Propeller
    prop_attributes = Data()
    prop_attributes.number_blades       = 2.0 
    prop_attributes.freestream_velocity = 70.0 # freestream m/s
    prop_attributes.angular_velocity    = 300.*(2.*np.pi/60.0)
    prop_attributes.tip_radius          = 3.0
    prop_attributes.hub_radius          = 0.0508
    prop_attributes.design_Cl           = 0.7 
    prop_attributes.design_altitude     = 23.0 * Units.km
    prop_attributes.design_thrust       = 0.0
    prop_attributes.design_power        = 3500.0
    prop_attributes                     = propeller_design(prop_attributes)
    
    prop                 = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes = prop_attributes
    net.propeller        = prop
    
    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    motor.resistance           = 0.008
    motor.no_load_current      = 4.5
    motor.speed_constant       = 120.*(2.*np.pi/60.) # RPM/volt converted to rad/s     
    motor.propeller_radius     = prop.prop_attributes.tip_radius
    motor.propeller_Cp         = prop.prop_attributes.Cp
    motor.gear_ratio           = 20. # Gear ratio
    motor.gearbox_efficiency   = .99 # Gear box efficiency
    motor.expected_current     = 160. # Expected current
    motor.mass_properties.mass = 2.0
    net.motor                  = motor   
    
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 150. #Watts 
    payload.mass_properties.mass = 5.0 * Units.kg
    net.payload                  = payload
    
    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 25. #Watts  
    net.avionics        = avionics    
    
    # ------------------------------------------------------------------
    #   Add up all of the masses
    # ------------------------------------------------------------------
    vehicle.mass_properties.breakdown = SUAVE.Methods.Weights.Correlations.Human_Powered.empty(vehicle)
    wingmass  = vehicle.wings['main_wing'].mass_properties.mass
    motmass   = motor.mass_properties.mass
    paylmass  = payload.mass_properties.mass
    panelmass = panel.mass_properties.mass
    
    batmass = weight - (wingmass + motmass + paylmass + panelmass)        

    # ------------------------------------------------------------------
    #   Propulsor again
    # ------------------------------------------------------------------
    
    # Component 8 the Battery # I already assume 250 Wh/kg for batteries
    bat = SUAVE.Components.Energy.Storages.Battery()
    bat.mass_properties.mass = batmass * Units.kg
    bat.type                 = 'Li-Ion'
    bat.resistance           = 0.0 #This needs updating
    net.battery              = bat
   
    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 60.0
    logic.MPPT_efficiency = 0.95
    net.solar_logic       = logic

    
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
def define_mission(vehicle,inputs):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'The Test Mission'

    # initial mass
    mission.m0 = vehicle.mass_properties.takeoff # linked copy updates if parent changes
    
    # atmospheric model
    mission.start_time  = time.strptime("Thu, Mar 20 06:00:00  2014", "%a, %b %d %H:%M:%S %Y",)
    mission.atmosphere  = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.planet      = SUAVE.Attributes.Planets.Earth()
    
    # ------------------------------------------------------------------
    #   Climb Segment: Constant Speed, constant Rate of Climb
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes
    segment.altitude_start = 18.0   * Units.km
    segment.altitude_end   = 20.0   * Units.km
    segment.air_speed      = 45.0  * Units['m/s']
    segment.battery_energy = vehicle.propulsion_model.battery.max_energy() #Charge the battery to start
    segment.latitude       = 35.0
    segment.longitude      = 0.0
    segment.climb_rate     = inputs[1]  
    
    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Climb Segment: Constant Speed, constant Rate of Climb
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb2"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes
    segment.altitude_start = 20.0   * Units.km
    segment.altitude_end   = 22.0   * Units.km
    segment.air_speed      = 50.0  * Units['m/s']
    segment.climb_rate     = inputs[2]  
    
    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Climb Segment: Constant Speed, constant Rate of Climb
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb3"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes
    segment.altitude_start = 22.0   * Units.km
    segment.altitude_end   = 24.0   * Units.km
    segment.air_speed      = 55.0  * Units['m/s']
    segment.climb_rate     = inputs[3]  
    
    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Climb Segment: Constant Speed, constant Rate of Climb
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb4"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes

    segment.altitude_start = 24.0   * Units.km
    segment.altitude_end   = 26.0   * Units.km
    segment.air_speed      = 60.0  * Units['m/s']
    segment.climb_rate     = inputs[4]  
    
    # add to misison
    mission.append_segment(segment)      
         
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.altitude   = 26.0   * Units.km     # Optional
    segment.air_speed  = 65.0   * Units['m/s']
    segment.distance   = inputs[5]*1000. * Units.km
        
    mission.append_segment(segment)    
       
    
    # ------------------------------------------------------------------    
    #   Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent1"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.altitude_end = 24.  * Units.km
    segment.air_speed    = 55.0 * Units['m/s']
    segment.descent_rate = inputs[6]  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)      
    
    # ------------------------------------------------------------------    
    #   Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent2"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.altitude_end = 22.  * Units.km
    segment.air_speed    = 50.0 * Units['m/s']
    segment.descent_rate = inputs[7]  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------    
    #   Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent3"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.altitude_end = 20.  * Units.km
    segment.air_speed    = 45.0 * Units['m/s']
    segment.descent_rate = inputs[8]  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)      
    
    # ------------------------------------------------------------------    
    #   Descent Segment: constant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent4"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.altitude_end = 18.  * Units.km
    segment.air_speed    = 40.0 * Units['m/s']
    segment.descent_rate = inputs[9]  * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)     
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise2"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.altitude   = 18.0  * Units.km     # Optional
    segment.air_speed  = 30.0  * Units['m/s']
    segment.distance   = inputs[10]*1000. * Units.km
        
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