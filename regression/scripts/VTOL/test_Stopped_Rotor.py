# test_Stopped_Rotor.py
# 
# Created:  Feb 2020, M. Clarke
#
""" setup file for a mission with a  Stopped Rotor eVTOL 
"""

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core import Units, Data 
import copy
from SUAVE.Components.Energy.Networks.Lift_Cruise import Lift_Cruise 
from SUAVE.Methods.Weights.Buildups.Electric_Lift_Cruise.empty import empty
import sys
import pylab as plt
import numpy as np  

sys.path.append('../Vehicles')
# the analysis functions
 
from Stopped_Rotor import vehicle_setup   

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():        
        
    # ------------------------------------------------------------------------------------------------------------------
    # Stopped-Rotor   
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup() 
    analyses.finalize()     
    weights   = analyses.weights
    breakdown = weights.evaluate() 
    mission   = analyses.mission  
    
    # evaluate mission     
    results   = mission.evaluate()
        
    # plot results
    plot_mission(results,configs)
      
    # save, load and plot old results 
    #save_stopped_rotor_results(results)
    old_results = load_stopped_rotor_results()
    plot_mission(old_results,configs, 'k-')
    plt.show(block=True)    
    
    # RPM of rotor check during hover
    RPM        = results.segments.climb_1.conditions.propulsion.rpm_lift[0][0]
    RPM_true   = 2258.286261769841
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # Battery Energy Check During Transition
    battery_energy_hover_to_transition         = results.segments.transition_1.conditions.propulsion.battery_energy[:,0]
    battery_energy_hover_to_transition_true    = np.array([3.06429161e+08,3.06380479e+08,3.06186506e+08 ,3.05843629e+08,
                                                           3.05384280e+08,3.04820654e+08,3.04178123e+08 ,3.03484506e+08,
                                                           3.02775394e+08,3.02087748e+08,3.01458035e+08 ,3.00916095e+08,
                                                           3.00483060e+08,3.00170388e+08,2.99982462e+08 ,2.99919877e+08])
    
    print(battery_energy_hover_to_transition)
    diff_battery_energy_hover_to_transition    = np.abs(battery_energy_hover_to_transition  - battery_energy_hover_to_transition_true) 
    print('battery_energy_hover_to_transition difference')
    print(diff_battery_energy_hover_to_transition)   
    assert all(np.abs((battery_energy_hover_to_transition - battery_energy_hover_to_transition_true)/battery_energy_hover_to_transition) < 1e-3)

    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true   = 0.6962308249944807
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

    # vehicle analyses
    analyses = base_analysis(vehicle)

    # mission analyses
    mission  = mission_setup(analyses,vehicle)

    analyses.mission = mission
    
    return  vehicle, analyses


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
    weights = SUAVE.Analyses.Weights.Weights_Electric_Lift_Cruise()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
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


def mission_setup(analyses,vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission            = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag        = 'the_mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport    = airport    

    # unpack Segments module
    Segments                                                 = SUAVE.Analyses.Mission.Segments
                                                             
    # base segment                                           
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_transition
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_transition
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)    


    # VSTALL Calculation
    m      = vehicle.mass_properties.max_takeoff
    g      = 9.81
    S      = vehicle.reference_area
    atmo   = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho    = atmo.compute_values(1000.*Units.feet,0.).density
    CLmax  = 1.2 
    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax)))

    # ------------------------------------------------------------------
    #   First Taxi Segment: Constant Speed
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "Ground_Taxi"

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "climb_1"
    
    segment.analyses.extend( analyses ) 
    
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 40.  * Units.ft
    segment.climb_rate                                       = 500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.propulsors.propulsor.battery.max_energy*0.95
                                                             
    segment.state.unknowns.propeller_power_coefficient_lift  = 0.04 * ones_row(1)
    segment.state.unknowns.throttle_lift                     = 0.85 * ones_row(1)
    segment.state.unknowns.__delitem__('throttle')

    segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_no_forward
    segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_no_forward       
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------

    segment     = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag = "transition_1"

    segment.analyses.extend( analyses )

    segment.altitude        = 40.  * Units.ft
    segment.air_speed_start = 0.   * Units['ft/min']
    segment.air_speed_end   = 0.8 * Vstall
    segment.acceleration    = 9.81/5
    segment.pitch_initial   = 0.0
    segment.pitch_final     = 5. * Units.degrees

    segment.state.unknowns.propeller_power_coefficient_lift = 0.04 * ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.80 * ones_row(1) 
    segment.state.unknowns.propeller_power_coefficient      = 0.06 * ones_row(1)
    segment.state.unknowns.throttle                         = .70  * ones_row(1)   
    segment.state.residuals.network                         = 0.   * ones_row(3)    

    segment.process.iterate.unknowns.network                = vehicle.propulsors.propulsor.unpack_unknowns_transition
    segment.process.iterate.residuals.network               = vehicle.propulsors.propulsor.residuals_transition    
    segment.process.iterate.unknowns.mission                = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------

    segment     = Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb(base_segment)
    segment.tag = "transition_2"

    segment.analyses.extend( analyses )
     
    segment.altitude_start         = 40.0 * Units.ft
    segment.altitude_end           = 50.0 * Units.ft
    segment.air_speed              = 0.8 * Vstall
    segment.climb_angle            = 1 * Units.degrees
    segment.acceleration           = 1. * Units['m/s/s']    
    segment.pitch_initial          = 8. * Units.degrees  
    segment.pitch_final            = 7. * Units.degrees       
    
    segment.state.unknowns.propeller_power_coefficient_lift = 0.05 * ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.70 * ones_row(1) 
    segment.state.unknowns.propeller_power_coefficient      = 0.06 * ones_row(1)
    segment.state.unknowns.throttle                         = .40  * ones_row(1)   
    segment.state.residuals.network                         = 0.   * ones_row(3)    

    segment.process.iterate.unknowns.network                = vehicle.propulsors.propulsor.unpack_unknowns_transition
    segment.process.iterate.residuals.network               = vehicle.propulsors.propulsor.residuals_transition    
    segment.process.iterate.unknowns.mission                = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment     = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend( analyses )

    segment.air_speed       = 56.7362 
    segment.altitude_start  = 50.0 * Units.ft
    segment.altitude_end    = 300. * Units.ft
    segment.climb_rate      = 500. * Units['ft/min']

    segment.state.unknowns.propeller_power_coefficient         = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                            = 0.70 * ones_row(1)
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift     

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------

    segment     = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag = "departure_terminal_procedures"

    segment.analyses.extend( analyses )

    segment.altitude  = 300.0 * Units.ft
    segment.time      = 60.   * Units.second
    segment.air_speed = 1.2*Vstall

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift     


    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Acceleration, Constant Rate
    # ------------------------------------------------------------------

    segment     = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "accelerated_climb"

    segment.analyses.extend( analyses )

    segment.altitude_start  = 300.0 * Units.ft
    segment.altitude_end    = 1000. * Units.ft
    segment.climb_rate      = 500.  * Units['ft/min']
    segment.air_speed_start = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.air_speed_end   = 110.  * Units['mph']                                            

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift  


    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Third Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------

    segment     = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses )

    segment.altitude  = 1000.0 * Units.ft
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 60.    * Units.miles                       

    segment.state.unknowns.propeller_power_coefficient = 0.02 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.40 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift    


    # add to misison
    mission.append_segment(segment)     

    return mission



# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results,vec_configs,line_color='bo-'):  
    fig =  plt.figure("Battery",figsize=(8,10))
    fig.set_size_inches(12, 10)
    for i in range(len(results.segments)):  
    
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        power    = results.segments[i].conditions.propulsion.battery_draw[:,0] 
        energy   = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        volts    = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]     
        current = results.segments[i].conditions.propulsion.current[:,0]      
        battery_amp_hr = (energy*0.000277778)/volts
        C_rating   = current/battery_amp_hr
        
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, -power, line_color)
        axes.set_ylabel('Battery Power (Watts)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        axes.grid(True)       
    
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, energy*0.000277778, line_color)
        axes.set_ylabel('Battery Energy (W-hr)' )
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, volts, 'bo-',label='Under Load')
        axes.plot(time,volts_oc, 'ks--',label='Open Circuit')
        axes.set_xlabel('Time (mins)' )
        axes.set_ylabel('Battery Voltage (Volts)' )  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        if i == 0:
            axes.legend(loc='upper right')          
        axes.grid(True)         
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, C_rating, line_color)
        axes.set_xlabel('Time (mins)' )
        axes.set_ylabel('C-Rating (C)' )  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True) 
 
    # ------------------------------------------------------------------
    #   Electronic Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Lift_Cruise_Electric_Conditions")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time           = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta            = results.segments[i].conditions.propulsion.throttle[:,0]
        eta_l          = results.segments[i].conditions.propulsion.throttle_lift[:,0]
        energy         = results.segments[i].conditions.propulsion.battery_energy[:,0]*0.000277778
        specific_power = results.segments[i].conditions.propulsion.battery_specfic_power[:,0]
        volts          = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc       = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]  
                    
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, eta, 'bo-',label='Forward Motor')
        axes.plot(time, eta_l, 'r^-',label='Lift Motors')
        axes.set_ylabel('Throttle')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
        plt.ylim((0,1))
        if i == 0:
            axes.legend(loc='upper center')         
    
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, energy, 'bo-')
        axes.set_ylabel('Battery Energy (W-hr)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, volts, 'bo-',label='Under Load')
        axes.plot(time,volts_oc, 'ks--',label='Open Circuit')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Battery Voltage (Volts)')  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)
        if i == 0:
            axes.legend(loc='upper center')                
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, specific_power, 'bo-') 
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Specific Power')  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
         
   
    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Prop-Rotor Network")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        prop_rpm     = results.segments[i].conditions.propulsion.rpm_forward [:,0] 
        prop_thrust  = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        prop_torque  = results.segments[i].conditions.propulsion.motor_torque_forward[:,0]
        prop_effp    = results.segments[i].conditions.propulsion.propeller_efficiency_forward[:,0]
        prop_effm    = results.segments[i].conditions.propulsion.motor_efficiency_forward[:,0]
        prop_Cp      = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]
        rotor_rpm    = results.segments[i].conditions.propulsion.rpm_lift[:,0] 
        rotor_thrust = -results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        rotor_torque = results.segments[i].conditions.propulsion.motor_torque_lift
        rotor_effp   = results.segments[i].conditions.propulsion.propeller_efficiency_lift[:,0]
        rotor_effm   = results.segments[i].conditions.propulsion.motor_efficiency_lift[:,0] 
        rotor_Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient_lift[:,0]        
    
        axes = fig.add_subplot(2,3,1)
        axes.plot(time, prop_rpm, 'bo-')
        axes.plot(time, rotor_rpm, 'r^-')
        axes.set_ylabel('RPM')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
    
        axes = fig.add_subplot(2,3,2)
        axes.plot(time, prop_thrust, 'bo-')
        axes.plot(time, rotor_thrust, 'r^-')
        axes.set_ylabel('Thrust (N)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,3)
        axes.plot(time, prop_torque, 'bo-' )
        axes.plot(time, rotor_torque, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Torque (N-m)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,4)
        axes.plot(time, prop_effp, 'bo-' )
        axes.plot(time, rotor_effp, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Propeller Efficiency, $\eta_{propeller}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)           
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,5)
        axes.plot(time, prop_effm, 'bo-' )
        axes.plot(time, rotor_effm, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Motor Efficiency, $\eta_{motor}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)         
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,6)
        axes.plot(time, prop_Cp, 'bo-' )
        axes.plot(time, rotor_Cp, 'r^-'  )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Coefficient')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')  
        axes.grid(True) 
         
            
    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Rotor")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.rpm_lift [:,0] 
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        torque = results.segments[i].conditions.propulsion.motor_torque_lift[:,0]
        effp   = results.segments[i].conditions.propulsion.propeller_efficiency_lift[:,0]
        effm   = results.segments[i].conditions.propulsion.motor_efficiency_lift[:,0] 
        Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient_lift[:,0]
    
        axes = fig.add_subplot(2,3,1)
        axes.plot(time, rpm, 'r^-')
        axes.set_ylabel('RPM')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
    
        axes = fig.add_subplot(2,3,2)
        axes.plot(time, -thrust, 'r^-')
        axes.set_ylabel('Thrust (N)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,3)
        axes.plot(time, torque, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Torque (N-m)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,4)
        axes.plot(time, effp, 'r^-',label= r'$\eta_{rotor}$' ) 
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Propeller Efficiency $\eta_{rotor}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        #if i == 0:
            #axes.legend(loc='upper center')   
        axes.grid(True)           
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,5)
        axes.plot(time, effm, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Motor Efficiency $\eta_{mot}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        plt.ylim((0,1))
        axes.grid(True)  
    
        axes = fig.add_subplot(2,3,6)
        axes.plot(time, Cp , 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Coefficient')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')    
        axes.grid(True)            
        
        
    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Propeller")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.rpm_forward [:,0] 
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        torque = results.segments[i].conditions.propulsion.motor_torque_forward
        effp   = results.segments[i].conditions.propulsion.propeller_efficiency_forward[:,0]
        effm   = results.segments[i].conditions.propulsion.motor_efficiency_forward[:,0]
        Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]
    
        axes = fig.add_subplot(2,3,1)
        axes.plot(time, rpm, 'bo-')
        axes.set_ylabel('RPM')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
    
        axes = fig.add_subplot(2,3,2)
        axes.plot(time, thrust, 'bo-')
        axes.set_ylabel('Thrust (N)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,3)
        axes.plot(time, torque, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Torque (N-m)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,4)
        axes.plot(time, effp, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Propeller Efficiency $\eta_{propeller}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)           
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,5)
        axes.plot(time, effm, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Motor Efficiency $\eta_{motor}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)         
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,6)
        axes.plot(time, Cp, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Coefficient')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')  
        axes.grid(True)  
        
    return

def load_stopped_rotor_results():
    return SUAVE.Input_Output.SUAVE.load('results_stopped_rotor.res')

def save_stopped_rotor_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_stopped_rotor.res')
    return
 

if __name__ == '__main__': 
    main()   
    plt.show(block=True) 
