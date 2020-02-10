# mission_Tiltwing.py
# 
# Created: Feb 2020, M Clarke

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data  
import sys
import pylab as plt
import numpy as np  

sys.path.append('../Vehicles')
# the analysis functions

from Tiltwing      import vehicle_setup, configs_setup 

# ----------------------------------------------------------------------
#   Setup
# ----------------------------------------------------------------------
def full_setup():    
    
    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission           = mission_setup(configs_analyses,vehicle)
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
    weights = SUAVE.Analyses.Weights.Weights_Electric_Vectored_Thrust()
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

    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row 
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.05 * ones_row(1) 
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
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "Departure"

    segment.analyses.extend( analyses.hover_climb ) 
    segment.altitude_start  = 0.0  * Units.ft
    segment.altitude_end    = 40.  * Units.ft
    segment.climb_rate      = 300. * Units['ft/min']
    segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy  
    
    segment.state.unknowns.propeller_power_coefficient = 0.04 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.8 * ones_row(1)
    
    segment.process.iterate.unknowns.network          = vehicle.propulsors.propulsor.unpack_unknowns 
    segment.process.iterate.residuals.network         = vehicle.propulsors.propulsor.residuals   
    segment.process.iterate.unknowns.mission          = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Hover Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment     = Segments.Hover.Hover(base_segment)
    segment.tag = "Hover"

    segment.analyses.extend( analyses.hover )
 
    segment.altitude        = 40.  * Units.ft
    segment.time            = 2*60

    segment.state.unknowns.propeller_power_coefficient      = 0.01 * ones_row(1)     
    segment.state.unknowns.throttle                         = 0.42 * ones_row(1)
    
    segment.process.iterate.unknowns.network          = vehicle.propulsors.propulsor.unpack_unknowns 
    segment.process.iterate.residuals.network         = vehicle.propulsors.propulsor.residuals   
    segment.process.iterate.unknowns.mission          = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment) 
    
    # ------------------------------------------------------------------
    #   First Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    
    segment     = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "Climb"
    
    segment.analyses.extend(analyses.cruise)
    
    segment.climb_rate       = 600. * Units['ft/min']
    segment.air_speed_start  = 85.   * Units['mph']
    segment.air_speed_end    = 110.   * Units['mph']
    segment.altitude_start   = 40.0 * Units.ft
    segment.altitude_end     = 1000.0 * Units.ft               
    
    segment.state.unknowns.propeller_power_coefficient = 0.03 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.80 * ones_row(1)
    
    segment.process.iterate.unknowns.network        = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network       = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
                
    # ------------------------------------------------------------------
    #   First Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    
    segment     = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "Cruise"
    
    segment.analyses.extend(analyses.cruise)
    
    segment.altitude  = 1000.0 * Units.ft
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 30.    * Units.miles                       
    
    segment.state.unknowns.propeller_power_coefficient = 0.03 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.60 * ones_row(1)
    
    segment.process.iterate.unknowns.network        = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network       = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
                
    # ------------------------------------------------------------------
    #   First Descent Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    
    segment     = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "Descent"
    
    segment.analyses.extend(analyses.cruise)
    segment.climb_rate       = -600. * Units['ft/min']
    segment.air_speed_start  = 110.   * Units['mph']
    segment.air_speed_end    = 85.   * Units['mph']
    segment.altitude_start   = 1000.0 * Units.ft
    segment.altitude_end     = 40.0 * Units.ft
    
    segment.process.iterate.unknowns.network        = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network       = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
     
    # ------------------------------------------------------------------
    #   Descent Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment     = Segments.Hover.Descent(base_segment)
    segment.tag = "Arrival"

    segment.analyses.extend( analyses.hover_descent )

    segment.altitude_start  = 40.0  * Units.ft
    segment.altitude_end    = 0.  * Units.ft
    segment.descent_rate    = 300. * Units['ft/min']  
    
    segment.state.unknowns.propeller_power_coefficient = 0.04 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.8 * ones_row(1)
    
    segment.process.iterate.unknowns.network          = vehicle.propulsors.propulsor.unpack_unknowns 
    segment.process.iterate.residuals.network         = vehicle.propulsors.propulsor.residuals   
    segment.process.iterate.unknowns.mission          = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

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


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results): 
    
    axis_font = {'size':'14'} 
    line_color = 'bo-'
    fig = plt.figure( )
    fig.set_size_inches(12, 10)
    for i in range(len(results.segments)):  
    
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        power    = results.segments[i].conditions.propulsion.battery_draw[:,0] 
        eta      = results.segments[i].conditions.propulsion.throttle[:,0]
        energy   = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        volts    = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]     
        current = results.segments[i].conditions.propulsion.current[:,0]      
        battery_amp_hr = (energy*0.000277778)/volts
        C_rating   = current/battery_amp_hr 
        
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, eta, 'bo-' ) 
        axes.set_ylabel('Throttle')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
        plt.ylim((0,1)) 
            
    
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, energy*0.000277778, line_color)
        axes.set_ylabel('Battery Energy (W-hr)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, volts, 'bo-',label='Under Load')
        axes.plot(time,volts_oc, 'ks--',label='Open Circuit')
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Battery Voltage (Volts)',axis_font)  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        if i == 0:
            axes.legend(loc='upper right')          
        axes.grid(True)         
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, C_rating, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('C-Rating (C)',axis_font)  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True) 
        
        
    fig = plt.figure( )
    fig.set_size_inches(12, 10)  
    for segment in results.segments.values(): 

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        rpm    = segment.conditions.propulsion.rpm[:,0] 
        thrust = segment.conditions.frames.body.thrust_force_vector[:,2]
        torque = segment.conditions.propulsion.motor_torque[:,0] 
        tm     = segment.conditions.propulsion.propeller_tip_mach[:,0]
 
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, -thrust, line_color)
        axes.set_ylabel('Thrust (N)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
        
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, rpm, line_color)
        axes.set_ylabel('RPM',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)      
        
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, torque, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Torque (N-m)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, tm, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Tip Mach',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True) 
 
    fig = plt.figure( )
    fig.set_size_inches(12, 10)  
    for segment in results.segments.values(): 

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        effp   = segment.conditions.propulsion.etap[:,0]
        effm   = segment.conditions.propulsion.etam[:,0]
        
        axes = fig.add_subplot(1,2,1)
        axes.plot(time, effp, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Propeller Efficiency',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)           
        plt.ylim((0,1))
        
        axes = fig.add_subplot(1,2,2)
        axes.plot(time, effm, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Motor Efficiency',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)
         
    return
 
