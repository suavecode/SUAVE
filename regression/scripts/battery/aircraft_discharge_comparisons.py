
    # ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
import numpy as np
import matplotlib.pyplot as plt 
import copy, time
import pickle 
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import  initialize_from_module_packaging 
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass , size_optimal_motor
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():  
    '''This scrpt runs a specified mission for n times, n = the number 
    of days'''
    ti = time.time()
    # Inputs 
    days                 = 1
    num_flights_per_day  = 1
    Climb_Rates          = np.array([500])* Units['ft/min']
    Climb_Speeds         = np.array([120]) * Units['mph']  
    Descent_Rates        = np.array([300])* Units['ft/min']
    Descent_Speeds       = np.array([110])* Units['mph'] 
    battery_fidelity     = 3
    Nominal_Range        = 90 * Units['miles']
    Nominal_Flight_Time  = 29.3*(Nominal_Range/ Units['miles']) + 38.1
    Cruise_Altitude      = 2500 * Units.feet
    
    unknowns  = np.array([0.85, 0.3, 0.6, 0.4, 0.5, 0.3])
 
                         
    configs, analyses = full_setup(days,num_flights_per_day,Climb_Rates[0],Descent_Rates[0],Climb_Speeds[0],Descent_Speeds[0], \
                                    Cruise_Altitude, Nominal_Range , Nominal_Flight_Time ,battery_fidelity,unknowns) 
    simple_sizing(configs) 
    configs.finalize()
    analyses.finalize()     
    mission = analyses.missions.base
    results = mission.evaluate() 
    
    tf = time.time()
    print ('Time taken: ' + str(round((tf-ti)/60,4)) + ' mins')
    plot_mission(results)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(days,num_flights,Climb_Rate,Descent_Rate , Climb_Speed , Descent_Speed,\
                Cruise_Altitude, Nominal_Range , Nominal_Flight_Time ,battery_fidelity,unknowns):

    # vehicle data
    vehicle  = vehicle_setup(battery_fidelity,days)
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses, days,num_flights, vehicle,Climb_Rate, Descent_Rate , Climb_Speed,Descent_Speed,\
                              Cruise_Altitude, Nominal_Range , Nominal_Flight_Time ,battery_fidelity, unknowns) 
        
    
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
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
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
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup(battery_fidelity,days):

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'X57_Maxwell' 

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff   = 3000. * Units.pounds
    vehicle.mass_properties.takeoff       = 3000. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = 3000. * Units.pounds
    vehicle.mass_properties.cargo         = 0. 

    # envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.8

    # basic parameters
    vehicle.reference_area         = 15.45  
    vehicle.passengers             = 4

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing' 
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 15.45 * Units['meters**2']  
    wing.spans.projected         = 11. * Units.meter   
    wing.chords.root             = 1.67 * Units.meter  
    wing.chords.tip              = 1.14 * Units.meter  
    wing.chords.mean_aerodynamic = 1.47 * Units.meter   
    wing.taper                   = wing.chords.root/wing.chords.tip 
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 1.5 * Units.degrees 
    wing.origin                  = [[2.032, 0., 0.]]
    wing.aerodynamic_center      = [0.558, 0., 0.] 
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True 
    wing.dynamic_pressure_ratio  = 1.0 
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.95
    wing.areas.reference         = 3.74 * Units['meters**2']  
    wing.spans.projected         = 3.454  * Units.meter 
    wing.sweeps.quarter_chord    = 12.5 * Units.deg 
    wing.chords.root             = 1.397 * Units.meter 
    wing.chords.tip              = 0.762 * Units.meter 
    wing.chords.mean_aerodynamic = 1.09 * Units.meter 
    wing.taper                   = wing.chords.root/wing.chords.tip 
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference  
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees 
    wing.origin                  = [[6.248, 0., 0.784]] 
    wing.aerodynamic_center      = [0.558, 0., 0.]
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False 
    wing.dynamic_pressure_ratio  = 0.9

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    

    wing.sweeps.quarter_chord    = 25. * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 2.258 * Units['meters**2']  
    wing.spans.projected         = 1.854   * Units.meter  
    wing.chords.root             = 1.6764 * Units.meter 
    wing.chords.tip              = 0.6858 * Units.meter 
    wing.chords.mean_aerodynamic = 1.21 * Units.meter 
    wing.taper                   = wing.chords.root/wing.chords.tip 
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees 
    wing.origin                  = [[6.01 ,0,  0.623]] 
    wing.aerodynamic_center      = [0.508 ,0,0]  
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False 
    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)



    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------ 
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage' 
    fuselage.seats_abreast                      = 2. 
    fuselage.fineness.nose                      = 1.6
    fuselage.fineness.tail                      = 2. 
    fuselage.lengths.nose                       = 60.  * Units.inches
    fuselage.lengths.tail                       = 161. * Units.inches
    fuselage.lengths.cabin                      = 105. * Units.inches
    fuselage.lengths.total                      = 332.2* Units.inches
    fuselage.lengths.fore_space                 = 0.
    fuselage.lengths.aft_space                  = 0.   
    fuselage.width                              = 42. * Units.inches 
    fuselage.heights.maximum                    = 62. * Units.inches
    fuselage.heights.at_quarter_length          = 62. * Units.inches
    fuselage.heights.at_three_quarters_length   = 62. * Units.inches
    fuselage.heights.at_wing_root_quarter_chord = 23. * Units.inches 
    fuselage.areas.side_projected               = 8000.  * Units.inches**2.
    fuselage.areas.wetted                       = 30000. * Units.inches**2.
    fuselage.areas.front_projected              = 42.* 62. * Units.inches**2. 
    fuselage.effective_diameter                 = 50. * Units.inches


    # Segment  
    segment                                     = SUAVE.Components.Fuselages.Segment() 
    segment.tag                                 = 'segment_0'  
    segment.percent_x_location                  = 0 
    segment.percent_z_location                  = 0 
    segment.height                              = 0.01 
    segment.width                               = 0.01 
    fuselage.Segments.append(segment)             
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_1'   
    segment.percent_x_location                  = 0.007279116466
    segment.percent_z_location                  = 0.002502014453
    segment.height                              = 0.1669064748
    segment.width                               = 0.2780205877
    fuselage.Segments.append(segment)    

    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_2'   
    segment.percent_x_location                  = 0.01941097724 
    segment.percent_z_location                  = 0.001216095397 
    segment.height                              = 0.3129496403 
    segment.width                               = 0.4365777215 
    fuselage.Segments.append(segment)         

    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_3'   
    segment.percent_x_location                  = 0.06308567604
    segment.percent_z_location                  = 0.007395489231
    segment.height                              = 0.5841726619
    segment.width                               = 0.6735119903
    fuselage.Segments.append(segment)      
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_4'   
    segment.percent_x_location                  = 0.1653761217 
    segment.percent_z_location                  = 0.02891281352 
    segment.height                              = 1.064028777 
    segment.width                               = 1.067200529 
    fuselage.Segments.append(segment)  
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_5'    
    segment.percent_x_location                  = 0.2426372155 
    segment.percent_z_location                  = 0.04214148761 
    segment.height                              = 1.293766653 
    segment.width                               = 1.183058255 
    fuselage.Segments.append(segment)  
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_6'    
    segment.percent_x_location                  = 0.2960174029  
    segment.percent_z_location                  = 0.04705241831  
    segment.height                              = 1.377026712  
    segment.width                               = 1.181540054  
    fuselage.Segments.append(segment)  
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_7'   
    segment.percent_x_location                  = 0.3809404284 
    segment.percent_z_location                  = 0.05313580461 
    segment.height                              = 1.439568345 
    segment.width                               = 1.178218989 
    fuselage.Segments.append(segment)    
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_8'   
    segment.percent_x_location                  = 0.5046854083 
    segment.percent_z_location                  = 0.04655492473 
    segment.height                              = 1.29352518 
    segment.width                               = 1.054390707 
    fuselage.Segments.append(segment)   
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_9'   
    segment.percent_x_location                  = 0.6454149933 
    segment.percent_z_location                  = 0.03741966266 
    segment.height                              = 0.8971223022 
    segment.width                               = 0.8501926505   
    fuselage.Segments.append(segment)  
      
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_10'   
    segment.percent_x_location                  = 0.985107095 
    segment.percent_z_location                  = 0.04540283436 
    segment.height                              = 0.2920863309 
    segment.width                               = 0.2012565415  
    fuselage.Segments.append(segment)         
       
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_11'   
    segment.percent_x_location                  = 1 
    segment.percent_z_location                  = 0.04787575562  
    segment.height                              = 0.1251798561 
    segment.width                               = 0.1206021048 
    fuselage.Segments.append(segment)             
    
    # add to vehicle
    vehicle.append_component(fuselage)   

    #---------------------------------------------------------------------------------------------
    # DEFINE PROPELLER
    #---------------------------------------------------------------------------------------------
    # build network    
    net = Battery_Propeller() 
    net.number_of_engines       = 2.
    net.nacelle_diameter        = 14 * Units.inches
    net.engine_length           = 3  * Units.feet
    net.areas                   = Data() 
    net.areas.wetted            = net.engine_length*(2*np.pi*net.nacelle_diameter/2)  

    # Component 1 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc

    # Component 2 the Propeller
    # Design the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller() 
    
    prop.number_of_blades    = 3
    prop.freestream_velocity = 135.*Units['mph']    
    prop.angular_velocity    = 1400.  * Units.rpm
    prop.tip_radius          = 2.75* Units.feet
    prop.hub_radius          = 0.4* Units.feet
    prop.design_Cl           = 0.8
    prop.design_altitude     = 8000. * Units.feet
    prop.design_altitude     = 8000. * Units.feet
    prop.design_thrust       = 1000.   
    prop.origin              = [[2.,2.5,0.784],[2.,-2.5,0.784]]  #  prop influence            
    prop.rotation            = [1,-1]
    prop                     = propeller_design(prop)    
    net.propeller            = prop    

    # Component 8 the Battery  
    bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()  
    bat.cell.max_mission_discharge     = 9.        # Amps  
    bat.cell.max_discharge_rate        = 15.       # Amps   
    bat.cell.surface_area              = (np.pi*bat.cell.height*bat.cell.diameter)  
    bat.pack_config.series             =  140 # 160 # 140 # 128   
    bat.pack_config.parallel           =  48  # 45  # 45  # 40     
    bat.module_config.normal_count     =  16  # 20  # 15  # 16     
    bat.module_config.parallel_count   =  30  # 25  # 28  # 20      
    bat.age_in_days                    = days  
    initialize_from_module_packaging(bat)       
    net.battery                        = bat
    net.voltage                        = bat.max_voltage
    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Propeller  motor
    # Component 4 the Motor
    motor                      = SUAVE.Components.Energy.Converters.Motor()
    motor.mass_properties.mass = 10. * Units.kg
    motor.origin               = prop.origin  
    motor.efficiency           = 0.95
    motor.gear_ratio           = 1. 
    motor.gearbox_efficiency   = 1. # Gear box efficiency        
    motor.nominal_voltage      = bat.max_voltage *3/4  
    motor.propeller_radius     = prop.tip_radius  
    motor.no_load_current      = 4.0 
    motor                      = size_optimal_motor(motor,prop) 
    net.motor                  = motor 
    
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 5. # Watts 
    payload.mass_properties.mass = 1.0 * Units.kg
    net.payload                  = payload

    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 5. #Watts  
    net.avionics        = avionics      

    # add the solar network to the vehicle
    vehicle.append_component(net)          

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle

# ---------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle): 
    configs = SUAVE.Components.Configs.Config.Container() 
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    base_config.propulsors.propulsor.pitch_command = 0 
    configs.append(base_config)  
    
    return configs


def compile_results(results): 
    
    Aging_time  = []
    C_Fade      = []
    R_Growth    = []
    Charge      = []
    for i in range(len(results.segments)):  
        if results.segments[i].battery_discharge == False: 
            Aging_time.append(results.segments[i].conditions.frames.inertial.time[-1,0] /Units.hr)
            C_Fade.append(results.segments[i].conditions.propulsion.battery_capacity_fade_factor) 
            Charge.append(results.segments[i].conditions.propulsion.battery_cumulative_charge_throughput[:,0][-1])
            R_Growth.append(results.segments[i].conditions.propulsion.battery_resistance_growth_factor)   
    
    res             = Data()
    res.C_Fade      = C_Fade  
    res.R_Growth    = R_Growth 
    res.Aging_time  = Aging_time 
    res.Charge      = Charge 
        
    return res 

def plot_mission(results): 
    n_parallel =  50
    line_color = 'ks-'
    line_width = 4        
    width  = 8
    height = 6  
    lp = 35 
    
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.linewidth'] = 2.
    
    fig1 = plt.figure('Altitude')
    fig2 = plt.figure('Airspeed')
    fig3 = plt.figure('SOC')
    fig4 = plt.figure('Battery_Voltage')
    fig5 = plt.figure('Battery_Pack_Temp')
    fig6 = plt.figure('C_Rate')
    fig7 = plt.figure('Currrent')
    fig8 = plt.figure('Bar_Chart')
    
    fig1.set_size_inches(width,height) 
    fig2.set_size_inches(width,height) 
    fig3.set_size_inches(width,height) 
    fig4.set_size_inches(width,height) 
    fig5.set_size_inches(width,height) 
    fig6.set_size_inches(width,height) 
    fig7.set_size_inches(width,height)
    
    ax1 = fig1.add_subplot(1,1,1)
    ax2 = fig2.add_subplot(1,1,1)
    ax3 = fig3.add_subplot(1,1,1)
    ax4 = fig4.add_subplot(1,1,1)
    ax5 = fig5.add_subplot(1,1,1) 
    ax6 = fig6.add_subplot(1,1,1)
    ax7 = fig7.add_subplot(1,1,1)
     
    dim_seg   = len(results.segments)
    charge_ah = np.zeros(dim_seg-1)
    q_seg     = np.zeros(dim_seg-1)
    
    for i in range(dim_seg):  
        time           = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        velocity       = results.segments[i].conditions.freestream.velocity[:,0]
        density        = results.segments[i].conditions.freestream.density[:,0]
        EAS            = velocity * np.sqrt(density/1.225)
        altitude       = results.segments[i].conditions.freestream.altitude[:,0] / Units.feet  
        power          = results.segments[i].conditions.propulsion.battery_power_draw[:,0] 
        energy         = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        volts          = results.segments[i].conditions.propulsion.battery_voltage_under_load [:,0] 
        volts_oc       = results.segments[i].conditions.propulsion.battery_voltage_open_circuit[:,0]  
        charge         = results.segments[i].conditions.propulsion.battery_cumulative_charge_throughput[:,0] 
        bat_pack_temp  = results.segments[i].conditions.propulsion.battery_pack_temperature[:,0] 
        current        = results.segments[i].conditions.propulsion.battery_current[:,0]      
        SOC            = results.segments[i].conditions.propulsion.battery_state_of_charge[:,0]
        battery_amp_hr = (energy/ Units.Wh )/volts  
        C_rating       = current/battery_amp_hr  
        
        if i < dim_seg-1: 
            q_seg[i]     =  np.mean(results.segments[i].conditions.propulsion.battery_cell_heat_energy_generated[:,0])
            charge_ah[i]         =  results.segments[i].conditions.propulsion.battery_charge_throughput[-1,0] 
        
        ax1.plot(time, altitude, line_color, linewidth= line_width) 
        ax1.set_xlabel('Time (mins)')
        ax1.set_ylabel('Altitude (ft)')
 
        ax2.plot( time , EAS / Units['mph']   , line_color, linewidth= line_width )          
        ax2.set_xlabel('Time (mins)')
        ax2.set_ylabel('Equivalent Airspeed (mph)') 
     
        ax3.plot(time, SOC, line_color, linewidth= line_width) 
        ax3.set_xlabel('Time (mins)')
        ax3.set_ylabel('State of Charge'  ) 
     
        ax4.plot(time, volts, line_color, linewidth= line_width)  
        ax4.set_ylabel('Voltage (V)' )  
        ax4.set_xlabel('Time (mins)') 
     
        ax5.plot(time, bat_pack_temp, line_color, linewidth= line_width)       
        ax5.set_xlabel('Time (mins)')
        ax5.set_ylabel('Tempertature ($\degree$C)' )  
           
        ax6.plot(time, C_rating, line_color, linewidth= line_width) 
        ax6.set_xlabel('Time (mins)')
        ax6.set_ylabel('C-Rate (C)')  
        
        ax7.plot(time, current/n_parallel, line_color, linewidth= line_width) 
        ax7.set_xlabel('Time (mins)')
        ax7.set_ylabel('Cell Current (A)')         
        
    
    ## plot results
    #fig8.set_size_inches(12,height) 
    #ax8 = fig8.add_subplot(1,1,1) 
    #ax9 = ax8.twinx()
    #bar_width = 0.35  
    #r1 = np.arange(len(charge_ah))
    #r2 = [x + bar_width for x in r1]      
    #pop = ax8.bar(r1, charge_ah, width=bar_width, color='k', align='center')  
    #gdp = ax9.bar(r2, q_seg, width=bar_width,color='grey',align='center')   
    #plt.xticks([r + bar_width*0.5 for r in range(len(charge_ah))], ['TO','DER','ICA', 'CL', 'CR','D','DL','RCL','RCR','RD','BL','FA','L','RT'])
    #ax8.set_xlabel('Flight Profile Segments')
    #ax8.set_ylabel('NMC Cell Q (Ah)') 
    #ax9.set_ylabel('Module Heat (W)')    
    #ax8.set_yscale('log')
    #ax9.set_yscale('log')
    #plt.legend([pop, gdp],['Charge Throughput', 'Generated Module Heat'],loc='upper right')  
    
    
    #plt.grid(zorder=0, linestyle='dotted')  
    
    plot_aircraft_velocities(results)
    plot_aerodynamic_coefficients(results)
    plot_aerodynamic_forces(results)  
    plot_flight_conditions(results)     
    plot_propeller_conditions(results)
    plot_battery_pack_conditions(results) 
    plot_battery_age_conditions(results) 
    plot_battery_cell_conditions(results) 

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
 
def mission_setup(analyses, days, num_flights, vehicle, Climb_Rate, Descent_Rate , Climb_Speed,\
                  Descent_Speed, Cruise_Altitude, Nominal_Range , Nominal_Flight_Time , battery_fidelity, ukns):
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
    mission.airport    = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 

    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points             = 10 
    base_segment.use_Jacobian                                     = False
    base_segment.state.numerics.jacobian_evaluations              = 0 
    base_segment.state.numerics.iterations                        = 0      
    base_segment.state.reverse_thrust                             = False 
    base_segment.process.iterate.initials.initialize_battery      = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery    
    base_segment.process.finalize.post_process.update_battery_age = SUAVE.Methods.Missions.Segments.Common.Energy.update_battery_age         
    base_segment.process.iterate.conditions.planet_position       = SUAVE.Methods.skip       
    base_segment.state.residuals.network                          = 0.    * ones_row(4) 
    bat                                                           = vehicle.propulsors.propulsor.battery  
    base_segment.initial_mission_battery_energy                   = bat.initial_max_energy    
    base_segment.charging_SOC_cutoff                              = bat.cell.charging_SOC_cutoff  
    base_segment.charging_voltage                                 = bat.cell.charging_voltage  * bat.pack_config.series
    base_segment.charging_current                                 = bat.cell.charging_current  * bat.pack_config.parallel
    base_segment.battery_configuration                            = bat.pack_config
    base_segment.max_energy                                       = bat.max_energy   
    
    
    # Determine Stall Speed 
    m     = vehicle.mass_properties.max_takeoff
    g     = 9.81
    S     = vehicle.reference_area
    atmo  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho   = atmo.compute_values(1000.*Units.feet,0.).density
    CLmax = 1.2 
    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax))) 
    
    # Calculate Flight Parameters 
    ICA_Altitude        = 500 * Units.feet 
    Downleg_Altitude    = 1000 * Units.feet  
    Climb_Altitude      = Cruise_Altitude  - ICA_Altitude
    Descent_Altitude    = Cruise_Altitude - Downleg_Altitude  
     
    Climb_Range_Speed   = np.sqrt(Climb_Speed**2 - Climb_Rate**2)
    Climb_Time          = Climb_Altitude/Climb_Rate 
    Climb_Range         = Climb_Range_Speed * Climb_Time 
    Descent_Range_Speed = np.sqrt(Descent_Speed**2 - Descent_Rate**2)
    Descent_Time        = Descent_Altitude /(Descent_Rate )
    Descent_Range       = Descent_Range_Speed * Descent_Time 
    Cruise_Distance     = Nominal_Range - (Climb_Range + Descent_Range)
    Cruise_Time         = Nominal_Flight_Time - (Climb_Time+Descent_Time )
    Cruise_Speed        = Cruise_Distance/Cruise_Time  
    
    
    # Calculate Flight Parameters 
    Downleg_Distance        = 6000 * Units.feet # length or runway
    Downleg_Altitude        = 1000 * Units.feet
    Res_Cruise_Altitude     = 1500 * Units.feet  
    Res_Climb_Altitude      = Res_Cruise_Altitude - Downleg_Altitude 
    Res_Descent_Altitude    = Res_Cruise_Altitude - Downleg_Altitude
    Res_Nominal_Range       = 0.1 * Nominal_Range  
     
    Res_Climb_Range_Speed   = np.sqrt(Climb_Speed**2 - Climb_Rate**2)
    Res_Climb_Time          = Res_Climb_Altitude/Climb_Rate 
    Res_Climb_Range         = Res_Climb_Range_Speed * Res_Climb_Time 
    Res_Descent_Range_Speed = np.sqrt(Descent_Speed**2 - Descent_Rate**2)
    Res_Descent_Time        = Res_Descent_Altitude/(Descent_Rate )
    Res_Descent_Range       = Res_Descent_Range_Speed * Res_Descent_Time  
    Res_Cruise_Distance     = Res_Nominal_Range - (Res_Climb_Range + Res_Descent_Range) 
    Res_Cruise_Speed        = Cruise_Speed 
                        
    # Calculate Flight Pameters 
    print('\n\n Climb Rate: ' + str(round(Climb_Rate / Units['ft/min'],4)) + ' ft/min')
    print('Descent Rate: ' + str(round(Descent_Rate / Units['ft/min'],4)) + ' ft/min \n') 
    print('Cruise Speed : ' + str(round(Cruise_Speed,4)) + ' m/s')
    print('Cruise Distance : ' + str(round(Cruise_Distance,4))+ ' m \n ')
    print('Reserve Cruise Speed : ' + str(round(Res_Cruise_Speed,4)) + ' m/s')
    print('Reserve Cruise Distance : ' + str(round(Res_Cruise_Distance,4))+ ' m \n \n')
    

    # Unpack 
    cl_throt_ukn  = ukns[0]
    cl_CP_ukn     = ukns[1]
    cr_throt_ukn  = ukns[2]
    cr_CP_ukn     = ukns[3]
    des_throt_ukn = ukns[4]
    des_CP_ukn    = ukns[5]     
         
    for day in range(days):
        # compute daily temperature in san francisco: link: https://www.usclimatedata.com/climate/san-francisco/california/united-states/usca0987/2019/1
        # NY
        daily_temp = 6 -0.191*day + 4.59E-3*(day**2) - 2.02E-5*(day**3) + 2.48E-8*(day**4)
        
        # LA 
        daily_temp = 18.2 - 9.7E-3*(day) + 2.41E-4*(day**2) -7.74E-6*(day**3) \
                    + 1.38E-7*(day**4) - 1.01E-9*(day**5) + 3.67E-12*(day**6) \
                    - 6.78E-15*(day**7) + 5.1E-18*(day**8)
        
        # HOU 
        daily_temp = 17.1 - 0.0435*(day) + 2.77E-3*(day**2) -2.2E-5*(day**3) \
                    + 9.72E-8*(day**4) - 2.53E-10*(day**5) + 2.67E-13*(day**6)  
        
        # SF 
        daily_temp = 13.5 + (day)*(-0.00882) + (day**2)*(0.00221) + \
                    (day**3)*(-0.0000314) + (day**4)*(0.000000185)  +\
                    (day**5)*(-0.000000000483)  + (day**6)*(4.57E-13)
        
        # CHI 
        daily_temp = -0.145 + (day)*(-0.11) + (day**2)*(4.57E-3) + \
                    (day**3)*(-2.71E-5) + (day**4)*(7.92E-8)  +\
                    (day**5)*(-1.66E-10)  + (day**6)*(1.76E-13)   
        
        daily_temp = 27
        for nf in range(num_flights):
            
            # Thevenin Discharge Model 
            if battery_fidelity == 3:        
                ## ------------------------------------------------------------------
                ##   Takeoff Segment  Flight 1   : 
                ## ------------------------------------------------------------------             
                #segment = Segments.Ground.Takeoff(base_segment)
                #segment_name = 'Takeoff Day ' + str (day)+ ' Flight ' + str(nf+1)
                #segment.tag = segment_name          
                #segment.analyses.extend( analyses.base )  
                #segment.state.numerics.number_control_points             = 40 
                #segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                #segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                #segment.state.unknowns.throttle                          = 1.0 * ones_row(1)   
                #segment.state.unknowns.propeller_power_coefficient       = 0.15 * ones_row(1)   
                #segment.state.unknowns.battery_state_of_charge           = 0.95  * ones_row(1) 
                #segment.state.unknowns.battery_current                   = 700 * ones_row(1)  
                #segment.state.unknowns.battery_cell_temperature          = (daily_temp + 1) * ones_row(1)  
                #segment.velocity_start                                   = Vstall*0.5
                #segment.velocity_end                                     = Vstall
                #segment.friction_coefficient                             = 0.3
                #segment.time                                             = 35.0
                #segment.battery_thevenin_voltage                         = 0 
                #segment.ambient_temperature                              = daily_temp  
                #segment.battery_age_in_days                              = day   
                #segment.battery_discharge                                = True 
                #segment.battery_cell_temperature                         = daily_temp + 1
                #segment.battery_pack_temperature                         = daily_temp + 1                   
                #if day == 0: 
                    #if nf == 0:                         
                        #segment.battery_resistance_growth_factor             = 1 
                        #segment.battery_capacity_fade_factor                 = 1    
                        #segment.battery_energy                               = bat.initial_max_energy    
                        #segment.battery_cumulative_charge_throughput         = 0                          
                    
                #mission.append_segment(segment)
                
                
                # ------------------------------------------------------------------
                #   Departure End of Runway Segment Flight 1 : 
                # ------------------------------------------------------------------ 
                segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
                segment_name = 'DER Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name          
                segment.analyses.extend( analyses.base ) 
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                segment.state.unknowns.throttle                          = 1.0 * ones_row(1)
                segment.state.unknowns.propeller_power_coefficient       = 0.2  * ones_row(1)  
                segment.state.unknowns.battery_state_of_charge           = 0.95 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)
                segment.altitude_start                                   = 0.0 * Units.feet
                segment.altitude_end                                     = 50.0 * Units.feet
                segment.air_speed_start                                  = Vstall  
                segment.air_speed_end                                    = 45      
                segment.ambient_temperature                              = daily_temp                  
                segment.climb_rate                                       = 700 * Units['ft/min']
                segment.battery_age_in_days                              = day   
                segment.battery_discharge                                = True 
                segment.battery_thevenin_voltage                         = 0    
                segment.battery_cell_temperature                         = daily_temp + 1
                segment.battery_pack_temperature                         = daily_temp + 1   
                
                if day == 0: 
                    if nf == 0:                         
                        segment.battery_resistance_growth_factor             = 1 
                        segment.battery_capacity_fade_factor                 = 1    
                        segment.battery_energy                               = bat.initial_max_energy    
                        segment.battery_cumulative_charge_throughput         = 0                          
                    
                mission.append_segment(segment)          
                # ------------------------------------------------------------------
                #   Initial Climb Area Segment Flight 1  
                # ------------------------------------------------------------------ 
                segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
                segment_name = 'ICA_AltitudeDay ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name          
                segment.analyses.extend( analyses.base ) 
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                segment.state.unknowns.throttle                          = cl_throt_ukn * ones_row(1)
                segment.state.unknowns.propeller_power_coefficient       = cl_CP_ukn  * ones_row(1)  
                segment.state.unknowns.battery_state_of_charge           = 0.8 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)
                segment.altitude_start                                   = 50.0 * Units.feet
                segment.altitude_end                                     = 500.0 * Units.feet
                segment.air_speed_start                                  = 45  
                segment.air_speed_end                                    = Climb_Speed           
                segment.climb_rate                                       = 600 * Units['ft/min']
                segment.battery_age_in_days                              = day   
                segment.ambient_temperature                              = daily_temp                  
                segment.battery_discharge                                = True                
                mission.append_segment(segment)
             
                         
                # ------------------------------------------------------------------
                #   Climb Segment Flight 1 
                # ------------------------------------------------------------------ 
                segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
                segment_name = 'Climb Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name          
                segment.analyses.extend( analyses.base ) 
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                segment.state.unknowns.throttle                          = cl_throt_ukn * ones_row(1)
                segment.state.unknowns.propeller_power_coefficient       = cl_CP_ukn  * ones_row(1)  
                segment.state.unknowns.battery_state_of_charge           = 0.8 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 200 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)
                segment.altitude_start                                   = 500.0 * Units.feet
                segment.altitude_end                                     = Cruise_Altitude
                segment.air_speed                                        = Climb_Speed
                segment.climb_rate                                       = Climb_Rate   
                segment.ambient_temperature                              = daily_temp                  
                segment.battery_age_in_days                              = day   
                segment.battery_discharge                                = True                
                mission.append_segment(segment)
                
                # ------------------------------------------------------------------
                #   Cruise Segment Flight 1 
                # ------------------------------------------------------------------ 
                segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
                segment_name = 'Cruise Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name       
                segment.analyses.extend(analyses.base)
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco   
                segment.air_speed                                        = Cruise_Speed
                segment.distance                                         = Cruise_Distance 
                segment.state.unknowns.throttle                          = cr_throt_ukn * ones_row(1)    
                segment.state.unknowns.propeller_power_coefficient       = cr_CP_ukn * ones_row(1) 
                segment.state.unknowns.battery_state_of_charge           = 0.6 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 200  * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1 ) * ones_row(1)               
                segment.battery_discharge                                = True   
                segment.ambient_temperature                              = daily_temp                  
                segment.battery_age_in_days                              = day         
                mission.append_segment(segment)    
                
                # ------------------------------------------------------------------
                #   Descent Segment Flight 1   
                # ------------------------------------------------------------------ 
                segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
                segment_name = 'Descent Day ' + str (day) + ' Flight ' + str(nf+1)
                segment.tag = segment_name  
                segment.analyses.extend( analyses.base )
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco      
                segment.altitude_end                                     = Downleg_Altitude
                segment.air_speed                                        = Descent_Speed
                segment.descent_rate                                     = Descent_Rate 
                segment.state.unknowns.throttle                          = des_throt_ukn  * ones_row(1)  
                segment.state.unknowns.propeller_power_coefficient       = des_CP_ukn     * ones_row(1) 
                segment.state.unknowns.battery_state_of_charge           = 0.5 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 200  * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)  	            
                segment.battery_discharge                                = True  
                segment.battery_age_in_days                              = day    
                segment.ambient_temperature                              = daily_temp                  
                mission.append_segment(segment)  
            
                # ------------------------------------------------------------------
                #  Downleg_Altitude Segment Flight 1 
                # ------------------------------------------------------------------ 
                segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
                segment_name = 'Downleg_Altitude Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name       
                segment.analyses.extend(analyses.base)
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco  
                segment.acceleration                                     = -0.05307 * Units['m/s/s']
                segment.air_speed_start                                  = Descent_Speed * Units['m/s']
                segment.air_speed_end                                    = 45.0 * Units['m/s']            
                segment.distance                                         = Downleg_Distance
                segment.state.unknowns.throttle                          = 0.5 * ones_row(1)    
                segment.state.unknowns.propeller_power_coefficient       = 0.3 * ones_row(1) 
                segment.state.unknowns.battery_state_of_charge           = 0.4 * ones_row(1)  
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1 ) * ones_row(1)               
                segment.battery_discharge                                = True  
                segment.battery_age_in_days                              = day   
                segment.ambient_temperature                              = daily_temp                  
                mission.append_segment(segment)     
                
                # ------------------------------------------------------------------
                #   Reserve Climb Segment Flight 1 
                # ------------------------------------------------------------------ 
                segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
                segment_name = 'Reserve Reserve Climb Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name          
                segment.analyses.extend( analyses.base ) 
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                segment.state.unknowns.throttle                          = cl_throt_ukn * ones_row(1)
                segment.state.unknowns.propeller_power_coefficient       = cl_CP_ukn  * ones_row(1)  
                segment.state.unknowns.battery_state_of_charge           = 0.4 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)
                segment.altitude_start                                   = Downleg_Altitude
                segment.altitude_end                                     = Res_Cruise_Altitude
                segment.air_speed                                        = Climb_Speed
                segment.climb_rate                                       = Climb_Rate  
                segment.battery_age_in_days                              = day    
                segment.ambient_temperature                              = daily_temp                  
                segment.battery_discharge                                = True                
                mission.append_segment(segment)
                
                # ------------------------------------------------------------------
                #   Reserve Cruise Segment Flight 1 
                # ------------------------------------------------------------------ 
                segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
                segment_name = 'Reserve Cruise Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name       
                segment.analyses.extend(analyses.base)
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco   
                segment.air_speed                                        = Res_Cruise_Speed
                segment.distance                                         = Res_Cruise_Distance
                segment.state.unknowns.throttle                          = cr_throt_ukn * ones_row(1)    
                segment.state.unknowns.propeller_power_coefficient       = cr_CP_ukn * ones_row(1) 
                segment.state.unknowns.battery_state_of_charge           = 0.4 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1 ) * ones_row(1)               
                segment.battery_discharge                                = True   
                segment.ambient_temperature                              = daily_temp                  
                segment.battery_age_in_days                              = day         
                mission.append_segment(segment)    
                
                # ------------------------------------------------------------------
                #    Reserve Descent Segment Flight 1  
                # ------------------------------------------------------------------ 
                segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
                segment_name = 'Reserve Descent Day ' + str (day) + ' Flight ' + str(nf+1)
                segment.tag = segment_name  
                segment.analyses.extend( analyses.base )
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco      
                segment.altitude_end                                     = Downleg_Altitude
                segment.air_speed                                        = Descent_Speed
                segment.descent_rate                                     = Descent_Rate 
                segment.state.unknowns.throttle                          = des_throt_ukn  * ones_row(1)  
                segment.state.unknowns.propeller_power_coefficient       = des_CP_ukn  * ones_row(1) 
                segment.state.unknowns.battery_state_of_charge           = 0.4 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)  	            
                segment.battery_discharge                                = True  
                segment.ambient_temperature                              = daily_temp  
                segment.battery_age_in_days                              = day   
                mission.append_segment(segment)  
            
                # ------------------------------------------------------------------
                #  Baseleg Segment Flight 1  
                # ------------------------------------------------------------------ 
                segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
                segment_name = 'Baseleg Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name          
                segment.analyses.extend( analyses.base ) 
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco
                segment.state.unknowns.throttle                          = des_throt_ukn * ones_row(1)
                segment.state.unknowns.propeller_power_coefficient       = des_CP_ukn  * ones_row(1)  
                segment.state.unknowns.battery_state_of_charge           = 0.3 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)
                segment.altitude_start                                   = Downleg_Altitude
                segment.altitude_end                                     = 500.0 * Units.feet
                segment.air_speed_start                                  = 45 
                segment.air_speed_end                                    = 40    
                segment.climb_rate                                       = -350 * Units['ft/min']
                segment.battery_age_in_days                              = day  
                segment.ambient_temperature                              = daily_temp   
                segment.battery_discharge                                = True                
                mission.append_segment(segment) 
            
                # ------------------------------------------------------------------
                #  Final Approach Segment Flight 1  
                # ------------------------------------------------------------------ 
                segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
                segment_name = 'Final Approach Day ' + str (day)+ ' Flight ' + str(nf+1)
                segment.tag = segment_name          
                segment.analyses.extend( analyses.base ) 
                segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                segment.state.unknowns.throttle                          = des_throt_ukn * ones_row(1)
                segment.state.unknowns.propeller_power_coefficient       = des_CP_ukn * ones_row(1)  
                segment.state.unknowns.battery_state_of_charge           = 0.4 * ones_row(1) 
                segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)
                segment.altitude_start                                   = 500.0 * Units.feet
                segment.altitude_end                                     = 00.0 * Units.feet
                segment.air_speed_start                                  = 40 
                segment.air_speed_end                                    = 35   
                segment.climb_rate                                       = -300 * Units['ft/min']
                segment.battery_age_in_days                              = day   
                segment.ambient_temperature                              = daily_temp  
                segment.battery_discharge                                = True                
                mission.append_segment(segment)  
            
                ## ------------------------------------------------------------------
                ##   Landing Segment Flight 1  
                ## ------------------------------------------------------------------             
                #segment = Segments.Ground.Landing(base_segment)
                #segment_name = 'Landing Day ' + str (day)+ ' Flight ' + str(nf+1)
                #segment.tag = segment_name          
                #segment.analyses.extend( analyses.base ) 
                #segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                #segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                #segment.state.unknowns.throttle                          = des_throt_ukn * ones_row(1)
                #segment.state.unknowns.propeller_power_coefficient       = des_CP_ukn  * ones_row(1)    
                #segment.state.unknowns.battery_state_of_charge           = 0.3 * ones_row(1) 
                #segment.state.unknowns.battery_current                   = 550 * ones_row(1)  
                #segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1) 
                #segment.velocity_start                                   = Vstall
                #segment.velocity_end                                     = Vstall 
                #segment.friction_coefficient                             = 0.3
                #segment.throttle                                         = 0.1
                #segment.time                                             = 5.0 
                #segment.battery_age_in_days                              = day   
                #segment.ambient_temperature                              = daily_temp  
                #segment.battery_discharge                                = True 
                #mission.append_segment(segment)
                
                ## ------------------------------------------------------------------
                ##   Reverse Thrust Flight 1   
                ## ------------------------------------------------------------------             
                #segment = Segments.Ground.Landing(base_segment)
                #segment_name = 'Reverse Thrust Day ' + str (day)+ ' Flight ' + str(nf+1)
                #segment.tag = segment_name          
                #segment.analyses.extend( analyses.base ) 
                #segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_linmco
                #segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_linmco 
                #segment.state.unknowns.throttle                          = 1.5 * ones_row(1)
                #segment.state.unknowns.propeller_power_coefficient       = 0.1* ones_row(1)    
                #segment.state.unknowns.battery_state_of_charge           = 0.3 * ones_row(1) 
                #segment.state.unknowns.battery_current                   = 200 * ones_row(1)  
                #segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1) 
                #segment.velocity_start                                   = Vstall
                #segment.velocity_end                                     = Vstall*0.5
                #segment.state.reverse_thrust                             = True             
                #segment.friction_coefficient                             = 0.3
                #segment.throttle                                         = 0.8
                #segment.time                                             = 40.0 
                #segment.battery_age_in_days                              = day   
                #segment.ambient_temperature                              = daily_temp  
                #segment.battery_discharge                                = True 
                #mission.append_segment(segment)   
            
                # ------------------------------------------------------------------
                #  Battery Charge 1
                # ------------------------------------------------------------------ 
                segment      = Segments.Ground.Battery_Charge_Discharge(base_segment) 
                segment_name = 'Charge Day ' + str (day)+ ' Flight ' + str(nf+1)          
                segment.tag  = segment_name 
                segment.analyses.extend(analyses.base)     
                segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_linmco_charge
                segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_linmco_charge  
                segment.state.unknowns.battery_state_of_charge     = 0.5 * ones_row(1) 
                segment.state.unknowns.battery_current             = 200 * ones_row(1)  
                segment.state.unknowns.battery_cell_temperature    = (daily_temp+ 1) * ones_row(1)  
                segment.state.residuals.network                    = 0. * ones_row(3)    
                segment.battery_discharge                          = False
                segment.battery_age_in_days                        = day 
                segment.ambient_temperature                        = daily_temp          
                mission.append_segment(segment)        
                       
                                
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

def save_results(results,filename):

    # Store data (serialize)
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
        
    return   

def set_axes(axes):
    """This sets the axis parameters for all plots

    Assumptions:
    None

    Source:
    None

    Inputs
    axes
        
    Outputs: 
    axes

    Properties Used:
    N/A	
    """   
    
    axes.minorticks_on()
    axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
    #axes.grid(True)   

    return  

if __name__ == '__main__': 
    main()     
    plt.show()
