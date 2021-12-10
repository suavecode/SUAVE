# aircraft_discharge_comparisons.py
# 
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke  

""" setup file for comparing battery packs of three chemistries in all-electric aircraft """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Core import Data
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty 
from SUAVE.Methods.Power.Battery.Sizing         import initialize_from_mass
import sys

sys.path.append('../Vehicles')
from X57_Maxwell_Mod2    import vehicle_setup as   GA_vehicle_setup
from X57_Maxwell_Mod2    import configs_setup as   GA_configs_setup
from Stopped_Rotor       import vehicle_setup as   EVTOL_vehicle_setup 
from Stopped_Rotor       import configs_setup as   EVTOL_configs_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():     
    
    battery_chemistry  =  ['NMC','LFP']
    line_style_new     =  ['bo-','ro-','ko-']
    line_style2_new    =  ['bs-','rs-','ks-'] 
    
    # ----------------------------------------------------------------------
    #  True Values  
    # ----------------------------------------------------------------------    

    # General Aviation Aircraft   
    GA_RPM_true              = [973.9534348587468,973.953434872128]
    GA_lift_coefficient_true = [0.5474716962005756,0.5474716962005768]
    

    # EVTOL Aircraft      
    EVTOL_RPM_true              = [2385.0695323649907,2385.06953270966]
    EVTOL_lift_coefficient_true = [0.8075122900985718,0.8075122900986484]
    
        
    for i in range(len(battery_chemistry)):
        print('***********************************')
        print(battery_chemistry[i] + ' Cell Powered Aircraft')
        print('***********************************')
        
        print('\nBattery Propeller Network Analysis')
        print('--------------------------------------')
        
        GA_configs, GA_analyses = GA_full_setup(battery_chemistry[i])  
        GA_configs.finalize()
        GA_analyses.finalize()   
         
        # mission analysis
        GA_mission = GA_analyses.missions.base
        GA_results = GA_mission.evaluate() 
         
        # plot the results
        plot_results(GA_results,line_style_new[i],line_style2_new[i])  
        
        # RPM of rotor check during hover
        GA_RPM        = GA_results.segments.climb_1.conditions.propulsion.propeller_rpm[3][0] 
        print('GA RPM: ' + str(GA_RPM))
        GA_diff_RPM   = np.abs(GA_RPM - GA_RPM_true[i])
        print('RPM difference')
        print(GA_diff_RPM)
        assert np.abs((GA_RPM - GA_RPM_true[i])/GA_RPM_true[i]) < 1e-6  
        
        # lift Coefficient Check During Cruise
        GA_lift_coefficient        = GA_results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0] 
        print('GA CL: ' + str(GA_lift_coefficient)) 
        GA_diff_CL                 = np.abs(GA_lift_coefficient  - GA_lift_coefficient_true[i]) 
        print('CL difference')
        print(GA_diff_CL)
        assert np.abs((GA_lift_coefficient  - GA_lift_coefficient_true[i])/GA_lift_coefficient_true[i]) < 1e-6
            
            
      
        print('\nLift-Cruise Network Analysis')  
        print('--------------------------------------')
        
        EVTOL_configs, EVTOL_analyses = EVTOL_full_setup(battery_chemistry[i])   
        EVTOL_configs.finalize()
        EVTOL_analyses.finalize()   
         
        # mission analysis
        EVTOL_mission = EVTOL_analyses.missions.base
        EVTOL_results = EVTOL_mission.evaluate()  
        
        # plot the results
        plot_results(EVTOL_results,line_style_new[i],line_style2_new[i])  
        
        # RPM of rotor check during hover
        EVTOL_RPM        = EVTOL_results.segments.climb_1.conditions.propulsion.lift_rotor_rpm[2][0] 
        print('EVTOL RPM: ' + str(EVTOL_RPM)) 
        EVTOL_diff_RPM   = np.abs(EVTOL_RPM - EVTOL_RPM_true[i])
        print('EVTOL_RPM difference')
        print(EVTOL_diff_RPM)
        assert np.abs((EVTOL_RPM - EVTOL_RPM_true[i])/EVTOL_RPM_true[i]) < 1e-6  
        
        # lift Coefficient Check During Cruise
        EVTOL_lift_coefficient        = EVTOL_results.segments.departure_terminal_procedures.conditions.aerodynamics.lift_coefficient[2][0] 
        print('EVTOL CL: ' + str(EVTOL_lift_coefficient)) 
        EVTOL_diff_CL                 = np.abs(EVTOL_lift_coefficient  - EVTOL_lift_coefficient_true[i]) 
        print('CL difference')
        print(EVTOL_diff_CL)
        assert np.abs((EVTOL_lift_coefficient  - EVTOL_lift_coefficient_true[i])/EVTOL_lift_coefficient_true[i]) < 1e-6   
                
            
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ---------------------------------------------------------------------- 

def GA_full_setup(battery_chemistry):

    # vehicle data
    vehicle  = GA_vehicle_setup()
    
    # Modify  Battery  
    net = vehicle.networks.battery_propeller
    bat = net.battery 
    if battery_chemistry == 'NMC': 
        bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()  
    elif battery_chemistry == 'LFP': 
        bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiFePO4_18650()  
    
    bat.mass_properties.mass = 500. * Units.kg  
    bat.max_voltage          = 500.             
    initialize_from_mass(bat)
    
    # Assume a battery pack module shape. This step is optional but
    # required for thermal analysis of the pack
    number_of_modules                = 10
    bat.module_config.total          = int(np.ceil(bat.pack_config.total/number_of_modules))
    bat.module_config.normal_count   = int(np.ceil(bat.module_config.total/bat.pack_config.series))
    bat.module_config.parallel_count = int(np.ceil(bat.module_config.total/bat.pack_config.parallel))
    net.battery                      = bat      
    
    net.battery              = bat
    net.voltage              = bat.max_voltage     
    
    # Set up configs
    configs  = GA_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = GA_mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

 
def EVTOL_full_setup(battery_chemistry):

    # vehicle data
    vehicle  = EVTOL_vehicle_setup() 


    # Modify  Battery  
    net = vehicle.networks.lift_cruise
    bat = net.battery 
    if battery_chemistry == 'NMC': 
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()
    elif battery_chemistry == 'LFP': 
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiFePO4_18650()
    
    bat.mass_properties.mass = 500. * Units.kg  
    bat.max_voltage          = 500.             
    initialize_from_mass(bat)
    
    # Assume a battery pack module shape. This step is optional but required for thermal analysis of the pack. We will assume that all cells electrically connected 
    # in series wihtin the module are arranged in one row normal direction to the airflow. Likewise ,
    # all cells electrically in paralllel are arranged in the direction to the cooling fluid  
    number_of_modules                = 10
    bat.module_config.total          = int(np.ceil(bat.pack_config.total/number_of_modules))
    bat.module_config.normal_count   = int(np.ceil(bat.module_config.total/bat.pack_config.series))
    bat.module_config.parallel_count = int(np.ceil(bat.module_config.total/bat.pack_config.parallel))
    
    net.battery              = bat
    net.voltage              = bat.max_voltage     
     
    configs  = EVTOL_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = EVTOL_mission_setup(configs_analyses,vehicle)
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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
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
    energy.network = vehicle.networks 
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
#   Define the Mission
# ----------------------------------------------------------------------

def GA_mission_setup(analyses,vehicle): 
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

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 4 
    base_segment.battery_age_in_days                         = 1 # optional but added for regression
    base_segment.temperature_deviation                       = 1 # Kelvin #  optional but added for regression
    
    # ------------------------------------------------------------------
    #   Climb 1 : constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.battery_energy                   = vehicle.networks.battery_propeller.battery.max_energy * 0.89
    segment.altitude_start                   = 2500.0  * Units.feet
    segment.altitude_end                     = 8012    * Units.feet 
    segment.air_speed                        = 96.4260 * Units['mph'] 
    segment.climb_rate                       = 700.034 * Units['ft/min']    
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment) 

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 8012   * Units.feet
    segment.air_speed                 = 120.91 * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile   
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)   

    # add to misison
    mission.append_segment(segment)    


    # ------------------------------------------------------------------
    #   Descent Segment Flight 1   
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment) 
    segment.tag = "decent"  
    segment.analyses.extend( analyses.base )       
    segment.altitude_start                                   = 8012 * Units.feet  
    segment.altitude_end                                     = 2500.0 * Units.feet
    segment.air_speed_start                                  = 175.* Units['mph']  
    segment.air_speed_end                                    = 110 * Units['mph']   
    segment.climb_rate                                       = -200 * Units['ft/min']  
    segment.state.unknowns.throttle                          = 0.8 * ones_row(1)  
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment,  initial_power_coefficient = 0.1)   
    
    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #  Charge Segment: 
    # ------------------------------------------------------------------     
    # Charge Model 
    segment                                                 = Segments.Ground.Battery_Charge_Discharge(base_segment)     
    segment.tag                                             = 'Charge'
    segment.analyses.extend(analyses.base)           
    segment.battery_discharge                               = False      
    segment.increment_battery_cycle_day                     = True            
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)    
    
    # add to misison
    mission.append_segment(segment)        

    return mission


def EVTOL_mission_setup(analyses,vehicle): 
        
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
    base_segment.state.numerics.number_control_points        = 3 
    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip

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
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 40.  * Units.ft
    segment.climb_rate                                       = 500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.networks.lift_cruise.battery.max_energy
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip   
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment,\
                                                                                    initial_lift_rotor_power_coefficient = 0.01,
                                                                                    initial_throttle_lift = 0.9)
    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------
    segment                                            = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag                                        = "transition_1"
    segment.analyses.extend( analyses.base ) 

    segment.altitude                                 = 40.  * Units.ft
    segment.air_speed_start                          = 500. * Units['ft/min']
    segment.air_speed_end                            = 0.8 * Vstall
    segment.acceleration                             = 9.8/5
    segment.pitch_initial                            = 0.0 * Units.degrees
    segment.pitch_final                              = 5. * Units.degrees
    ones_row                                         = segment.state.ones_row
    segment.state.unknowns.throttle                  = 0.95  *  ones_row(1)
    segment.process.iterate.unknowns.mission         = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability     = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability  = SUAVE.Methods.skip
    segment = vehicle.networks.lift_cruise.add_transition_unknowns_and_residuals_to_segment(segment,
                                                         initial_prop_power_coefficient = 0.2,
                                                         initial_lift_rotor_power_coefficient = 0.01,
                                                         initial_throttle_lift = 0.9,)

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------
    segment                                             = Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb(base_segment)
    segment.tag                                         = "transition_2"
    segment.analyses.extend( analyses.base )
    segment.altitude_start                          = 40.0 * Units.ft
    segment.altitude_end                            = 50.0 * Units.ft
    segment.air_speed                               = 0.8 * Vstall
    segment.climb_angle                             = 1 * Units.degrees
    segment.acceleration                            = 0.5 * Units['m/s/s']
    segment.pitch_initial                           = 5. * Units.degrees
    segment.pitch_final                             = 7. * Units.degrees
    segment.state.unknowns.throttle                 = 0.95  * ones_row(1)
    segment.process.iterate.unknowns.mission        = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip
    segment = vehicle.networks.lift_cruise.add_transition_unknowns_and_residuals_to_segment(segment,
                                                         initial_prop_power_coefficient = 0.2,
                                                         initial_lift_rotor_power_coefficient = 0.01,
                                                         initial_throttle_lift = 0.9,)

    # add to misison
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment                                            = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "climb_2"
    segment.analyses.extend( analyses.base )
    segment.air_speed                                  = 1.1*Vstall
    segment.altitude_start                             = 50.0 * Units.ft
    segment.altitude_end                               = 300. * Units.ft
    segment.climb_rate                                 = 500. * Units['ft/min'] 
    segment.state.unknowns.throttle =  0.80 * ones_row(1)
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------
    segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag                                        = "departure_terminal_procedures"
    segment.analyses.extend( analyses.base )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = 60.   * Units.second
    segment.air_speed                                  = 1.2*Vstall
    segment.state.unknowns.throttle =  0.80 * ones_row(1)
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment,\
                                                                                          initial_prop_power_coefficient = 0.16)
    # add to misison
    mission.append_segment(segment) 

    segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag                                        = "cruise" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                                   = 1000.0 * Units.ft
    segment.air_speed                                  = 110. * Units['mph']
    segment.distance                                   = 40. * Units.miles 
    segment.state.unknowns.throttle                    = 0.8 * ones_row(1) 
    segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment ) 

    # add to misison
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------
    #  Charge Segment: 
    # ------------------------------------------------------------------  
    # Charge Model 
    segment                                                  = Segments.Ground.Battery_Charge_Discharge(base_segment)     
    segment.tag                                              = 'Charge'
    segment.analyses.extend(analyses.base)           
    segment.battery_discharge                                = False    
    segment.increment_battery_cycle_day                      = True         
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment)    
    
    # add to misison
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


def plot_results(results,line_style,line_style2):  
    
    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style) 
    
    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)  
    
    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)
    
    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results, line_style,line_style2) 
    plot_battery_cell_conditions(results, line_style,line_style2)
    plot_battery_degradation(results)
    
    # Plot Propeller Conditions 
    plot_propeller_conditions(results, line_style) 
    
    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results, line_style)
     
    return
 
if __name__ == '__main__': 
    main()    
    plt.show()
