# test_VTOL.py
# 
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke 

""" setup file for electric aircraft regression """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np
import pylab as plt 
import copy, time
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Core import Data, Container
from SUAVE.Methods.Power.Battery.Sizing                 import initialize_from_module_packaging  
import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from X57_Maxwell    import vehicle_setup, configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():                
    '''This script allows you to choose compare different discharge methods of SUAVE 
    2 -> Thevevin Model with RC components LiNCA 
    3 -> LiNiMnCo '''
    battery_chemistries =  ['Li_Generic','Li_NMC','Li_NCA'] 
    line_cols           = ['g^','bo-','rs-']
    
    RPM_true   = [880.2614944605407,830.5794536196578,839.8263027718258]   
    lift_coefficient_true   = [0.38376159297585954,0.3700909711946843,0.3700909711875884]

    for i in range(len(battery_chemistries)):         

        configs, analyses = full_setup(battery_chemistries[i])
        
        simple_sizing(configs)
        
        configs.finalize()
        analyses.finalize()  
        
        # mission analysis
        mission = analyses.missions.base
        results = mission.evaluate() 
        
        # save results 
        # This should be always left uncommented to test the SUAVE's Input/Output archive functions
        save_results(results,battery_chemistries[i])
        
        # plot the results
        plot_results(results,line_cols[i]) 
          
        # load and plot old results  
        old_results = load_results(battery_chemistries[i])
        plot_results(old_results,line_cols[i])  
        
        # RPM of rotor check during hover
        RPM        = results.segments.climb.conditions.propulsion.rpm[3][0]
        print('\nRPM') 
        print(RPM) 
        diff_RPM   = np.abs(RPM - RPM_true[i])
        print('RPM difference')
        print(diff_RPM)
        assert np.abs((RPM - RPM_true[i])/RPM_true[i]) < 1e-3  
        
        # lift Coefficient Check During Cruise
        lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0]
        print('\nLift Coefficient') 
        print(lift_coefficient)
        diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true[i]) 
        print('CL difference')
        print(diff_CL)
        assert np.abs((lift_coefficient  - lift_coefficient_true[i])/lift_coefficient_true[i]) < 1e-3   
        
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ---------------------------------------------------------------------- 

def full_setup(battery_chemistry):

    # vehicle data
    vehicle  = vehicle_setup()
    
    # modify vehicle chemstry
    vehicle  = modify_baseline_vehicle(vehicle,battery_chemistry)
    
    # Set up configs
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle,battery_chemistry)     
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.AERODAS() 
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

# ----------------------------------------------------------------------
#   Mofigy Maxwell Battery Network
# ----------------------------------------------------------------------
def modify_baseline_vehicle(vehicle,battery_chemistry):
    if battery_chemistry == 'Li_Generic':   
        pass
    elif battery_chemistry == 'Li_NCA':  
        net = vehicle.propulsors.battery_propeller
        # remove battery from network 
        
        # append new battery
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNCA_18650()  
        bat.cell.max_mission_discharge     = 9.    # Amps  
        bat.cell.max_discharge_rate        = 15.   # Amps   
        bat.cell.surface_area              = (np.pi*bat.cell.height*bat.cell.diameter)  
        bat.pack_config.series             = 128   
        bat.pack_config.parallel           = 40     
        bat.module_config.normal_count     = 16     
        bat.module_config.parallel_count   = 20      
        bat.age_in_days                    = 1 
        initialize_from_module_packaging(bat)       
        net.battery                        = bat
        net.voltage                        = bat.max_voltage    
    
    elif battery_chemistry == 'Li_NMC':  
        net = vehicle.propulsors.battery_propeller
        # remove battery from network 
        
        # append new battery        
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()      
        bat.cell.max_mission_discharge     = 9.    # Amps  
        bat.cell.max_discharge_rate        = 15.   # Amps   
        bat.cell.surface_area              = (np.pi*bat.cell.height*bat.cell.diameter)  
        bat.pack_config.series             = 128   
        bat.pack_config.parallel           = 40     
        bat.module_config.normal_count     = 16     
        bat.module_config.parallel_count   = 20      
        bat.age_in_days                    = 1 
        initialize_from_module_packaging(bat)       
        net.battery                        = bat
        net.voltage                        = bat.max_voltage      
    
    return  vehicle 

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses,vehicle,battery_chemistry): 
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
    daily_temp = 20  

    if battery_chemistry == 'Li_Generic':    
        # base segment
        base_segment = Segments.Segment()
        ones_row     = base_segment.state.ones_row
        base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
        base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
        base_segment.state.numerics.number_control_points        = 4
        base_segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns
        base_segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals
        base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.battery_propeller.battery.max_voltage * ones_row(1)  
        base_segment.state.residuals.network                     = 0. * ones_row(2)
        
        bat                                                      = vehicle.propulsors.battery_propeller.battery  
        base_segment.battery_configuration                       = bat.pack_config
        base_segment.initial_mission_battery_energy              = bat.initial_max_energy            
        base_segment.max_energy                                  = bat.max_energy
        base_segment.charging_SOC_cutoff                         = bat.charging_SOC_cutoff  
        base_segment.charging_voltage                            = bat.charging_voltage   
        base_segment.charging_current                            = bat.charging_current     
        
        # ------------------------------------------------------------------
        #   Climb 1 : constant Speed, constant rate segment 
        # ------------------------------------------------------------------ 
        segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
        segment.tag = "climb"
        segment.analyses.extend( analyses.base )
        segment.battery_energy                              = vehicle.propulsors.battery_propeller.battery.max_energy * 0.89
        segment.altitude_start                              = 2500.0  * Units.feet
        segment.altitude_end                                = 8012    * Units.feet 
        segment.air_speed                                   = 96.4260 * Units['mph'] 
        segment.climb_rate                                  = 700.034 * Units['ft/min']  
        segment.state.unknowns.throttle                     = 0.85 * ones_row(1)   
        segment.state.unknowns.propeller_power_coefficient  = 0.005 * ones_row(1) 
        
        segment.battery_temperature                         = daily_temp    	                 
        segment.battery_age_in_days                         = 1 	       
        segment.battery_charge_throughput                   = 0	       
        segment.ambient_temperature                         = daily_temp	
        segment.battery_cell_temperature                    = daily_temp
        segment.battery_discharge                           = True	       
        segment.battery_resistance_growth_factor            = 1	       
        segment.battery_capacity_fade_factor                = 1  
        segment.battery_cumulative_charge_throughput        = 0      
        segment.battery_pack_temperature                    = daily_temp   
        segment.battery_thevenin_voltage                    = 0 
        
        mission.append_segment(segment)               
        
        # ------------------------------------------------------------------
        #   Cruise Segment: constant Speed, constant altitude
        # ------------------------------------------------------------------ 
        segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
        segment.tag = "cruise" 
        segment.analyses.extend(analyses.base) 
        segment.altitude                                   = 8012   * Units.feet
        segment.air_speed                                  = 140.91 * Units['mph'] 
        segment.distance                                   =  20.   * Units.nautical_mile   
        segment.state.unknowns.propeller_power_coefficient = 0.005 * ones_row(1)         
        segment.state.unknowns.throttle                    = 0.9 *  ones_row(1) 
        
        segment.battery_age_in_days                        = 1 	                        
        segment.ambient_temperature                        = daily_temp                        
        segment.battery_discharge                          = True 
        
        mission.append_segment(segment)    
        
        # ------------------------------------------------------------------
        #   Descent Segment: constant Speed, constant rate segment 
        # ------------------------------------------------------------------ 
        segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
        segment.tag = "decent" 
        segment.analyses.extend( analyses.base ) 
        segment.altitude_start                             = 8012  * Units.feet
        segment.altitude_end                               = 2500  * Units.feet
        segment.air_speed                                  = 140.91 * Units['mph']  
        segment.climb_rate                                 = - 500.401  * Units['ft/min']  
        segment.state.unknowns.throttle                    = 0.9 * ones_row(1)   
        segment.state.unknowns.propeller_power_coefficient = 0.005 * ones_row(1)    
        
        segment.battery_age_in_days                        = 1 	                        
        segment.ambient_temperature                        = daily_temp                        
        segment.battery_discharge                          = True 	
        
        mission.append_segment(segment) 
    
        # ------------------------------------------------------------------
        # Charge Segment
        # ------------------------------------------------------------------ 
        segment      = Segments.Ground.Battery_Charge_Discharge(base_segment) 
        segment_name = 'charge'      
        segment.tag  = segment_name 
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network           = vehicle.propulsors.battery_propeller.unpack_unknowns_charge
        segment.process.iterate.residuals.network          = vehicle.propulsors.battery_propeller.residuals_charge    
        segment.state.residuals.network                    = 0. * ones_row(1)     
        segment.battery_discharge                          = False
        segment.battery_age_in_days                        = 1
        segment.ambient_temperature                        = daily_temp          
        mission.append_segment(segment)        
        
        
    # Component 8 the Battery
    elif battery_chemistry == 'Li_NCA':  
        # base segment 
        base_segment = Segments.Segment()
        ones_row     = base_segment.state.ones_row
        base_segment.state.numerics.number_control_points             = 10    
        base_segment.process.iterate.initials.initialize_battery      = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery    
        base_segment.process.finalize.post_process.update_battery_age = SUAVE.Methods.Missions.Segments.Common.Energy.update_battery_age         
        base_segment.process.iterate.conditions.planet_position       = SUAVE.Methods.skip       
        base_segment.state.residuals.network                          = 0.    * ones_row(4) 
        bat                                                           = vehicle.propulsors.battery_propeller.battery  
        base_segment.initial_mission_battery_energy                   = bat.initial_max_energy    
        base_segment.battery_configuration                            = bat.pack_config
        base_segment.max_energy                                       = bat.max_energy
        base_segment.charging_SOC_cutoff                              = bat.cell.charging_SOC_cutoff  
        base_segment.charging_voltage                                 = bat.cell.charging_voltage  * bat.pack_config.series
        base_segment.charging_current                                 = bat.cell.charging_current  * bat.pack_config.parallel
        
        # ------------------------------------------------------------------	 
        #   Climb 1 : constant Speed, constant rate segment 	                 
        # ------------------------------------------------------------------ 	 
        segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)	         
        segment.tag = "climb"	        
        segment.analyses.extend( analyses.base )  
        segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns_linca
        segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals_linca 
      
        segment.state.unknowns.throttle                     = 0.85 * ones_row(1)
        segment.state.unknowns.propeller_power_coefficient  = 0.3  * ones_row(1)  
        segment.state.unknowns.battery_state_of_charge      = 0.8 * ones_row(1) 
        segment.state.unknowns.battery_thevenin_voltage     = 10 * ones_row(1)   
        segment.state.unknowns.battery_cell_temperature     = daily_temp + 10  * ones_row(1)   
        
        segment.altitude_start                              = 2500.0  * Units.feet	 
        segment.altitude_end                                = 8012    * Units.feet 	 
        segment.air_speed                                   = 40.640698 # 96.4260 * Units['mph'] 	 
        segment.climb_rate                                  = 700.034 * Units['ft/min'] 
                                                            
        segment.battery_energy                              = bat.max_energy * 0.89      
        segment.battery_temperature                         = daily_temp    	                 
        segment.battery_age_in_days                         = 1 	       
        segment.battery_charge_throughput                   = 0	       
        segment.ambient_temperature                         = daily_temp	
        segment.battery_cell_temperature                    = daily_temp
        segment.battery_discharge                           = True	       
        segment.battery_resistance_growth_factor            = 1	       
        segment.battery_capacity_fade_factor                = 1  
        segment.battery_cumulative_charge_throughput        = 0      
        segment.battery_pack_temperature                    = daily_temp  
        segment.battery_thevenin_voltage                    = 0   
        
        
        # add to misison	     
        mission.append_segment(segment)	       
        
        # ------------------------------------------------------------------	 
        #   Cruise Segment: constant Speed, constant altitude	           
        # ------------------------------------------------------------------ 	 
        segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)	 
        segment.tag = "cruise" 	         
        segment.analyses.extend(analyses.base) 	  
        segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns_linca
        segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals_linca 
      
        segment.state.unknowns.throttle                     = 0.85 * ones_row(1)
        segment.state.unknowns.propeller_power_coefficient  = 0.3  * ones_row(1)  
        segment.state.unknowns.battery_state_of_charge      = 0.5  *  ones_row(1) 
        segment.state.unknowns.battery_thevenin_voltage     = 10   * ones_row(1)    
        segment.state.unknowns.battery_cell_temperature     = daily_temp + 10  * ones_row(1)   
                
        segment.altitude                                    = 8012   * Units.feet	        
        segment.air_speed                                   = 64.14712542 # 140.91 * Units['mph'] 	   
        segment.distance                                    = 20.75   * Units.nautical_mile 	  
        segment.battery_age_in_days                         = 1 	                        
        segment.ambient_temperature                         = daily_temp                        
        segment.battery_discharge                           = True 	                                     
        
        # add to misison	
        mission.append_segment(segment)    	
        
        # ------------------------------------------------------------------	
        #   Descent Segment: constant Speed, constant rate segment 	
        # ------------------------------------------------------------------ 	
        segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)	
        segment.tag = "descent" 	
        segment.analyses.extend(analyses.base) 	  
        segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns_linca
        segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals_linca 
        
        segment.state.unknowns.throttle                     = 0.8 *  ones_row(1)   
        segment.state.unknowns.battery_state_of_charge      = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_thevenin_voltage     = 10   * ones_row(1) 
        segment.state.unknowns.battery_cell_temperature     = daily_temp  + 10 * ones_row(1)   
        segment.state.unknowns.propeller_power_coefficient  = 0.3  * ones_row(1)  
        
        segment.altitude_start                              = 8012  * Units.feet	
        segment.altitude_end                                = 2500  * Units.feet	
        segment.air_speed                                   = 61.8495481 	
        segment.descent_rate                                = 500.401  * Units['ft/min']
        
        segment.battery_age_in_days                         = 1 	
        segment.ambient_temperature                         = daily_temp	
        segment.battery_discharge                           = True 	 
        
        # add to misison	
        mission.append_segment(segment) 
        
        # ------------------------------------------------------------------
        # Charge Segment
        # ------------------------------------------------------------------ 
        segment      = Segments.Ground.Battery_Charge_Discharge(base_segment) 
        segment_name = 'charge'      
        segment.tag  = segment_name 
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network           =  vehicle.propulsors.battery_propeller.unpack_unknowns_linca_charge
        segment.process.iterate.residuals.network          =  vehicle.propulsors.battery_propeller.residuals_linca_charge 
        segment.state.unknowns.battery_state_of_charge     = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_thevenin_voltage    = 10   * ones_row(1) 
        segment.state.unknowns.battery_cell_temperature    = daily_temp  + 10 * ones_row(1)   
        segment.state.residuals.network                    = 0. * ones_row(3)    
        segment.battery_discharge                          = False
        segment.battery_age_in_days                        = 1
        segment.ambient_temperature                        = daily_temp          
        mission.append_segment(segment)        
    
    
    elif battery_chemistry == 'Li_NMC':       
        # base segment 
        base_segment = Segments.Segment()
        ones_row     = base_segment.state.ones_row
        base_segment.state.numerics.number_control_points             = 10  
        base_segment.process.iterate.initials.initialize_battery      = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery    
        base_segment.process.finalize.post_process.update_battery_age = SUAVE.Methods.Missions.Segments.Common.Energy.update_battery_age         
        base_segment.process.iterate.conditions.planet_position       = SUAVE.Methods.skip     
        base_segment.state.residuals.network                          = 0.    * ones_row(4) 
        bat                                                           = vehicle.propulsors.battery_propeller.battery  
        base_segment.initial_mission_battery_energy                   = bat.initial_max_energy    
        base_segment.battery_configuration                            = bat.pack_config
        base_segment.max_energy                                       = bat.max_energy
        base_segment.charging_SOC_cutoff                              = bat.cell.charging_SOC_cutoff  
        base_segment.charging_voltage                                 = bat.cell.charging_voltage  * bat.pack_config.series
        base_segment.charging_current                                 = bat.cell.charging_current  * bat.pack_config.parallel
        
        # ------------------------------------------------------------------
        #   Climb Segment Flight 1 
        # ------------------------------------------------------------------ 
        segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
        segment_name = 'climb'
        segment.tag = segment_name          
        segment.analyses.extend( analyses.base ) 
        segment.process.iterate.unknowns.network                 = vehicle.propulsors.battery_propeller.unpack_unknowns_linmco
        segment.process.iterate.residuals.network                = vehicle.propulsors.battery_propeller.residuals_linmco 
      
        segment.state.unknowns.throttle                          = 0.85 * ones_row(1) 
        segment.state.unknowns.battery_state_of_charge           = 0.8 * ones_row(1) 
        segment.state.unknowns.battery_current                   = 200 * ones_row(1)  
        segment.state.unknowns.battery_cell_temperature          = (daily_temp) * ones_row(1) 
        segment.state.unknowns.propeller_power_coefficient       = 0.3 * ones_row(1) 
        
        segment.altitude_start                                   = 2500.0  * Units.feet	
        segment.altitude_end                                     = 8012    * Units.feet 
        segment.air_speed                                        = 40.640698 # 96.4260 * Units['mph'] 	 
        segment.climb_rate                                       = 670.6525407 * Units['ft/min'] # 700.034 * Units['ft/min'] 
        
        segment.ambient_temperature                              = daily_temp                  
        segment.battery_age_in_days                              = 1  
        segment.battery_discharge                                = True  
        segment.battery_resistance_growth_factor                 = 1 
        segment.battery_capacity_fade_factor                     = 1    
        segment.battery_energy                                   = bat.max_energy * 0.89
        segment.battery_cumulative_charge_throughput             = 0     
        segment.battery_thevenin_voltage                         = 0  
        segment.battery_discharge                                = True 
        segment.battery_cell_temperature                         = daily_temp 
        segment.battery_pack_temperature                         = daily_temp  
        
        mission.append_segment(segment)	       
        
        # ------------------------------------------------------------------	 
        #   Cruise Segment: constant Speed, constant altitude	               
        # ------------------------------------------------------------------ 	 
        segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)	 
        segment.tag = "cruise" 	         
        segment.analyses.extend(analyses.base) 	         
        segment.process.iterate.unknowns.network                 = vehicle.propulsors.battery_propeller.unpack_unknowns_linmco
        segment.process.iterate.residuals.network                = vehicle.propulsors.battery_propeller.residuals_linmco 
        
        segment.state.unknowns.throttle                          = 0.8 * ones_row(1)     
        segment.state.unknowns.battery_state_of_charge           = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_current                   = 200  * ones_row(1)  
        segment.state.unknowns.battery_cell_temperature          = (daily_temp ) * ones_row(1)               
        segment.state.unknowns.propeller_power_coefficient       = 0.05 * ones_row(1) 
        
        segment.altitude                                         = 8012   * Units.feet	        
        segment.air_speed                                        = 64.14712542 # 140.91 * Units['mph'] 	       
        segment.distance                                         = 20.75    * Units.nautical_mile  
        segment.battery_discharge                                = True   
        segment.ambient_temperature                              = daily_temp                  
        segment.battery_age_in_days                              = 1         
        mission.append_segment(segment)    
        
        # add to misison	
        mission.append_segment(segment)    	
        
        # ------------------------------------------------------------------	
        #   Descent Segment: constant Speed, constant rate segment 	
        # ------------------------------------------------------------------ 	
        segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)	
        segment.tag = "descent" 	
        segment.analyses.extend( analyses.base ) 	
        segment.process.iterate.unknowns.network                 = vehicle.propulsors.battery_propeller.unpack_unknowns_linmco
        segment.process.iterate.residuals.network                = vehicle.propulsors.battery_propeller.residuals_linmco  
        
        segment.state.unknowns.throttle                          = 0.6  * ones_row(1)   
        segment.state.unknowns.battery_state_of_charge           = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_current                   = 200  * ones_row(1)  
        segment.state.unknowns.battery_cell_temperature          = (daily_temp+ 1) * ones_row(1)  
        segment.state.unknowns.propeller_power_coefficient       = 0.05 * ones_row(1) 
        
        segment.altitude_start                                   = 8012  * Units.feet	
        segment.altitude_end                                     = 2500  * Units.feet	
        segment.air_speed                                        = 61.8495481 	
        segment.descent_rate                                     = 500.401  * Units['ft/min']
            
        segment.battery_discharge                                = True  
        segment.battery_age_in_days                              = 1   
        segment.ambient_temperature                              = daily_temp                  
        mission.append_segment(segment)    
        
        # ------------------------------------------------------------------
        # Charge Segment
        # ------------------------------------------------------------------ 
        segment      = Segments.Ground.Battery_Charge_Discharge(base_segment) 
        segment_name = 'charge'      
        segment.tag  = segment_name 
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network           = vehicle.propulsors.battery_propeller.unpack_unknowns_linmco_charge
        segment.process.iterate.residuals.network          = vehicle.propulsors.battery_propeller.residuals_linmco_charge  
        segment.state.unknowns.battery_state_of_charge     = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_current             = 200 * ones_row(1)  
        segment.state.unknowns.battery_cell_temperature    = (daily_temp+ 1) * ones_row(1)  
        segment.state.residuals.network                    = 0. * ones_row(3)     
        segment.battery_discharge                          = False
        segment.battery_age_in_days                        = 1
        segment.ambient_temperature                        = daily_temp          
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


def plot_results(results,line_style ):  
    
    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style) 
    
    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)  
    
    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)
    
    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results, line_style)
    
    # Plot Propeller Conditions 
    plot_propeller_conditions(results, line_style) 
    
    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results, line_style)
    
    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results, line_style)   
     
     
    return


def load_results(battery_chemistry):
    file_name = 'electric_aircraft' + battery_chemistry + '.res'
    return SUAVE.Input_Output.SUAVE.load(file_name)

def save_results(results,battery_chemistry):
    file_name = 'electric_aircraft' + battery_chemistry + '.res'
    SUAVE.Input_Output.SUAVE.archive(results,file_name)
    
    return  
if __name__ == '__main__': 
    main()    
