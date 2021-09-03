# test_VTOL.py
# 
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke 
#          Jul 2021, R. Erhard

""" setup file for electric aircraft regression """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Core import Data
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty 
from SUAVE.Methods.Power.Battery.Sizing                 import initialize_from_mass
import sys

sys.path.append('../Vehicles')
from X57_Maxwell_Mod2    import vehicle_setup, configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
    
    battery_chemistry  =  ['LFP','NCA','NMC']
    line_style_new     =  ['bo-','ro-','ko-']
    line_style2_new    =  ['bs-','rs-','ks-']
    line_style_saved   =   ['bo--','ro--','ko--']
    line_style2_saved  =  ['bs--','rs--','ks--']
    
    # ----------------------------------------------------------------------
    #  General Aviation 
    # ----------------------------------------------------------------------    
    RPM_true              = [887.1356296331286,887.1356296331286,887.1356296331286]
    lift_coefficient_true = [887.1356296331286,887.1356296331286,887.1356296331286]
        
    for i in range(len(battery_chemistry)):
        print(battery_chemistry[i] + ' cell powered aircraft')
        configs, analyses = GA_full_setup(battery_chemistry[i]) 
        
        configs.finalize()
        analyses.finalize()   
         
        # mission analysis
        mission = analyses.missions.base
        results = mission.evaluate() 
        
        # save results 
        # This should be always left uncommented to test the SUAVE's Input/Output archive functions
        save_results(results,battery_chemistry[i])
        
        # plot the results
        plot_results(results,line_style_new[i],line_style2_new[i]) 
          
        # load and plot old results  
        old_results = load_results(battery_chemistry[i])
        plot_results(old_results,line_style_saved[i],line_style2_saved[i])  
        
        ## RPM of rotor check during hover
        #RPM        = results.segments.climb_1.conditions.propulsion.propeller_rpm[3][0]  
        #diff_RPM   = np.abs(RPM - RPM_true[i])
        #print('RPM difference')
        #print(diff_RPM)
        #assert np.abs((RPM - RPM_true[i])/RPM_true[i]) < 1e-3  
        
        ## lift Coefficient Check During Cruise
        #lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0] 
        #diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true[i]) 
        #print('CL difference')
        #print(diff_CL)
        #assert np.abs((lift_coefficient  - lift_coefficient_true[i])/lift_coefficient_true[i]) < 1e-3   
            
            
    
        # ----------------------------------------------------------------------
        #  EVTOL
        # ----------------------------------------------------------------------  
        RPM_true              = [887.1356296331286,887.1356296331286,887.1356296331286]
        lift_coefficient_true = [887.1356296331286,887.1356296331286,887.1356296331286]
            
        for i in range(len(battery_chemistry)):
            print(battery_chemistry[i] + ' cell powered aircraft')
            configs, analyses = full_setup(battery_chemistry[i]) 
            
            configs.finalize()
            analyses.finalize()   
             
            # mission analysis
            mission = analyses.missions.base
            results = mission.evaluate() 
            
            # save results 
            # This should be always left uncommented to test the SUAVE's Input/Output archive functions
            save_results(results,battery_chemistry[i])
            
            # plot the results
            plot_results(results,line_style_new[i],line_style2_new[i]) 
              
            # load and plot old results  
            old_results = load_results(battery_chemistry[i])
            plot_results(old_results,line_style_saved[i],line_style2_saved[i])  
            
            ## RPM of rotor check during hover
            #RPM        = results.segments.climb_1.conditions.propulsion.propeller_rpm[3][0]  
            #diff_RPM   = np.abs(RPM - RPM_true[i])
            #print('RPM difference')
            #print(diff_RPM)
            #assert np.abs((RPM - RPM_true[i])/RPM_true[i]) < 1e-3  
            
            ## lift Coefficient Check During Cruise
            #lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[2][0] 
            #diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true[i]) 
            #print('CL difference')
            #print(diff_CL)
            #assert np.abs((lift_coefficient  - lift_coefficient_true[i])/lift_coefficient_true[i]) < 1e-3   
                
                
            
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ---------------------------------------------------------------------- 

def full_setup(battery_chemistry):

    # vehicle data
    vehicle  = vehicle_setup()
    
    # Modify vehicle

    # Battery  
    net = vehicle.networks.battery_propeller
    bat = net.battery
    if battery_chemistry == 'NCA':
        bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNCA_18650()     
    elif battery_chemistry == 'NMC': 
        bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()  
    elif battery_chemistry == 'LFP': 
        bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiFePO4_38120()  
    
    bat.mass_properties.mass = 500. * Units.kg  
    bat.max_voltage          = 500.             
    initialize_from_mass(bat)
    
    # Here we, are going to assume a battery pack module shape. This step is optional but
    # required for thermal analysis of tge pack
    number_of_modules                = 10
    bat.module_config.total          = int(np.ceil(bat.pack_config.total/number_of_modules))
    bat.module_config.normal_count   = int(np.ceil(bat.module_config.total/bat.pack_config.parallel))
    bat.module_config.parallel_count = bat.pack_config.parallel
    
    net.battery              = bat
    net.voltage              = bat.max_voltage     
    
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
    
    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 4
    base_segment.battery_discharge                           = True 
    base_segment.battery_age_in_days                         = 1
    
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
 
    segment.battery_cell_temperature                    = 300  
    segment.battery_pack_temperature                    = 300  
    segment.ambient_temperature                         = 300   
    segment.battery_charge_throughput                   = 0 
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
    segment.air_speed                 = 140.91 * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile   
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)   

    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Descent Segment: constant Speed, constant rate segment 
    # ------------------------------------------------------------------ 
    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "decent" 
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start            = 8012  * Units.feet
    segment.altitude_end              = 2500  * Units.feet
    segment.air_speed                 = 96.4260 * Units['mph'] 
    segment.descent_rate              = 100.401  * Units['ft/min']   
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)   
    
    # add to misison
    mission.append_segment(segment)  

    # ------------------------------------------------------------------
    #  Charge Segment:  
    # ------------------------------------------------------------------  
    segment      = Segments.Ground.Battery_Charge_Discharge(base_segment)  
    segment.tag  = 'charge'  
    segment.analyses.extend(analyses.base)                 
    segment.battery_discharge        = False 
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)   
    
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


def plot_results(results,line_style,line_style2):  
    
    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style) 
    
    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)  
    
    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)
    
    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results, line_style,line_style2) 
    plot_battery_pack_conditions(results, line_style,line_style2)
    
    # Plot Propeller Conditions 
    plot_propeller_conditions(results, line_style) 
    
    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results, line_style)
    
    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results, line_style)  
     
    return


def load_results(battery_chemistry):
    filename = 'electric_aircraft' + battery_chemistry +  '.res'
    return SUAVE.Input_Output.SUAVE.load(filename)

def save_results(results,battery_chemistry):
    filename = 'electric_aircraft' + battery_chemistry +  '.res'
    SUAVE.Input_Output.SUAVE.archive(results,filename)
    
    return  
if __name__ == '__main__': 
    main()    
    plt.show()
