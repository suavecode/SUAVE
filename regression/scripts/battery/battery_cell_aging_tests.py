# battery_cell_discharge_tests.py 
# 
# Created: Sep 2021, M. Clarke  

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')
import SUAVE   
from SUAVE.Core import Units 
from SUAVE.Methods.Power.Battery.Sizing import  initialize_from_circuit_configuration 
import matplotlib.pyplot as plt
import numpy as np

def main(): 
    curr                  = [1.5, 3, 6, 9 ] 
    temp_guess            = [289 , 289 , 292 , 298 ]  
    mAh                   = 3550  
    discharge_days        = 10
    
    plt.rcParams.update({'font.size': 12})
    fig1 = plt.figure('Cell Comparison') 
    fig1.set_size_inches(12,7)   
    axes1 = fig1.add_subplot(3,4,1)
    axes2 = fig1.add_subplot(3,4,2)    
    axes3 = fig1.add_subplot(3,4,3)
    axes4 = fig1.add_subplot(3,4,4) 
    axes5 = fig1.add_subplot(3,4,5)
    axes6 = fig1.add_subplot(3,4,6) 
    axes7 = fig1.add_subplot(3,4,7)
    axes8 = fig1.add_subplot(3,4,8)  
    axes9 = fig1.add_subplot(3,4,9)
    axes10 = fig1.add_subplot(3,4,10) 
    axes11 = fig1.add_subplot(3,4,11)
    axes12 = fig1.add_subplot(3,4,12)         
    
    for j in range(len(curr)):   
        print("Cell agging at discharge current: " + str(curr[j]))
        configs, analyses = full_setup(curr[j],temp_guess[j],mAh,discharge_days  )
        analyses.finalize()     
        mission = analyses.missions.base
        results = mission.evaluate()  
        plot_results(results ,j, axes1, axes2, axes3, axes4, axes5, axes6, axes7, axes8,axes9,axes10 ,axes11,axes12,discharge_days )  
    
    min_temp = 285
    max_temp = 310
    min_change = 0.95
    max_change = 1.05    
    
    axes1.set_ylabel('Voltage $(V_{UL}$)')     
    axes1.set_ylim([2,4.5])   
    axes1.set_xlabel('Amp-Hours (A-hr)')     
    axes2.set_ylim([2,4.5])    
    axes2.set_xlabel('Amp-Hours (A-hr)')     
    axes3.set_ylim([2,4.5])  
    axes3.set_xlabel('Amp-Hours (A-hr)')      
    axes4.set_ylim([2,4.5])  
    axes4.set_xlabel('Amp-Hours (A-hr)')       
    
    axes5.set_ylabel(r'Temperature ($\degree$C)')    
    axes5.set_xlabel('Amp-Hours (A-hr)')         
    axes5.set_ylim([min_temp,max_temp ]) 
    axes6.set_xlabel('Amp-Hours (A-hr)')      
    axes6.set_ylim([min_temp,max_temp ]) 
    axes7.set_xlabel('Amp-Hours (A-hr)')       
    axes7.set_ylim([min_temp,max_temp ]) 
    axes8.set_xlabel('Amp-Hours (A-hr)')          
    axes8.set_ylim([min_temp,max_temp ])
   
    axes9.set_ylabel(r'Rel. $E_{Fade}$ and $R_{Growth}$')    
    axes9.set_xlabel('Days')         
    axes9.set_ylim([min_change,max_change]) 
    axes10.set_xlabel('Days')      
    axes10.set_ylim([min_change,max_change]) 
    axes11.set_xlabel('Days')       
    axes11.set_ylim([min_change,max_change ]) 
    axes12.set_xlabel('Days')          
    axes12.set_ylim([min_change,max_change])
   
   
    plt.tight_layout()
    
    return 

def plot_results(results ,j,axes1, axes2, axes3, axes4, axes5, axes6, axes7, axes8,axes9,axes10 ,axes11,axes12,discharge_days ): 
    
    C_rat  = [0.5,1,2,3]     
    mark_1 = ['s' ,'s' ,'s' ,'s','s']
    mark_2 = ['o' ,'o' ,'o' ,'o','o']
    mark_3 = ['P' ,'P' ,'P' ,'P','P']
    ls_1   = ['-' ,'-' ,'-' ,'-','-']
    ls_2   = ['--' ,'--' ,'--' ,'--']
    ls_3   = [':' ,':' ,':' ,':']
    lc_1   = ['green' , 'blue' , 'red' , 'orange' ]
    lc_2   = ['darkgreen', 'darkblue' , 'darkred'  ,'brown']
    lc_3   = ['limegreen', 'lightblue' , 'pink'  ,'yellow'] 
    ms = 8
      
    for segment in results.segments.values():
        time          = segment.conditions.frames.inertial.time[:,0]/60 
        volts         = segment.conditions.propulsion.battery_voltage_under_load[:,0]   
        cell_temp     = segment.conditions.propulsion.battery_cell_temperature[:,0]   
        Amp_Hrs       = segment.conditions.propulsion.battery_charge_throughput[:,0]  
        E_fade        = np.array(segment.conditions.propulsion.battery_capacity_fade_factor)
        R_growth      = np.array(segment.conditions.propulsion.battery_resistance_growth_factor)
        cell_age      = np.array(segment.conditions.propulsion.battery_age )
        
        use_amp_hrs = True
        
        if use_amp_hrs == True:
            x_vals = Amp_Hrs
        else:
            x_vals = time                    
         
        if j == 0:   
            axes1.plot(x_vals, volts,  marker= mark_2[j] , linestyle = ls_2[j],  color= lc_2[j] , markersize=ms   )     
            axes5.plot(x_vals, cell_temp ,  marker= mark_2[j], linestyle = ls_2[j],  color= lc_2[j] , markersize=ms)    
            axes9.scatter(cell_age, E_fade  ,  marker= mark_2[j],  c= lc_1[j] , s=ms)  
            axes9.scatter(cell_age, R_growth ,  marker= mark_2[j], c= lc_2[j] , s=ms)  
            
        elif  j == 1:  
            axes2.plot(x_vals, volts,  marker= mark_2[j] , linestyle = ls_2[j],  color= lc_2[j] , markersize=ms   )     
            axes6.plot(x_vals, cell_temp ,  marker= mark_2[j], linestyle = ls_2[j],  color= lc_2[j] , markersize=ms)   
            axes10.scatter(cell_age, E_fade  ,  marker= mark_2[j],c= lc_1[j] , s=ms)  
            axes10.scatter(cell_age, R_growth ,  marker= mark_2[j], c= lc_2[j] ,s=ms)    
            
        elif  j == 2:  
            axes3.plot(x_vals, volts,  marker= mark_2[j] , linestyle = ls_2[j],  color= lc_2[j] , markersize=ms   )     
            axes7.plot(x_vals, cell_temp ,  marker= mark_2[j], linestyle = ls_2[j],  color= lc_2[j] , markersize=ms)     
            axes11.scatter(cell_age, E_fade  ,  marker= mark_2[j],  c= lc_1[j] , s=ms)  
            axes11.scatter(cell_age, R_growth ,  marker= mark_2[j],   c= lc_2[j] , s=ms)               
        elif  j == 3:  

            axes4.plot(x_vals, volts,  marker= mark_2[j] , linestyle = ls_2[j],  color= lc_2[j] , markersize=ms   )     
            axes8.plot(x_vals, cell_temp ,  marker= mark_2[j], linestyle = ls_2[j],  color= lc_2[j] , markersize=ms)    
            axes12.scatter(cell_age, E_fade  ,  marker= mark_2[j],    c= lc_1[j] , s=ms)  
            axes12.scatter(cell_age, R_growth ,  marker= mark_2[j],  c= lc_2[j] , s=ms)  
                           
         
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup(current,temp_guess,mAh,discharge_days ):

    # vehicle data
    vehicle  = vehicle_setup(current)
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle, current,temp_guess,mAh,discharge_days )
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses


    return vehicle, analyses


# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup(current): 

    vehicle                       = SUAVE.Vehicle() 
    vehicle.tag                   = 'battery'  

    net                           = SUAVE.Components.Energy.Networks.Battery_Cell_Cycler()
    net.tag                       ='battery_cell'    

    # Battery    
    bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()  
    initialize_from_circuit_configuration(bat) 
    net.battery                     = bat  
    
    vehicle.mass_properties.takeoff = bat.mass_properties.mass 

    avionics                      = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.current              = current 
    net.avionics                  = avionics  

    vehicle.append_component(net)

    return vehicle

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses

def base_analysis(vehicle):   
    analyses = SUAVE.Analyses.Vehicle() 
    
    # ------------------------------------------------------------------
    #  Energy
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy) 

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)    
    
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



def configs_setup(vehicle): 
    configs         = SUAVE.Components.Configs.Config.Container()  
    base_config     = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base' 
    configs.append(base_config)   
    return configs

def mission_setup(analyses,vehicle, current,temp_guess,mAh,discharge_days):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'  
       
    # unpack Segments module
    Segments     = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment() 
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery   
    base_segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability     = SUAVE.Methods.skip 
    base_segment.process.iterate.conditions.aerodynamics     = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.aerodynamics  = SUAVE.Methods.skip     
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    
   
    bat                                                      = vehicle.networks.battery_cell.battery    
    base_segment.max_energy                                  = bat.max_energy
    base_segment.charging_SOC_cutoff                         = bat.cell.charging_SOC_cutoff 
    base_segment.charging_current                            = bat.charging_current
    base_segment.charging_voltage                            = bat.charging_voltage  
    discharge_time                                           = 1.0 * (mAh/1000)/current * Units.hrs
    
    for day in range(discharge_days):
        segment                                             = Segments.Ground.Battery_Charge_Discharge(base_segment)
        segment.tag                                         = "NMC_Discharge_Day_" + str(day)
        segment.analyses.extend(analyses.base)       
        segment.time                                        = discharge_time  
        if day == 0: 
            segment.battery_energy                          = bat.max_energy * 1.
        segment = vehicle.networks.battery_cell.add_unknowns_and_residuals_to_segment(segment,initial_battery_cell_temperature =temp_guess)    
        mission.append_segment(segment)          
    
        # Charge Model 
        segment                                             = Segments.Ground.Battery_Charge_Discharge(base_segment)     
        segment.tag                                         = "NMC_Charge_Day_" + str(day)
        segment.battery_discharge                           = False 
        segment.analyses.extend(analyses.base)            
        segment = vehicle.networks.battery_cell.add_unknowns_and_residuals_to_segment(segment,initial_battery_cell_temperature =temp_guess)  
        segment.process.finalize.post_process.update_battery_age = SUAVE.Methods.Missions.Segments.Common.Energy.update_battery_age   
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

if __name__ == '__main__':
    main()
    plt.show()