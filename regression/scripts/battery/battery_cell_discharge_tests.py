# battery_cell_discharge_tests.py 
# 
# Created: Sep 2021, M. Clarke  

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')
import SUAVE  
from SUAVE.Core import Units, Data 
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass  
from SUAVE.Components.Energy.Storages.Batteries import Battery
from SUAVE.Core import Units 
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass, initialize_from_circuit_configuration
from SUAVE.Core import Data
from SUAVE.Methods.Power.Battery.Ragone import find_ragone_properties, find_specific_power, find_ragone_optimum
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate, find_total_mass_gain
import numpy as np
import matplotlib.pyplot as plt


def main():
    # size the battery
    Mission_total = SUAVE.Analyses.Mission.Sequential_Segments()
    Ereq          = 4000*Units.Wh # required energy for the mission in Joules 
    Preq          = 3000. # maximum power requirements for mission in W
    
    numerics                      = Data()
    battery_inputs                = Data() #create inputs data structure for inputs for testing discharge model
    specific_energy_guess         = 500*Units.Wh/Units.kg
    battery_li_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Lithium_Air()
    battery_al_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Aluminum_Air()    
    battery_li_ion                = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiFePO4_18650()
    battery_li_s                  = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Sulfur()
    li_ion_mass                   = 10*Units.kg
    
    # build numerics
    numerics.time                 = Data()
    numerics.time.integrate       = np.array([[0, 0],[0, 10]])
    numerics.time.differentiate   = np.array([[0, 0],[0, 1]])
    numerics.time.control_points  = np.array([[0], [1]])
    
    # build battery_inputs(i.e. current it's run at, power, normally done from energy network
    battery_inputs.current        = np.array([[90],[90]])*Units.amps
    battery_inputs.power_in       = np.array([[Preq/2.] ,[ Preq]])
    print('battery_inputs=', battery_inputs)
    battery_li_ion.inputs         = battery_inputs
    battery_li_ion.max_voltage    = battery_li_ion.cell.max_voltage
    
    # run tests on functionality
    test_initialize_from_energy_and_power(battery_al_air, Ereq, Preq)
    test_mass_gain(battery_al_air, Preq)
    test_find_ragone_properties(specific_energy_guess,battery_li_s, Ereq,Preq)
    test_find_ragone_optimum(battery_li_ion,Ereq,Preq)
   
    test_initialize_from_mass(battery_li_ion,li_ion_mass)
    
    # make sure battery starts fully charged
    battery_li_ion.current_energy         = np.array([[battery_li_ion.max_energy], [battery_li_ion.max_energy]]) #normally handle making sure arrays are same length in network
    battery_li_ion.pack_temperature       = np.array([[20],[20]])
    battery_li_ion.cell_charge_throughput = np.array([[0],[0]])
    battery_li_ion.R_growth_factor        = 1
    battery_li_ion.E_growth_factor        = 1 
    
    # run discharge model
    battery_li_ion.energy_calc(numerics)
    print(battery_li_ion)
    plot_ragone(battery_li_ion, 'lithium ion')
    plot_ragone(battery_li_s,   'lithium sulfur') 
     
 
    battery_chemistry     = ['NMC','LFP'] 
    marker                = [['s' ,'s' ,'s' ,'s','s'],['o' ,'o' ,'o' ,'o','o'],['P' ,'P' ,'P' ,'P','P']]
    linestyles            = [['-' ,'-' ,'-' ,'-','-'], ['--' ,'--' ,'--' ,'--'],[':' ,':' ,':' ,':']]
    linecolors            = [['green' , 'blue' , 'red' , 'orange' ],['darkgreen', 'darkblue' , 'darkred'  ,'brown'], ['limegreen', 'lightblue' , 'pink'  ,'yellow']]     
    curr                  = [1.5, 3, 6, 9 ] 
    C_rat                 = [0.5,1,2,3]   
    marker_size           = 8 
    mAh                   = np.array([3550,1500]) 
    V_ul_true  = np.array([[3.9446784139104443,3.5590281079836554],
                           [3.890187851174907,3.546414176818607],
                           [3.698158733645092,3.5211734513546005],
                           [3.5998397257546997,3.4959155684694325]])
    bat_temp_true     = np.array([[290.88689276343854,289.1293674284379],
                                  [295.39761123008685,290.03365301513804],
                                  [307.85017179101123,291.84526567740375],
                                  [319.69372384478675,293.66098235984754]])  
 
    plt.rcParams.update({'font.size': 12})
    fig1 = plt.figure('Cell Comparison') 
    fig1.set_size_inches(12,7)   
    axes1  = fig1.add_subplot(2,4,1)
    axes2  = fig1.add_subplot(2,4,2)    
    axes3  = fig1.add_subplot(2,4,3)
    axes4  = fig1.add_subplot(2,4,4) 
    axes5  = fig1.add_subplot(2,4,5)
    axes6  = fig1.add_subplot(2,4,6) 
    axes7  = fig1.add_subplot(2,4,7)
    axes8  = fig1.add_subplot(2,4,8)  
    
    for j in range(len(curr)):      
        for i in range(len(battery_chemistry)):   
            configs, analyses = full_setup(curr[j],battery_chemistry[i],mAh[i] )
            analyses.finalize()     
            mission = analyses.missions.base
            results = mission.evaluate()   
            
            # Voltage Regression
            V_ul        = results.segments[0].conditions.propulsion.battery_voltage_under_load[2][0]   
            print('Under Load Voltage: ' + str(V_ul))
            V_ul_diff   = np.abs(V_ul - V_ul_true[j,i])
            print('Under Load voltage difference')
            print(V_ul_diff)
            assert np.abs((V_ul_diff)/V_ul_true[j,i]) < 1e-6 
            
            # Temperature Regression
            bat_temp        = results.segments[1].conditions.propulsion.battery_cell_temperature[2][0]  
            print('Cell Temperature: ' + str(bat_temp))
            bat_temp_diff   = np.abs(bat_temp  - bat_temp_true[j,i]) 
            print('Battery temperature difference')
            print(bat_temp_diff)
            assert np.abs((bat_temp_diff)/bat_temp_true[j,i]) < 1e-6    
            
            plot_results(results,j,battery_chemistry[i], axes1, axes2, axes3, axes4, axes5, axes6,
                         axes7, axes8,marker[i][j],marker_size,linecolors[i][j],linestyles[i][j],C_rat[j])  

    legend_font_size = 12                     
    axes1.set_ylabel('Voltage $(V_{UL}$)')    
    axes1.set_xlabel('Amp-Hours (A-hr)') 
    axes1.legend(loc='upper right', prop={'size': legend_font_size})  
    axes1.set_ylim([2,4.5]) 
    axes1.set_xlim([0,7])
    axes2.set_xlabel('Amp-Hours (A-hr)') 
    axes2.legend(loc='upper right', prop={'size': legend_font_size})  
    axes2.set_ylim([2,4.5])   
    axes2.set_xlim([0,7])
    axes3.set_xlabel('Amp-Hours (A-hr)')
    axes3.legend(loc='upper right', prop={'size': legend_font_size})  
    axes3.set_ylim([2,4.5]) 
    axes3.set_xlim([0,7])
    axes4.set_xlabel('Amp-Hours (A-hr)') 
    axes4.legend(loc='upper right', prop={'size': legend_font_size})
    axes4.set_ylim([2,4.5])    
    axes4.set_xlim([0,7]) 
    
    axes5.set_ylabel(r'Temperature ($\degree$C)')    
    axes5.set_xlabel('Amp-Hours (A-hr)')        
    axes5.legend(loc='upper left', prop={'size': legend_font_size})
    axes5.set_ylim([273,320])
    axes5.set_xlim([0,7]) 
    axes6.set_xlabel('Amp-Hours (A-hr)')     
    axes6.legend(loc='upper left', prop={'size': legend_font_size})  
    axes6.set_ylim([273,320])
    axes6.set_xlim([0,7]) 
    axes7.set_xlabel('Amp-Hours (A-hr)')    
    axes7.legend(loc='upper left', prop={'size': legend_font_size})   
    axes7.set_ylim([273,320])
    axes7.set_xlim([0,7]) 
    axes8.set_xlabel('Amp-Hours (A-hr)')    
    axes8.legend(loc='upper left', prop={'size': legend_font_size})      
    axes8.set_ylim([273,320])
    axes8.set_xlim([0,7]) 
    
   
    plt.tight_layout()
    
    return 

def plot_results(results,j,bat_chem, axes1, axes2, axes3, axes4, axes5, axes6, axes7,axes8,m,ms,lc,ls,C_rat): 
    
    for segment in results.segments.values():
        time          = segment.conditions.frames.inertial.time[:,0]/60 
        volts         = segment.conditions.propulsion.battery_voltage_under_load[:,0]   
        cell_temp     = segment.conditions.propulsion.battery_cell_temperature[:,0]   
        Amp_Hrs       = segment.conditions.propulsion.battery_cell_charge_throughput[:,0]   
        
        use_amp_hrs = True
        
        if use_amp_hrs == True:
            x_vals = Amp_Hrs
        else:
            x_vals = time                   
          
        if j == 0:
            axes1.plot(x_vals , volts , marker= m , linestyle = ls,  color= lc , markersize=ms   ,label = bat_chem + ': '+ str(C_rat) + ' C') 
            axes5.plot(x_vals , cell_temp, marker= m , linestyle = ls,  color= lc , markersize=ms,label = bat_chem + ': '+ str(C_rat) + ' C')   
        elif  j == 1: 
            axes2.plot(x_vals , volts , marker= m , linestyle = ls,  color= lc , markersize=ms   ,label = bat_chem + ': '+ str(C_rat) + ' C') 
            axes6.plot(x_vals , cell_temp, marker= m , linestyle = ls,  color= lc , markersize=ms,label = bat_chem + ': '+ str(C_rat) + ' C')                   
        elif  j == 2: 
            axes3.plot(x_vals , volts , marker= m , linestyle = ls,  color= lc , markersize=ms   ,label = bat_chem + ': '+ str(C_rat) + ' C') 
            axes7.plot(x_vals , cell_temp, marker= m , linestyle = ls,  color= lc , markersize=ms,label = bat_chem + ': '+ str(C_rat) + ' C')            
        elif  j == 3: 
            axes4.plot(x_vals , volts , marker= m , linestyle = ls,  color= lc , markersize=ms   ,label = bat_chem + ': '+ str(C_rat) + ' C') 
            axes8.plot(x_vals , cell_temp, marker= m , linestyle = ls,  color= lc , markersize=ms,label = bat_chem + ': '+ str(C_rat) + ' C')    
     
         
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup(current,battery_chemistry,mAh ):

    # vehicle data
    vehicle  = vehicle_setup(current,battery_chemistry)
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle,battery_chemistry,current,mAh )
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses


    return vehicle, analyses


# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup(current,battery_chemistry): 

    vehicle                       = SUAVE.Vehicle() 
    vehicle.tag                   = 'battery'   
    vehicle.reference_area        = 1

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = 0.048 * Units.kg 
    vehicle.mass_properties.max_takeoff     = 0.048 * Units.kg 
    
    # basic parameters
    vehicle.reference_area      = 1.    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   
    wing                         = SUAVE.Components.Wings.Wing()
    wing.tag                     = 'main_wing' 
    wing.areas.reference         = 1.
    wing.spans.projected         = 1.
    wing.aspect_ratio            = 1.
    wing.symmetric               = True
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 1.
    wing.dynamic_pressure_ratio  = 1.
    wing.chords.mean_aerodynamic = 1.
    wing.chords.root             = 1.
    wing.chords.tip              = 1.
    wing.origin                  = [[0.0,0.0,0.0]] # meters
    wing.aerodynamic_center      = [0.0,0.0,0.0] # meters
    
    # add to vehicle
    vehicle.append_component(wing)
     

    net                           = SUAVE.Components.Energy.Networks.Battery_Cell_Cycler()
    net.tag                       ='battery_cell'   
    net.dischage_model_fidelity   = battery_chemistry

    # Battery    
    if battery_chemistry == 'NMC': 
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()
    elif battery_chemistry == 'LFP': 
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiFePO4_18650() 
    bat.charging_voltage            = bat.cell.nominal_voltage    
    bat.charging_current            = current   
    bat.convective_heat_transfer_coefficient = 7.17
    net.voltage                     = bat.cell.nominal_voltage 
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



def configs_setup(vehicle): 
    configs         = SUAVE.Components.Configs.Config.Container()  
    base_config     = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base' 
    configs.append(base_config)   
    return configs

def mission_setup(analyses,vehicle,battery_chemistry,current,mAh):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'  
       
    # unpack Segments module
    Segments     = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                                              = Segments.Segment()
    ones_row                                                                  = base_segment.state.ones_row
    base_segment.process.initialize.initialize_battery                        = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery 
    base_segment.process.finalize.post_process.update_battery_state_of_health = SUAVE.Methods.Missions.Segments.Common.Energy.update_battery_state_of_health    
    
   
    bat                                                      = vehicle.networks.battery_cell.battery    
    base_segment.max_energy                                  = bat.max_energy
    base_segment.charging_SOC_cutoff                         = bat.cell.charging_SOC_cutoff 
    base_segment.charging_current                            = bat.charging_current
    base_segment.charging_voltage                            = bat.charging_voltage 
    discharge_time                                           = 0.9 * (mAh/1000)/current * Units.hrs
    
    if battery_chemistry == 'LFP':
        discharge_tag = 'LFP_Discharge'   
        charge_tag    = 'LFP_Charge'   
    elif battery_chemistry == 'NMC':
        discharge_tag = 'NMC_Discharge'  
        charge_tag    = 'NMC_Charge'  
    
    # Discharge Segment 
    segment                                             = Segments.Ground.Battery_Charge_Discharge(base_segment) 
    segment.analyses.extend(analyses.base)       
    segment.tag                                         = discharge_tag
    segment.time                                        = discharge_time 
    segment.battery_energy                              = bat.max_energy * 1.
    segment = vehicle.networks.battery_cell.add_unknowns_and_residuals_to_segment(segment,initial_battery_cell_temperature = 295 )    
    mission.append_segment(segment)         
    
    # Charge Segment 
    segment                   = Segments.Ground.Battery_Charge_Discharge(base_segment)      
    segment.analyses.extend(analyses.base) 
    segment.tag               = charge_tag
    segment.battery_discharge = False        
    segment = vehicle.networks.battery_cell.add_unknowns_and_residuals_to_segment(segment,initial_battery_cell_temperature = 303 )      
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

def test_mass_gain(battery,power):
    print(battery)
    mass_gain       =find_total_mass_gain(battery)
    print('mass_gain=', mass_gain)
    mdot            =find_mass_gain_rate(battery,power)
    print('mass_gain_rate=', mdot)
    return
def test_initialize_from_energy_and_power(battery,energy,power):
    initialize_from_energy_and_power(battery, energy, power)
    print(battery)
    return
def test_find_ragone_properties(specific_energy,battery,energy,power):
    find_ragone_properties( specific_energy, battery, energy,power)
    print(battery)
    print('specific_energy (Wh/kg)=',battery.specific_energy/(Units.Wh/Units.kg))
    return
def test_find_ragone_optimum(battery, energy, power):
    find_ragone_optimum(battery,energy,power)
    print(battery)
    
    print('specific_energy (Wh/kg)=',battery.specific_energy/(Units.Wh/Units.kg))
    print('max_energy [W-h]=', battery.max_energy/Units.Wh)
    return

def test_initialize_from_mass(battery,mass):
    initialize_from_mass(battery,mass)
    print(battery)
    return
    
def plot_ragone(battery, name):
    title    ='Ragone Plot'
    axes     = plt.gca()
    esp_plot = np.linspace(battery.ragone.lower_bound, battery.ragone.upper_bound,50)
    psp_plot = battery.ragone.const_1*10**(esp_plot*battery.ragone.const_2)
    plt.plot(esp_plot/(Units.Wh/Units.kg),psp_plot/(Units.kW/Units.kg), label=name)
    plt.xlabel('specific energy (W-h/kg)')
    plt.ylabel('specific power (kW/kg)')
    axes.legend(loc='upper right')   
    plt.title(title)
    return

if __name__ == '__main__':
    main()
    plt.show()