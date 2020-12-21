# battery.py 
# Created:  Feb 2015, M. Vegh
# Modified: Dec 2020, M. Clarke   

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys 
sys.path.append('../trunk')

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import odeint 

import SUAVE
from SUAVE.Core                                     import Units, Data 
from SUAVE.Components.Energy.Networks.Battery_Test  import Battery_Test
from SUAVE.Components.Energy.Storages.Batteries     import Battery 
from SUAVE.Methods.Power.Battery.Sizing             import initialize_from_mass, initialize_from_module_packaging 
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass  
from SUAVE.Methods.Power.Battery.Discharge_Models   import datta_discharge
from SUAVE.Methods.Power.Battery.Sizing             import initialize_from_energy_and_power, initialize_from_mass 
from SUAVE.Methods.Power.Battery.Ragone             import find_ragone_properties, find_specific_power, find_ragone_optimum
from SUAVE.Methods.Power.Battery.Variable_Mass      import find_mass_gain_rate, find_total_mass_gain
from SUAVE.Plots.Mission_Plots                      import *  


def main():
    # Lithium Ion Battery Cell Chemistry Test
    Li_Ion_Chemistry_Test()    

    # Generatic Lithium Ion Tests
    generic_Li_Ion_Tests()
    
    return  
# ----------------------------------------------------------------------
#   Lithium Ion Cell Chemistry Tests
# ----------------------------------------------------------------------
def Li_Ion_Chemistry_Test():
    days                  = 1
    battery_chemistry     = ['NCA', 'NMC']
    line_styles           = ['bo-','rs-']
    curr                  =  3 
    battery_capacity_mAh  = 3300  
    temperature           = 27
    temp_guess            = 30   
    time_steps            = 20  
         
    voltage_truth_values       = [3.086112486083514,3.202070175950888]
    temperature_truth_values   = [36.53911956270846,33.695241115713415] 
    
    for i in range(len(battery_chemistry)):
        print('\n' + battery_chemistry[i] + ' Cell Chemistry')
        configs, analyses = full_setup(curr,temperature,battery_chemistry[i],temp_guess,days,battery_capacity_mAh,time_steps)
        analyses.finalize()     
        mission = analyses.missions.base
        results = mission.evaluate() 
        
        # Regression Check 
        volts         = results.segments.battery_discharge.conditions.propulsion.battery_voltage_under_load[-1,0]  
        cell_temp     = results.segments.battery_discharge.conditions.propulsion.battery_cell_temperature[-1,0]
        
        # battery cell voltage check  
        print(volts)
        voltage_diff   = np.abs(volts - voltage_truth_values[i]) 
        print('Voltage difference')
        print(voltage_diff)
        assert np.abs((volts  - voltage_truth_values[i])/voltage_truth_values[i]) < 1e-6
        
        # battery cell temperature check 
        print(cell_temp)
        temp_diff  = np.abs(cell_temp - temperature_truth_values[i])
        print('Temperature difference')
        print(temp_diff)
        assert np.abs((cell_temp - temperature_truth_values[i])/temperature_truth_values[i]) < 1e-6  
        
        # Plot Battery Discharge and Charge
        plot_results(results,line_styles[i])
 
    return 

 
def plot_results(results,lc) :
    
    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results,line_color= lc )
          
    return

 
def full_setup(current,temperature,battery_chemistry,temp_guess,days,battery_capacity_mAh,time_steps):

    # vehicle data
    vehicle  = vehicle_setup(current,temperature,battery_chemistry,battery_capacity_mAh)
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle,battery_chemistry,current,temp_guess,days,battery_capacity_mAh,time_steps)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses 

    return vehicle, analyses



def vehicle_setup(current,temperature,battery_chemistry,battery_capacity_mAh):

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle() 
    vehicle.tag = 'Battery'
    vehicle.mass_properties.takeoff = 3. * Units.kg
    #------------------------------------------------------------------
    # Propulsor
    #------------------------------------------------------------------

    net = Battery_Test()
    net.voltage                     = 4.1   
    net.dischage_model_fidelity     = battery_chemistry

    # Component 8 the Battery
    if battery_chemistry == 'NCA':
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNCA_18650()  
        
    elif battery_chemistry == 'NMC': 
        bat= SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()  
    amp_hour_rating                 = 3.55
    bat.nominal_voltage             = 3.6
    bat.heat_transfer_coefficient   = 7.71 
    watt_hour_rating                = amp_hour_rating * bat.nominal_voltage
    bat.specific_energy             = watt_hour_rating*Units.Wh/bat.mass_properties.mass 
    bat.ambient_temperature         = 27.  
    bat.temperature                 = temperature
    bat.charging_voltage            = bat.nominal_voltage   
    bat.cell.charging_SOC_cutoff    = 1.    
    bat.charging_current            = current  
    bat.max_voltage                 = net.voltage 
    
    initialize_from_mass(bat, bat.mass_properties.mass)
    
    net.battery                   = bat 
    

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
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors 
    analyses.append(energy)

    return analyses    



def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    base_config.propulsors.propulsor.pitch_command = 0 
    configs.append(base_config) 


    # done!
    return configs

def mission_setup(analyses,vehicle,battery_chemistry,current,temp_guess,days,battery_capacity_mAh,ctrl_pts ):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission' 

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points        = ctrl_pts 
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
   
    bat = vehicle.propulsors.propulsor.battery 
    base_segment.use_Jacobian =  True
    base_segment.state.numerics.jacobian_evaluations = 0 
    base_segment.state.numerics.iterations           = 0    
    base_segment.max_energy                       = bat.max_energy
    base_segment.charging_SOC_cutoff              = bat.cell.charging_SOC_cutoff 
    base_segment.charging_current                 = bat.charging_current
    base_segment.charging_voltage                 = bat.charging_voltage 
    base_segment.battery_resistance_growth_factor = 1
    base_segment.battery_capacity_fade_factor     = 1     
    base_segment.battery_configuration            = bat.pack_config
    charge                                        = bat.max_energy/bat.nominal_voltage    
    discharge_time                                = 0.9 * (battery_capacity_mAh/1000)/current * Units.hrs
    
    if battery_chemistry == 'NCA':
        segment     = Segments.Battery_Cell_Testbench.Charge_Discharge_Test(base_segment)
        segment.tag = "Battery Discharge" 
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_linca
        segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_linca      
        segment.state.unknowns.battery_state_of_charge      = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_thevenin_voltage     = 0.1 * ones_row(1) 
        segment.state.unknowns.battery_cell_temperature     = temp_guess  * ones_row(1) 
        segment.state.residuals.network                     = 0.* ones_row(3)  
        segment.time                                        = discharge_time
        segment.battery_discharge                           = True 
        segment.battery_cell_temperature                    = 28. 
        segment.ambient_temperature                         = 27.
        segment.battery_thevenin_voltage                    = 0
        segment.battery_age_in_days                         = days 
        segment.battery_cumulative_charge_throughput         = 0
        segment.battery_energy                              = bat.max_energy * 1.
        mission.append_segment(segment)         
        
        # Charge Model 
        segment     = Segments.Battery_Cell_Testbench.Charge_Discharge_Test(base_segment)     
        segment.tag = 'Battery Charge'  
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_linca
        segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_linca      
        segment.state.unknowns.battery_state_of_charge      = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_thevenin_voltage     = 0.1 * ones_row(1) 
        segment.state.unknowns.battery_cell_temperature     = temp_guess  * ones_row(1) 
        segment.state.residuals.network                     = 0.* ones_row(3)       
        segment.battery_discharge                           = False
        segment.battery_age_in_days                         = days  
        mission.append_segment(segment) 
        
        
    elif battery_chemistry == 'NMC':
        segment     = Segments.Battery_Cell_Testbench.Charge_Discharge_Test(base_segment)
        segment.tag = "Battery Discharge" 
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_linmco
        segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_linmco         
        segment.state.unknowns.battery_state_of_charge      = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_current              = 5 * ones_row(1) 
        segment.state.unknowns.battery_cell_temperature     = temp_guess  * ones_row(1) 
        segment.state.residuals.network                     = 0.* ones_row(3)  
        segment.time                                        = discharge_time
        segment.battery_discharge                           = True 
        segment.battery_cell_temperature                    = 28. 
        segment.ambient_temperature                         = 27.
        segment.battery_thevenin_voltage                    = 0
        segment.battery_age_in_days                         = days 
        segment.battery_cumulative_charge_throughput         = 0
        segment.battery_energy                              = bat.max_energy * 1.
        mission.append_segment(segment)          
    
        # Charge Model 
        segment     = Segments.Battery_Cell_Testbench.Charge_Discharge_Test(base_segment)     
        segment.tag = 'Battery Charge'  
        segment.analyses.extend(analyses.base)     
        segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_linmco
        segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_linmco 
        segment.state.unknowns.battery_state_of_charge      = 0.5 * ones_row(1) 
        segment.state.unknowns.battery_current              = 5 * ones_row(1) 
        segment.state.unknowns.battery_cell_temperature     = temp_guess  * ones_row(1) 
        segment.state.residuals.network                     = 0.* ones_row(3)       
        segment.battery_discharge                           = False
        segment.battery_age_in_days                         = days  
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

# ----------------------------------------------------------------------
#   Generic Lithium Ion Tests
# ---------------------------------------------------------------------- 
def generic_Li_Ion_Tests():
    #size the battery
    Mission_total = SUAVE.Analyses.Mission.Sequential_Segments()
    
    Ereq = 4000*Units.Wh #required energy for the mission in Joules 
    Preq = 3000. # maximum power requirements for mission in W
    
    numerics                      = Data()
    battery_inputs                = Data() #create inputs data structure for inputs for testing discharge model
    specific_energy_guess         = 500*Units.Wh/Units.kg
    battery_li_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Lithium_Air()
    battery_al_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Aluminum_Air()
    battery_li_air.discharge_model= datta_discharge           # default discharge model, but assign anyway
    battery_li_ion_generic        = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    battery_li_s                  = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Sulfur()
    li_ion_mass                   = 10*Units.kg
    
    #build numerics
    numerics.time                 = Data()
    numerics.time.integrate       = np.array([[0, 0],[0, 10]])
    numerics.time.differentiate   = np.array([[0, 0],[0, 1]])
    
    #build battery_inputs(i.e. current it's run at, power, normally done from energy network
    battery_inputs.current        = 90*Units.amps
    battery_inputs.power_in       = np.array([Preq/2. , Preq])
    print('battery_inputs=', battery_inputs)
    battery_li_ion_generic.inputs =battery_inputs
    
    #run tests on functionality
    test_initialize_from_energy_and_power(battery_al_air, Ereq, Preq)
    test_mass_gain(battery_al_air, Preq)
    test_find_ragone_properties(specific_energy_guess,battery_li_s, Ereq,Preq)
    test_find_ragone_optimum(battery_li_ion_generic,Ereq,Preq)
   
    test_initialize_from_mass(battery_li_ion_generic,li_ion_mass)
    #make sure battery starts fully charged
    battery_li_ion_generic.current_energy=[[battery_li_ion_generic.max_energy, battery_li_ion_generic.max_energy]] #normally handle making sure arrays are same length in network
    #run discharge model
    battery_li_ion_generic.temperature = np.array([20,20])
    battery_li_ion_generic.charge_throughput = 0.
    battery_li_ion_generic.R_growth_factor   = 1.
    battery_li_ion_generic.E_growth_factor   = 1.   
    battery_li_ion_generic.energy_discharge(numerics)
    print(battery_li_ion_generic)
    plot_ragone(battery_li_ion_generic, 'lithium ion')
    plot_ragone(battery_li_s,   'lithium sulfur') 
    return 
    
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
    fig       = plt.figure('Ragone Plot')
    title     ='Ragone Plot'
    axes      = fig.add_subplot(1,1,1) 
    esp_plot  = np.linspace(battery.ragone.lower_bound, battery.ragone.upper_bound,50)
    psp_plot  = battery.ragone.const_1*10**(esp_plot*battery.ragone.const_2)
    plt.plot(esp_plot/(Units.Wh/Units.kg),psp_plot/(Units.kW/Units.kg), label=name)
    plt.xlabel('specific energy (W-h/kg)')
    plt.ylabel('specific power (kW/kg)')
    axes.legend(loc='upper right')   
    plt.title(title) 
    return

if __name__ == '__main__':
    main()
    plt.show()