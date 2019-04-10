#test battery.py
# by M Vegh, last modified 2/05/2015


#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')
import SUAVE
from SUAVE.Components.Energy.Storages.Batteries import Battery
from SUAVE.Core import Units
from SUAVE.Methods.Power.Battery.Discharge import datta_discharge
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Core import Data
from SUAVE.Methods.Power.Battery.Ragone import find_ragone_properties, find_specific_power, find_ragone_optimum
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate, find_total_mass_gain
import numpy as np
import matplotlib.pyplot as plt


def main():
    #size the battery
    Mission_total=SUAVE.Analyses.Mission.Sequential_Segments()
    Ereq=4000*Units.Wh #required energy for the mission in Joules
   
    Preq=3000. #maximum power requirements for mission in W
    numerics                      =Data()
    battery_inputs                =Data() #create inputs data structure for inputs for testing discharge model
    specific_energy_guess         =500*Units.Wh/Units.kg
    battery_li_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Lithium_Air()
    battery_al_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Aluminum_Air()
    battery_li_air.discharge_model=datta_discharge           #default discharge model, but assign anyway
    battery_li_ion                = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    battery_li_s                  = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Sulfur()
    li_ion_mass                   = 10*Units.kg
    
    #build numerics
    numerics.time                 =Data()
    numerics.time.integrate       = np.array([[0, 0],[0, 10]])
    numerics.time.differentiate   = np.array([[0, 0],[0, 1]])
    
    #build battery_inputs(i.e. current it's run at, power, normally done from energy network
    battery_inputs.current        =90*Units.amps
    battery_inputs.power_in       =np.array([Preq/2. , Preq])
    print('battery_inputs=', battery_inputs)
    battery_li_ion.inputs         =battery_inputs
    
    #run tests on functionality
    test_initialize_from_energy_and_power(battery_al_air, Ereq, Preq)
    test_mass_gain(battery_al_air, Preq)
    test_find_ragone_properties(specific_energy_guess,battery_li_s, Ereq,Preq)
    test_find_ragone_optimum(battery_li_ion,Ereq,Preq)
   
    test_initialize_from_mass(battery_li_ion,li_ion_mass)
    #make sure battery starts fully charged
    battery_li_ion.current_energy=[[battery_li_ion.max_energy, battery_li_ion.max_energy]] #normally handle making sure arrays are same length in network
    #run discharge model
    battery_li_ion.energy_calc(numerics)
    print(battery_li_ion)
    plot_ragone(battery_li_ion, 'lithium ion')
    plot_ragone(battery_li_s,   'lithium sulfur')
    
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
    title='Ragone Plot'
    axes=plt.gca()
    esp_plot=np.linspace(battery.ragone.lower_bound, battery.ragone.upper_bound,50)
    psp_plot=battery.ragone.const_1*10**(esp_plot*battery.ragone.const_2)
    plt.plot(esp_plot/(Units.Wh/Units.kg),psp_plot/(Units.kW/Units.kg), label=name)
    plt.xlabel('specific energy (W-h/kg)')
    plt.ylabel('specific power (kW/kg)')
    axes.legend(loc='upper right')   
    plt.title(title)
    return
if __name__ == '__main__':
    main()
    plt.show()