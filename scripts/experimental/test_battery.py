#test battery.py
# by M Vegh, last modified 1/26/2015
import sys
sys.path.append('../trunk')
import SUAVE
from SUAVE.Components.Energy.Storages.Batteries import Battery
from SUAVE.Attributes import Units
from SUAVE.Methods.Power.Battery.Discharge import datta_discharge
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Structure import Data
from SUAVE.Methods.Power.Battery.Ragone import find_ragone_properties, find_specific_power, find_ragone_optimum
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate, find_total_mass_gain
import numpy as np
def main():
    #size the battery
    Mission_total=SUAVE.Attributes.Missions.Mission()
    Ereq=4000*Units.Wh #required energy for the mission in Joules
   
    Preq=3000. #maximum power requirements for mission in W
    numerics=Data()
    battery_inputs=Data()
    specific_energy_guess=500*Units.Wh/Units.kg
    aircraft    = SUAVE.Vehicle()
    battery_li_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Lithium_Air()
    battery_al_air                = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Aluminum_Air()
    battery_li_air.discharge_model=datta_discharge           #default discharge model, but assign anyway
    battery_li_ion                = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    battery_li_s                  = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Sulfur()
    li_ion_mass                   = 10*Units.kg
    #build numerics
    numerics.integrate_time       = np.array([[0, 0],[0, 1]])
    numerics.differentiate_time   = np.array([[0, 0],[0, 1]])
    
    #build battery_inputs
    battery_inputs.current        =90*Units.amps
    battery_inputs.power_in       =np.array([[Preq/2.] , [Preq]])
    battery_inputs.energy_transfer=-np.array([[1000],[2000]])*Units.Wh
    
    battery_li_ion.inputs=battery_inputs
    #test_initialize_from_energy_and_power(battery_al_air, Ereq, Preq)
    #test_mass_gain(battery_al_air)
    #test_find_ragone_properties(specific_energy_guess,battery_li_s, Ereq,Preq)
    test_find_ragone_optimum(battery_li_ion,Ereq,Preq)
    battery_li_ion.current_energy=[[battery_li_ion.max_energy]]
    battery_li_ion.energy_calc( numerics)
    print battery_li_ion
    #test_initialize_from_mass(battery_li_ion,li_ion_mass)
    
def test_mass_gain(battery):
    print battery
    mass_gain       =find_total_mass_gain(battery)
    print 'mass_gain=', mass_gain
    return
def test_initialize_from_energy_and_power(battery,energy,power):
    initialize_from_energy_and_power(battery, energy, power)
    print battery
    return
def test_find_ragone_properties(specific_energy,battery,energy,power):
    find_ragone_properties( specific_energy, battery, energy,power)
    print battery
    print 'specific_energy (Wh/kg)=',battery.specific_energy/(Units.Wh/Units.kg)
    return
def test_find_ragone_optimum(battery, energy, power):
    find_ragone_optimum(battery,energy,power)
    print battery
    
    print 'specific_energy (Wh/kg)=',battery.specific_energy/(Units.Wh/Units.kg)
    print 'max_energy [W-h]=', battery.max_energy/Units.Wh
    return
def test_initialize_from_mass(battery,mass):
    initialize_from_mass(battery,mass)
    print battery
    return
if __name__ == '__main__':
    main()