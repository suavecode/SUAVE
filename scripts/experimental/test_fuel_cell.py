
import sys
sys.path.append('../trunk')
import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Fuel_Cell import Fuel_Cell
from SUAVE.Methods.Power.Fuel_Cell.Sizing.initialize_from_power import initialize_from_power
from SUAVE.Methods.Power.Fuel_Cell.Sizing.initialize_larminie_from_power import initialize_larminie_from_power
from SUAVE.Methods.Power.Fuel_Cell.Chemistry.hydrogen import hydrogen
from SUAVE.Methods.Power.Fuel_Cell.Discharge.setup_larminie import setup_larminie
from SUAVE.Methods.Power.Fuel_Cell.Discharge.find_voltage_larminie import find_voltage_larminie
from SUAVE.Methods.Power.Fuel_Cell.Discharge.find_power_larminie import find_power_larminie
import matplotlib.pyplot as plt
import numpy as np

def main():
    fuel_cell=SUAVE.Components.Energy.Converters.Fuel_Cell()
    max_power=10000.*Units.W
    initialize_from_power(fuel_cell,max_power)
    inputs=Data()
    inputs.power_in=np.linspace(0, max_power, 30)
    fuel_cell.inputs=inputs
    conditions=Data() #not used in zero_fidelity, but may be used in higher fidelity evaluation
    numerics=Data()
    mdot=fuel_cell.energy_calc(conditions,numerics)
    print fuel_cell
    print mdot
    fuel_cell_larminie=SUAVE.Components.Energy.Converters.Fuel_Cell() #higher fidelity model\
    setup_larminie(fuel_cell_larminie)
    initialize_larminie_from_power(fuel_cell_larminie,max_power)
    fuel_cell_larminie.inputs=inputs
    mdot_larminie=fuel_cell_larminie.energy_calc(conditions,numerics)
    print fuel_cell_larminie
    print mdot_larminie
    current_density_vec=np.linspace(.1, 1000, 50)*(Units.mA/(Units.cm**2.))
    i1= current_density_vec/(Units.mA/(Units.cm**2.))
    vvec= find_voltage_larminie(fuel_cell_larminie,current_density_vec)
    pvec=find_power_larminie(current_density_vec,fuel_cell_larminie)
    
    fig, ax1=plt.subplots()
    title='Larminie Discharge Plot'
    ax1.plot(i1,vvec, label='voltage')
    ax1.set_xlabel('current density (mA/cm^2)')
    ax1.set_ylabel('voltage (V)')
    plt.title(title)    
    
    ax2 = ax1.twinx()
    title2='Power Plot'
    ax2.plot(current_density_vec/(Units.mA/(Units.cm**2.)),pvec, 'r', label='Power (W/cell)')
    plt.xlabel('current density (mA/cm^2)')
    ax2.set_ylabel('Power/cell(W)')
    
    '''
    N=3
    thermo=SUAVE.Core.Data()
    air=SUAVE.Attributes.Gases.Air()

    thermo.Tt=np.ones(N)*300.
    thermo.pt=np.ones(N)*100E3
    cp=air.compute_cp(thermo.Tt[0],thermo.pt[0])
    thermo.cp=cp*np.ones(N)
    
    gamma=air.compute_gamma(thermo.Tt[0],thermo.pt[0])
    thermo.gamma=gamma*np.ones(N)
    thermo.ht=np.ones(N)*thermo.cp[0]*thermo.Tt[0]
    maxpower=10000.
    aircraft    = SUAVE.Vehicle()
    compressor=SUAVE.Components.Propulsors.Segments.Compressor()
    compressor.eta_polytropic=[.6]
    compressor.pt_ratio=(2*101.3*10**3)/thermo.pt[0]
    
    mdot=.01;            #assign an overall mass flow rate
    fuel_cell=SUAVE.Components.Energy.Converters.Fuel_Cell()
  
    fuel_cell.active=1
    compressor.active=1
  
    SUAVE.Methods.Power.size_fuel_cell(fuel_cell,maxpower)
    
    power=maxpower/2
    compressor.i=0
    compressor.f=1
    fuel_cell.i=1
    fuel_cell.f=2
    compressor(thermo,power)
    power=power+mdot*(thermo.ht[1]-thermo.ht[0])
    [mdot_h2,mdot_products]=fuel_cell(power, thermo, mdot)
    
    print mdot_products
    print thermo
    
    
    #now run fuel cell without specifying any thermodynamic properties
    [mdot_h2,mdot_products]=fuel_cell(power)
    '''
    return
if __name__ == '__main__':
    main()
    plt.show()