#fuel_cell.py
# by M Vegh, last modified 2/22/2019


#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')
import SUAVE
from SUAVE.Components.Energy.Storages.Batteries import Battery
from SUAVE.Core import Units, Data
from SUAVE.Core import Data
from SUAVE.Methods.Power.Fuel_Cell.Discharge import larminie, setup_larminie, zero_fidelity
from SUAVE.Methods.Power.Fuel_Cell.Sizing import initialize_from_power, initialize_larminie_from_power
import numpy as np
import matplotlib.pyplot as plt


def main():
    numerics                      =Data()
    power                         =np.array([100])
   
    fuel_cell                     = SUAVE.Components.Energy.Converters.Fuel_Cell()
    fuel_cell.inputs.power_in     = power
    
    initialize_from_power(fuel_cell, power*2)
    fuel_cell.discharge_model     = zero_fidelity
    conditions                    = Data()
    #build numerics
    numerics.time                 =Data()
    numerics.time.integrate       = np.array([[0, 0],[0, 10]])
    numerics.time.differentiate   = np.array([[0, 0],[0, 1]])
    
    mdot0       = fuel_cell.energy_calc(conditions, numerics)
    mdot0_truth = 1.0844928369248122e-06
    m0          = fuel_cell.mass_properties.mass
    m0_truth    = 0.09615384615384616
    err_mdot0   = (mdot0 - mdot0_truth)/mdot0_truth
    err_m0      = (m0 - m0_truth)/m0_truth
    
    #try another discharge model
    fuel_cell.discharge_model    = larminie
    #populate fuel cell with required values for this discharge method
    setup_larminie(fuel_cell) 
    initialize_larminie_from_power(fuel_cell,power)
    mdot1          = fuel_cell.energy_calc(conditions, numerics)
    mdot1_truth    = 1.40641687e-06
    err_mdot1      = (mdot1 - mdot1_truth)/mdot1_truth
    
    
    err       = Data()
    err.fuel_cell_mass_error          = err_m0
    err.fuel_cell_fidelity_zero_error = err_mdot0
    err.fuel_cell_larminie_error      = err_mdot1
    for k,v in list(err.items()):
        assert(np.abs(v)<1E-6)    
    print(err)
    
if __name__ == '__main__':
    main()
