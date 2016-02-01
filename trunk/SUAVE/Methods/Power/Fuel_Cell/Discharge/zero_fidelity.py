# zero_fidelity.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Sep 2015, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Zero Fidelity
# ----------------------------------------------------------------------

def zero_fidelity(fuel_cell,conditions,numerics):
    
    power       = fuel_cell.inputs.power_in
    
    #mass flow rate of the fuel  
    mdot        = power/(fuel_cell.propellant.specific_energy*fuel_cell.efficiency)                      
  
    return mdot