## @ingroup Methods-Power-Turboelectric-Discharge
# zero_fidelity.py
#
# Created : Nov 2019, K. Hamilton

# ----------------------------------------------------------------------
#  Zero Fidelity
#  ----------------------------------------------------------------------

## @ingroup Methods-Power-Turboelectric-Discharge
def zero_fidelity(turboelectric,conditions,numerics):
    '''
    Assumptions:
    constant efficiency
    
    Inputs:
        turboelectric.
            inputs.
                power_in              [W]
            propellant.
                specific_energy       [J/kg]
            efficiency    
    
    Outputs:
        mdot                          [kg/s]
    
    '''
    
    
    power       = turboelectric.inputs.power_in
    
    #mass flow rate of the fuel  
    mdot        = power/(turboelectric.propellant.specific_energy*turboelectric.efficiency)          
  
    return mdot